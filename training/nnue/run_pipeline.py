#!/usr/bin/env python3
"""End-to-end NNUE bootstrap pipeline: ingest -> train -> export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from . import exporter, process_bins, train_stub
    try:
        from . import train_torch  # type: ignore
    except Exception:
        train_torch = None  # type: ignore
except Exception:
    import exporter  # type: ignore
    import process_bins  # type: ignore
    import train_stub  # type: ignore
    try:
        import train_torch  # type: ignore
    except Exception:
        train_torch = None  # type: ignore


_RELABEL_STAGE_PROVENANCE_VERSION = 2
_RELABEL_SELECTION_POLICY = "per-game-fnv1a-phase-v1"


def _clamp_int(v: float, lo: int, hi: int) -> int:
    iv = int(round(v))
    if iv < lo:
        return lo
    if iv > hi:
        return hi
    return iv


def _quant_i8(vals: Iterable[float]) -> List[int]:
    return [_clamp_int(v, -128, 127) for v in vals]


def _quant_i16(vals: Iterable[float]) -> List[int]:
    return [_clamp_int(v, -32768, 32767) for v in vals]


def _absmax(vals: Iterable[float]) -> float:
    m = 0.0
    for v in vals:
        a = abs(float(v))
        if a > m:
            m = a
    return m


def classifier_head_to_scalar(checkpoint: Dict[str, Any], cp_scale: float = 100.0) -> Tuple[List[float], float]:
    weights = checkpoint.get("weights")
    bias = checkpoint.get("bias")
    input_dim = int(checkpoint.get("input_dim", 0))
    num_classes = int(checkpoint.get("num_classes", 0))
    if not isinstance(weights, list) or len(weights) < 3:
        raise ValueError("checkpoint must contain at least 3 class weight rows")
    if not isinstance(bias, list) or len(bias) < 3:
        raise ValueError("checkpoint must contain 3 class biases")
    if num_classes and num_classes < 3:
        raise ValueError("num_classes must be >= 3")

    loss_row = [float(v) for v in weights[0]]
    win_row = [float(v) for v in weights[2]]
    if input_dim <= 0:
        input_dim = len(win_row)
    if len(loss_row) != input_dim or len(win_row) != input_dim:
        raise ValueError("checkpoint weight row length does not match input_dim")

    scalar_w = [(win_row[i] - loss_row[i]) * cp_scale for i in range(input_dim)]
    scalar_b = (float(bias[2]) - float(bias[0])) * cp_scale
    return scalar_w, scalar_b


def _identity_w1(input_dim: int) -> List[float]:
    # Flattened row-major matrix.
    out = [0.0] * (input_dim * input_dim)
    for i in range(input_dim):
        out[i * input_dim + i] = 1.0
    return out


def _export_v2_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    quant_path: Path,
) -> Dict[str, Any]:
    """Quantize an arch-v2 checkpoint to PIENNQ02.

    Mapping (mirrors the engine's integer head, verified by
    PieBot/tests/nnue_arch_v2.rs): w1_q = round(w1_f * QA) as i16 feature-major,
    b1_q = round(b1_f * QA) as i16, w2_q = round(w2_f * QB) as i8 (training
    clamps w2_f to +/-127/QB), b2_q = round(b2_f * QA^2 * QB) as i32,
    eval_cp = (sum(clamp(acc,0,QA)^2 * w2_q) + b2_q) * SCALE / (QA^2 * QB).
    """
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    qa = int(checkpoint.get("quant_qa", 255))
    qb = int(checkpoint.get("quant_qb", 64))
    scale = int(round(float(checkpoint.get("wdl_scale_cp", 400.0))))
    w1 = checkpoint["w1"]  # row-major [hidden][input]
    b1 = checkpoint["b1"]
    w2 = checkpoint["w2"]  # len 2*hidden, stm half first
    b2 = float(checkpoint["b2"])
    if len(w1) != input_dim * hidden_dim:
        raise ValueError("v2 checkpoint w1 size mismatch")
    if len(b1) != hidden_dim:
        raise ValueError("v2 checkpoint b1 size mismatch")
    if len(w2) != 2 * hidden_dim:
        raise ValueError("v2 checkpoint w2 size mismatch")

    try:
        import numpy as np

        w1_q = np.clip(
            np.rint(
                np.asarray(w1, dtype=np.float64).reshape(hidden_dim, input_dim).T * qa
            ),
            -32768,
            32767,
        ).astype(np.int16)
        b1_q = np.clip(
            np.rint(np.asarray(b1, dtype=np.float64) * qa), -32768, 32767
        ).astype(np.int16)
        w2_q = np.clip(
            np.rint(np.asarray(w2, dtype=np.float64) * qb), -128, 127
        ).astype(np.int8)
    except ImportError:
        w1_q = [0] * (input_dim * hidden_dim)
        for h in range(hidden_dim):
            for i in range(input_dim):
                w1_q[i * hidden_dim + h] = _clamp_int(
                    float(w1[h * input_dim + i]) * qa, -32768, 32767
                )
        b1_q = [_clamp_int(float(v) * qa, -32768, 32767) for v in b1]
        w2_q = [_clamp_int(float(v) * qb, -128, 127) for v in w2]
    b2_q = _clamp_int(b2 * qa * qa * qb, -(2**31), 2**31 - 1)

    quant_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.write_quant_v2(
        str(quant_path),
        per_perspective_input_dim=input_dim,
        hidden_dim=hidden_dim,
        qa=qa,
        qb=qb,
        scale=scale,
        w1=w1_q,
        b1=b1_q,
        w2=w2_q,
        b2=b2_q,
    )
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "arch": "v2",
        "qa": qa,
        "qb": qb,
        "scale": scale,
        "export_mode": "direct-v2",
        "quant_format": "PIENNQ02",
    }


def export_checkpoint_as_nnue(
    checkpoint: Dict[str, Any],
    *,
    dense_path: Path,
    quant_path: Path,
    cp_scale: float = 100.0,
) -> Dict[str, Any]:
    if checkpoint.get("arch") == "v2":
        return _export_v2_checkpoint(checkpoint, quant_path=quant_path)
    if all(k in checkpoint for k in ("w1", "b1", "w2", "b2", "hidden_dim", "input_dim")):
        input_dim = int(checkpoint.get("input_dim", 0))
        hidden_dim = int(checkpoint.get("hidden_dim", 0))
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("checkpoint has invalid dimensions")
        w1 = [float(v) for v in checkpoint["w1"]]
        b1 = [float(v) for v in checkpoint["b1"]]
        w2 = [float(v) for v in checkpoint["w2"]]
        b2 = [float(checkpoint["b2"])]
        if len(w1) != input_dim * hidden_dim:
            raise ValueError("checkpoint w1 size mismatch")
        if len(b1) != hidden_dim:
            raise ValueError("checkpoint b1 size mismatch")
        if len(w2) != hidden_dim:
            raise ValueError("checkpoint w2 size mismatch")
        export_mode = "direct"
    else:
        # Legacy classifier projection path.
        input_dim = int(checkpoint.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError("checkpoint missing positive input_dim")
        scalar_w, scalar_b = classifier_head_to_scalar(checkpoint, cp_scale=cp_scale)
        hidden_dim = input_dim
        w1 = _identity_w1(input_dim)
        b1 = [0.0] * hidden_dim
        w2 = scalar_w
        b2 = [scalar_b]
        export_mode = "projected_classifier"

    dense_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.write_dense_f32(
        str(dense_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )
    # Per-layer scaling keeps int8 ranges tight while preserving approximate float behavior.
    w1_abs = _absmax(w1)
    b1_abs = _absmax(b1)
    w2_abs = _absmax(w2)
    b2_abs = _absmax(b2)
    s1 = max(w1_abs / 127.0, b1_abs / 32767.0, 1e-6)
    s2 = max(w2_abs / 127.0, 1e-6)
    s2 = max(s2, b2_abs / (32767.0 * s1))
    w1_q = [_clamp_int(float(v) / s1, -128, 127) for v in w1]
    b1_q = [_clamp_int(float(v) / s1, -32768, 32767) for v in b1]
    w2_q = [_clamp_int(float(v) / s2, -128, 127) for v in w2]
    b2_q = [_clamp_int(float(v) / (s1 * s2), -32768, 32767) for v in b2]

    exporter.write_quant_simple(
        str(quant_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        w1_scale=s1,
        w2_scale=s2,
        w1=w1_q,
        b1=b1_q,
        w2=w2_q,
        b2=b2_q,
    )
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "cp_scale": cp_scale,
        "mode": export_mode,
        "quant_w1_scale": s1,
        "quant_w2_scale": s2,
    }


def _ingest_bins_to_jsonl(
    *,
    bin_inputs: Sequence[Path],
    jsonl_dir: Path,
    bin_glob: str,
    shard_size: int,
    top_policy: int,
    max_bin_records: int,
) -> int:
    _reset_jsonl_stage(jsonl_dir)
    writer = process_bins.ShardWriter(jsonl_dir, shard_size)
    max_records = max_bin_records if max_bin_records > 0 else None
    try:
        total = process_bins.process_inputs(
            [Path(p) for p in bin_inputs],
            writer,
            bin_glob,
            top_policy,
            max_records,
        )
    finally:
        writer.close()
    _write_jsonl_stage_manifest(jsonl_dir, "bin_ingest")
    return int(total)


def build_selfplay_command(
    *,
    piebot_dir: Path,
    jsonl_out: Path,
    games: int,
    max_plies: int,
    threads: int,
    parallel_games: int,
    depth: int,
    movetime_ms: Optional[int],
    seed: int,
    max_records_per_shard: int,
    use_engine: bool,
    openings: Optional[Path],
    temperature_tau: float,
    temp_cp_scale: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    dirichlet_plies: int,
    temperature_moves: int,
    temperature_tau_final: float,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    resign_cp: float = 900.0,
    resign_plies: int = 8,
    no_resign_fraction: float = 0.15,
    draw_adj_cp: float = 10.0,
    draw_adj_plies: int = 40,
    draw_adj_min_ply: int = 80,
    actor_tt_mb: int = 0,
    policy_node_cap: int = 10_000,
    bestmove_node_cap: int = 20_000,
) -> List[str]:
    cmd: List[str] = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "--bin",
        "selfplay",
        "--",
        "--games",
        str(games),
        "--max-plies",
        str(max_plies),
        "--threads",
        str(threads),
        "--parallel-games",
        str(max(0, int(parallel_games))),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
        "--max-records-per-shard",
        str(max_records_per_shard),
        "--temperature-tau",
        str(temperature_tau),
        "--temp-cp-scale",
        str(temp_cp_scale),
        "--dirichlet-alpha",
        str(dirichlet_alpha),
        "--dirichlet-epsilon",
        str(dirichlet_epsilon),
        "--dirichlet-plies",
        str(dirichlet_plies),
        "--temperature-moves",
        str(temperature_moves),
        "--temperature-tau-final",
        str(temperature_tau_final),
        "--jsonl-out",
        str(jsonl_out),
        "--skip-bin",
    ]
    if use_engine:
        cmd.append("--use-engine")
    if nnue_quant_file is not None:
        cmd.extend(["--nnue-quant-file", str(nnue_quant_file)])
        cmd.extend(
            [
                "--nnue-blend-percent",
                str(max(0, min(100, int(nnue_blend_percent)))),
            ]
        )
    if movetime_ms is not None:
        cmd.extend(["--movetime-ms", str(movetime_ms)])
    if openings is not None:
        cmd.extend(["--openings", str(openings)])
    cmd.extend(
        [
            "--resign-cp",
            str(resign_cp),
            "--resign-plies",
            str(resign_plies),
            "--no-resign-fraction",
            str(no_resign_fraction),
            "--draw-adj-cp",
            str(draw_adj_cp),
            "--draw-adj-plies",
            str(draw_adj_plies),
            "--draw-adj-min-ply",
            str(draw_adj_min_ply),
            "--actor-tt-mb",
            str(actor_tt_mb),
            "--policy-node-cap",
            str(policy_node_cap),
            "--bestmove-node-cap",
            str(bestmove_node_cap),
        ]
    )
    return cmd


def build_relabel_command(
    *,
    piebot_dir: Path,
    jsonl_in: Path,
    jsonl_out: Path,
    depth: int,
    every: int,
    threads: int,
    hash_mb: int,
    max_records: int,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    max_nodes: int = 0,
) -> List[str]:
    cmd: List[str] = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "--bin",
        "relabel_jsonl",
        "--",
        "--input",
        str(jsonl_in),
        "--output",
        str(jsonl_out),
        "--depth",
        str(depth),
        "--every",
        str(every),
        "--threads",
        str(threads),
        "--hash-mb",
        str(hash_mb),
    ]
    if nnue_quant_file is not None:
        cmd.extend(["--nnue-quant-file", str(nnue_quant_file)])
        cmd.extend(
            [
                "--nnue-blend-percent",
                str(max(0, min(100, int(nnue_blend_percent)))),
            ]
        )
    if max_records > 0:
        cmd.extend(["--max-records", str(max_records)])
    if max_nodes > 0:
        cmd.extend(["--max-nodes", str(max_nodes)])
    return cmd


def _generate_selfplay_jsonl(
    *,
    piebot_dir: Path,
    jsonl_out: Path,
    games: int,
    max_plies: int,
    threads: int,
    parallel_games: int,
    depth: int,
    movetime_ms: Optional[int],
    seed: int,
    max_records_per_shard: int,
    use_engine: bool,
    openings: Optional[Path],
    temperature_tau: float,
    temp_cp_scale: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    dirichlet_plies: int,
    temperature_moves: int,
    temperature_tau_final: float,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    resign_cp: float = 900.0,
    resign_plies: int = 8,
    no_resign_fraction: float = 0.15,
    draw_adj_cp: float = 10.0,
    draw_adj_plies: int = 40,
    draw_adj_min_ply: int = 80,
    actor_tt_mb: int = 0,
    policy_node_cap: int = 10_000,
    bestmove_node_cap: int = 20_000,
) -> List[str]:
    jsonl_out.mkdir(parents=True, exist_ok=True)
    cmd = build_selfplay_command(
        actor_tt_mb=actor_tt_mb,
        policy_node_cap=policy_node_cap,
        bestmove_node_cap=bestmove_node_cap,
        resign_cp=resign_cp,
        resign_plies=resign_plies,
        no_resign_fraction=no_resign_fraction,
        draw_adj_cp=draw_adj_cp,
        draw_adj_plies=draw_adj_plies,
        draw_adj_min_ply=draw_adj_min_ply,
        piebot_dir=piebot_dir,
        jsonl_out=jsonl_out,
        games=games,
        max_plies=max_plies,
        threads=threads,
        parallel_games=parallel_games,
        depth=depth,
        movetime_ms=movetime_ms,
        seed=seed,
        max_records_per_shard=max_records_per_shard,
        use_engine=use_engine,
        openings=openings,
        temperature_tau=temperature_tau,
        temp_cp_scale=temp_cp_scale,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        dirichlet_plies=dirichlet_plies,
        temperature_moves=temperature_moves,
        temperature_tau_final=temperature_tau_final,
        nnue_quant_file=nnue_quant_file,
        nnue_blend_percent=nnue_blend_percent,
    )
    subprocess.run(cmd, cwd=str(piebot_dir), check=True)
    return cmd


def _relabel_jsonl(
    *,
    piebot_dir: Path,
    jsonl_in: Path,
    jsonl_out: Path,
    depth: int,
    every: int,
    threads: int,
    hash_mb: int,
    max_records: int,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    max_nodes: int = 0,
) -> List[str]:
    jsonl_out.mkdir(parents=True, exist_ok=True)
    cmd = build_relabel_command(
        piebot_dir=piebot_dir,
        jsonl_in=jsonl_in,
        jsonl_out=jsonl_out,
        depth=depth,
        every=every,
        max_nodes=max_nodes,
        threads=threads,
        hash_mb=hash_mb,
        max_records=max_records,
        nnue_quant_file=nnue_quant_file,
        nnue_blend_percent=nnue_blend_percent,
    )
    subprocess.run(cmd, cwd=str(piebot_dir), check=True)
    return cmd


_JSONL_STAGE_MANIFEST = ".piebot_stage_complete.json"
_MERGED_STAGE_MANIFEST = ".piebot_merge_complete.json"
_TRAIN_STAGE_MANIFEST = ".piebot_train_complete.json"
_EXPORT_STAGE_MANIFEST = ".piebot_export_complete.json"


def _atomic_write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_jsonl_source(path: Path) -> str:
    """Hash JSONL contents together with stable source-relative names."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    digest = hashlib.sha256()
    for file_path in files:
        name = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _artifact_snapshot(base_dir: Path, paths: Sequence[Path]) -> Dict[str, Any]:
    base_resolved = base_dir.resolve()
    files: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"missing artifact: {path}")
        try:
            name = path.resolve().relative_to(base_resolved).as_posix()
        except ValueError:
            name = path.name
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    fingerprint_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "files": files,
        "fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
    }


def _normalized_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _file_content_identity(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"provenance input is not a file: {resolved}")
    return {
        "path": _normalized_path(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _selfplay_stage_provenance(
    *,
    piebot_dir: Path,
    games: int,
    max_plies: int,
    threads: int,
    parallel_games: int,
    depth: int,
    movetime_ms: Optional[int],
    seed: int,
    max_records_per_shard: int,
    use_engine: bool,
    openings: Optional[Path],
    temperature_tau: float,
    temp_cp_scale: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    dirichlet_plies: int,
    temperature_moves: int,
    temperature_tau_final: float,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    resign_cp: float = 900.0,
    resign_plies: int = 8,
    no_resign_fraction: float = 0.15,
    draw_adj_cp: float = 10.0,
    draw_adj_plies: int = 40,
    draw_adj_min_ply: int = 80,
    actor_tt_mb: int = 0,
    policy_node_cap: int = 10_000,
    bestmove_node_cap: int = 20_000,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "generator": "selfplay",
        "piebot_dir": _normalized_path(piebot_dir),
        "args": {
            "games": int(games),
            "max_plies": int(max_plies),
            "threads": int(threads),
            "resign_cp": float(resign_cp),
            "resign_plies": int(resign_plies),
            "no_resign_fraction": float(no_resign_fraction),
            "draw_adj_cp": float(draw_adj_cp),
            "draw_adj_plies": int(draw_adj_plies),
            "draw_adj_min_ply": int(draw_adj_min_ply),
            "actor_tt_mb": int(actor_tt_mb),
            "policy_node_cap": int(policy_node_cap),
            "bestmove_node_cap": int(bestmove_node_cap),
            "parallel_games": int(parallel_games),
            "depth": int(depth),
            "movetime_ms": None if movetime_ms is None else int(movetime_ms),
            "seed": int(seed),
            "max_records_per_shard": int(max_records_per_shard),
            "use_engine": bool(use_engine),
            "temperature_tau": float(temperature_tau),
            "temp_cp_scale": float(temp_cp_scale),
            "dirichlet_alpha": float(dirichlet_alpha),
            "dirichlet_epsilon": float(dirichlet_epsilon),
            "dirichlet_plies": int(dirichlet_plies),
            "temperature_moves": int(temperature_moves),
            "temperature_tau_final": float(temperature_tau_final),
            "nnue_blend_percent": int(nnue_blend_percent),
        },
        "openings": _file_content_identity(openings),
        "nnue_quant_file": _file_content_identity(nnue_quant_file),
    }


def _relabel_stage_provenance(
    *,
    piebot_dir: Path,
    jsonl_in: Path,
    depth: int,
    every: int,
    threads: int,
    hash_mb: int,
    max_records: int,
    nnue_quant_file: Optional[Path],
    nnue_blend_percent: int,
    max_nodes: int = 0,
) -> Dict[str, Any]:
    input_snapshot = _jsonl_stage_snapshot(jsonl_in)
    if not input_snapshot["files"] or int(input_snapshot["records"]) <= 0:
        raise ValueError("relabel input contains no JSONL records")
    args: Dict[str, Any] = {
        "depth": int(depth),
        "every": int(every),
        "threads": int(threads),
        "hash_mb": int(hash_mb),
        "max_records": int(max_records),
        "nnue_blend_percent": int(nnue_blend_percent),
    }
    # Uncapped (0) omits the key so pre-node-cap stage markers keep their identity.
    if max_nodes > 0:
        args["max_nodes"] = int(max_nodes)
    return {
        "version": _RELABEL_STAGE_PROVENANCE_VERSION,
        "generator": "relabel",
        "selection_policy": _RELABEL_SELECTION_POLICY,
        "piebot_dir": _normalized_path(piebot_dir),
        "input": {
            "path": _normalized_path(jsonl_in),
            **input_snapshot,
        },
        "args": args,
        "nnue_quant_file": _file_content_identity(nnue_quant_file),
    }


def _jsonl_stage_snapshot(jsonl_dir: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    total_records = 0
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        digest = hashlib.sha256()
        records = 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    records += 1
        files.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "records": records,
                "sha256": digest.hexdigest(),
            }
        )
        total_records += records
    fingerprint_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files": files,
        "records": total_records,
        "fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
    }


def _write_jsonl_stage_manifest(
    jsonl_dir: Path,
    stage: str,
    *,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = _jsonl_stage_snapshot(jsonl_dir)
    if not snapshot["files"] or int(snapshot["records"]) <= 0:
        raise ValueError(f"{stage} stage produced no JSONL records")
    manifest: Dict[str, Any] = {
        "version": 2 if provenance is not None else 1,
        "stage": str(stage),
        **snapshot,
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    path = jsonl_dir / _JSONL_STAGE_MANIFEST
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return manifest


def _validated_jsonl_stage_manifest(
    jsonl_dir: Path,
    stage: str,
    *,
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    path = jsonl_dir / _JSONL_STAGE_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        version = manifest.get("version")
        if version not in (1, 2) or manifest.get("stage") != str(stage):
            return None
        if expected_provenance is not None:
            if version != 2 or manifest.get("provenance") != expected_provenance:
                return None
        snapshot = _jsonl_stage_snapshot(jsonl_dir)
        if not snapshot["files"] or int(snapshot["records"]) <= 0:
            return None
        for key in ("files", "records", "fingerprint"):
            if manifest.get(key) != snapshot[key]:
                return None
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _jsonl_stage_is_complete(
    jsonl_dir: Path,
    stage: str,
    *,
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> bool:
    return (
        _validated_jsonl_stage_manifest(
            jsonl_dir,
            stage,
            expected_provenance=expected_provenance,
        )
        is not None
    )


def _checkpoint_dimensions(checkpoint: Dict[str, Any]) -> Tuple[int, int, int]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a JSON object")
    input_dim = int(checkpoint.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("checkpoint has invalid input_dim")

    direct_keys = ("w1", "b1", "w2", "b2", "hidden_dim")
    if all(key in checkpoint for key in direct_keys):
        hidden_dim = int(checkpoint.get("hidden_dim", 0))
        if hidden_dim <= 0:
            raise ValueError("checkpoint has invalid hidden_dim")
        w1 = checkpoint.get("w1")
        b1 = checkpoint.get("b1")
        w2 = checkpoint.get("w2")
        if not isinstance(w1, list) or len(w1) != input_dim * hidden_dim:
            raise ValueError("checkpoint w1 size mismatch")
        if not isinstance(b1, list) or len(b1) != hidden_dim:
            raise ValueError("checkpoint b1 size mismatch")
        if not isinstance(w2, list) or len(w2) != hidden_dim:
            raise ValueError("checkpoint w2 size mismatch")
        for label, values in (("w1", w1), ("b1", b1), ("w2", w2)):
            try:
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"checkpoint {label} contains a non-finite value")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"checkpoint {label} contains a non-numeric value") from exc
        if not math.isfinite(float(checkpoint["b2"])):
            raise ValueError("checkpoint b2 contains a non-finite value")
        return input_dim, hidden_dim, 1

    weights = checkpoint.get("weights")
    bias = checkpoint.get("bias")
    if not isinstance(weights, list) or len(weights) < 3:
        raise ValueError("legacy checkpoint is missing class weights")
    if not isinstance(bias, list) or len(bias) < 3:
        raise ValueError("legacy checkpoint is missing class biases")
    for row in (weights[0], weights[2]):
        if not isinstance(row, list) or len(row) != input_dim:
            raise ValueError("legacy checkpoint class weight size mismatch")
        try:
            if not all(math.isfinite(float(value)) for value in row):
                raise ValueError("legacy checkpoint contains a non-finite class weight")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("legacy checkpoint contains a non-numeric class weight") from exc
    if not math.isfinite(float(bias[0])) or not math.isfinite(float(bias[2])):
        raise ValueError("legacy checkpoint contains a non-finite class bias")
    return input_dim, input_dim, 1


def _load_training_artifacts(train_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = json.loads((train_dir / "checkpoint.json").read_text(encoding="utf-8"))
    metrics = json.loads((train_dir / "metrics.json").read_text(encoding="utf-8"))
    _checkpoint_dimensions(checkpoint)
    if not isinstance(metrics, dict):
        raise ValueError("training metrics must be a JSON object")
    train_samples = int(metrics.get("train_samples", -1))
    val_samples = int(metrics.get("val_samples", -1))
    if train_samples <= 0 or val_samples < 0:
        raise ValueError("training metrics have invalid sample counts")
    return checkpoint, metrics


def _validate_training_target_identity(
    checkpoint: Dict[str, Any],
    metrics: Dict[str, Any],
    objective: Dict[str, Any],
) -> None:
    if metrics.get("sampling_schema") != train_stub.SAMPLING_SCHEMA:
        raise ValueError("trainer sampling schema does not match the pipeline")
    if (
        metrics.get("validation_sampling_schema")
        != train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA
    ):
        raise ValueError("trainer validation sampling schema does not match the pipeline")
    if (
        metrics.get("reference_validation_sampling_schema")
        != train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA
    ):
        raise ValueError(
            "trainer reference validation sampling schema does not match the pipeline"
        )
    if (
        metrics.get("checkpoint_selection_schema")
        != train_stub.CHECKPOINT_SELECTION_SCHEMA
    ):
        raise ValueError("trainer checkpoint selection schema does not match the pipeline")
    if (
        metrics.get("reference_validation_max_relative_loss_regression")
        != train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
    ):
        raise ValueError(
            "trainer metrics reference validation guard does not match the pipeline"
        )
    if (
        metrics.get("primary_validation_hash_namespace")
        != train_stub.PRIMARY_VALIDATION_HASH_NAMESPACE
    ):
        raise ValueError(
            "trainer primary validation hash namespace does not match the pipeline"
        )
    if metrics.get("target_schema") != train_stub.TARGET_SCHEMA:
        raise ValueError("trainer target schema does not match the pipeline")
    if metrics.get("objective") != objective:
        raise ValueError("trainer objective metadata does not match the pipeline")
    if checkpoint.get("target_schema") != train_stub.TARGET_SCHEMA:
        raise ValueError("training checkpoint target schema does not match the pipeline")
    if checkpoint.get("objective") != objective:
        raise ValueError("training checkpoint objective does not match the pipeline")
    if (
        checkpoint.get("checkpoint_selection_schema")
        != train_stub.CHECKPOINT_SELECTION_SCHEMA
    ):
        raise ValueError(
            "training checkpoint selection schema does not match the pipeline"
        )
    if (
        checkpoint.get("reference_validation_max_relative_loss_regression")
        != train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
    ):
        raise ValueError(
            "training checkpoint reference validation guard does not match the pipeline"
        )
    if (
        checkpoint.get("primary_validation_hash_namespace")
        != train_stub.PRIMARY_VALIDATION_HASH_NAMESPACE
    ):
        raise ValueError(
            "training checkpoint validation hash namespace does not match the pipeline"
        )


def _validate_validation_source_binding(
    metrics: Dict[str, Any],
    expected: Optional[Dict[str, Any]],
) -> None:
    recorded = metrics.get("validation_source")
    if expected is None:
        if recorded is not None:
            raise ValueError("trainer recorded an unexpected fixed validation source")
        return
    if not isinstance(recorded, dict):
        raise ValueError("trainer did not record fixed validation source provenance")
    if recorded.get("path") != expected.get("path"):
        raise ValueError("trainer validation source path mismatch")
    if recorded.get("sha256") != expected.get("source_sha256"):
        raise ValueError("trainer validation source SHA-256 mismatch")
    try:
        recorded_count = int(recorded.get("records", -1))
        expected_count = int(expected.get("records", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("trainer validation source record count is invalid") from exc
    if recorded_count != expected_count:
        raise ValueError("trainer validation source record count mismatch")


def _training_provenance(
    *,
    train_jsonl_dir: Path,
    trainer_backend: str,
    trainer_device: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    data = _jsonl_stage_snapshot(train_jsonl_dir)
    if not data["files"] or int(data["records"]) <= 0:
        raise ValueError("training input contains no JSONL records")
    return {
        "version": 1,
        "data": data,
        "trainer_backend": str(trainer_backend),
        "trainer_device": str(trainer_device) if trainer_backend == "torch" else None,
        "config": config,
    }


def _initial_checkpoint_provenance(
    checkpoint_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"initial checkpoint does not exist: {path}")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid initial checkpoint JSON: {path}") from exc
    input_dim, hidden_dim, output_dim = _checkpoint_dimensions(checkpoint)
    return {
        "path": path.resolve().as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "format": checkpoint.get("format"),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "feature_set": checkpoint.get("feature_set"),
        "target_schema": checkpoint.get("target_schema"),
        "objective": checkpoint.get("objective"),
    }


def _write_training_stage_manifest(
    train_dir: Path,
    *,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _load_training_artifacts(train_dir)
    artifacts = [train_dir / "checkpoint.json", train_dir / "metrics.json"]
    optimizer_path = train_dir / "optimizer.pt"
    if optimizer_path.is_file():
        artifacts.append(optimizer_path)
    snapshot = _artifact_snapshot(train_dir, artifacts)
    manifest: Dict[str, Any] = {
        "version": 1,
        "stage": "train",
        **snapshot,
        "provenance": provenance,
    }
    _atomic_write_json_file(train_dir / _TRAIN_STAGE_MANIFEST, manifest)
    return manifest


def _validated_training_stage_manifest(
    train_dir: Path,
    *,
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        manifest = json.loads((train_dir / _TRAIN_STAGE_MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("version") != 1 or manifest.get("stage") != "train":
            return None
        if expected_provenance is not None and manifest.get("provenance") != expected_provenance:
            return None
        _load_training_artifacts(train_dir)
        artifacts = [train_dir / "checkpoint.json", train_dir / "metrics.json"]
        optimizer_path = train_dir / "optimizer.pt"
        if optimizer_path.is_file():
            artifacts.append(optimizer_path)
        snapshot = _artifact_snapshot(train_dir, artifacts)
        if manifest.get("files") != snapshot["files"]:
            return None
        if manifest.get("fingerprint") != snapshot["fingerprint"]:
            return None
        return manifest
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _validate_nnue_artifact(
    path: Path,
    *,
    quantized: bool,
    expected_dims: Tuple[int, int, int],
) -> None:
    expected_magic = exporter.Q_MAGIC if quantized else exporter.MAGIC
    header_size = 32 if quantized else 24
    with path.open("rb") as handle:
        header = handle.read(header_size)
    if len(header) != header_size or header[:8] != expected_magic:
        raise ValueError(f"invalid NNUE header: {path}")
    version, input_dim, hidden_dim, output_dim = struct.unpack("<IIII", header[8:24])
    if version != 1 or (input_dim, hidden_dim, output_dim) != expected_dims:
        raise ValueError(f"unexpected NNUE dimensions: {path}")
    if quantized:
        w1_scale, w2_scale = struct.unpack("<ff", header[24:32])
        if not all(math.isfinite(v) and v > 0.0 for v in (w1_scale, w2_scale)):
            raise ValueError(f"invalid NNUE quantization scale: {path}")
        expected_size = (
            32
            + input_dim * hidden_dim
            + 2 * hidden_dim
            + output_dim * hidden_dim
            + 2 * output_dim
        )
    else:
        expected_size = 24 + 4 * (
            input_dim * hidden_dim
            + hidden_dim
            + output_dim * hidden_dim
            + output_dim
        )
    if path.stat().st_size != expected_size:
        raise ValueError(f"truncated or oversized NNUE artifact: {path}")


def _write_export_stage_manifest(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    dense_path: Path,
    quant_path: Path,
    export_info: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_dims = _checkpoint_dimensions(checkpoint)
    _validate_nnue_artifact(dense_path, quantized=False, expected_dims=expected_dims)
    _validate_nnue_artifact(quant_path, quantized=True, expected_dims=expected_dims)
    snapshot = _artifact_snapshot(out_dir, [dense_path, quant_path])
    manifest: Dict[str, Any] = {
        "version": 1,
        "stage": "export",
        **snapshot,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "export_info": export_info,
    }
    _atomic_write_json_file(out_dir / _EXPORT_STAGE_MANIFEST, manifest)
    return manifest


def _validated_export_stage_manifest(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    dense_path: Path,
    quant_path: Path,
    expected_cp_scale: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        manifest = json.loads((out_dir / _EXPORT_STAGE_MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("version") != 1 or manifest.get("stage") != "export":
            return None
        if not isinstance(manifest.get("export_info"), dict):
            return None
        if expected_cp_scale is not None:
            if float(manifest["export_info"].get("cp_scale")) != float(expected_cp_scale):
                return None
        if manifest.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            return None
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_dims = _checkpoint_dimensions(checkpoint)
        _validate_nnue_artifact(dense_path, quantized=False, expected_dims=expected_dims)
        _validate_nnue_artifact(quant_path, quantized=True, expected_dims=expected_dims)
        snapshot = _artifact_snapshot(out_dir, [dense_path, quant_path])
        if manifest.get("files") != snapshot["files"]:
            return None
        if manifest.get("fingerprint") != snapshot["fingerprint"]:
            return None
        return manifest
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, struct.error):
        return None


def _reset_jsonl_stage(jsonl_dir: Path) -> None:
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for path in jsonl_dir.glob("*.jsonl"):
        path.unlink()
    manifest = jsonl_dir / _JSONL_STAGE_MANIFEST
    manifest.unlink(missing_ok=True)
    manifest.with_suffix(manifest.suffix + ".tmp").unlink(missing_ok=True)


def _has_jsonl_files(jsonl_dir: Path) -> bool:
    return jsonl_dir.exists() and any(jsonl_dir.glob("*.jsonl"))


def _is_same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)


def _merged_source_provenance(src_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for src_dir in src_dirs:
        snapshot = _jsonl_stage_snapshot(src_dir)
        if not snapshot["files"] or int(snapshot["records"]) <= 0:
            raise ValueError(f"JSONL source contains no records: {src_dir}")
        try:
            source_path = str(src_dir.resolve())
        except OSError:
            source_path = str(src_dir)
        sources.append({"path": source_path, **snapshot})
    return sources


def _assert_validation_source_disjoint(
    validation_dir: Path,
    training_dirs: Sequence[Path],
) -> None:
    """Reject path, inode, or exact-copy overlap between train and holdout data."""
    validation_dir = Path(validation_dir)
    validation_snapshot = _jsonl_stage_snapshot(validation_dir)
    if not validation_snapshot["files"] or int(validation_snapshot["records"]) <= 0:
        raise ValueError(f"validation JSONL source contains no records: {validation_dir}")

    def file_identities(root: Path) -> set[tuple[int, int]]:
        identities: set[tuple[int, int]] = set()
        for shard in root.glob("*.jsonl"):
            stat = shard.stat()
            identities.add((int(stat.st_dev), int(stat.st_ino)))
        return identities

    def shard_signatures(snapshot: Dict[str, Any]) -> set[tuple[Any, ...]]:
        return {
            (
                item.get("sha256"),
                int(item.get("size", -1)),
                int(item.get("records", -1)),
            )
            for item in snapshot["files"]
        }

    def game_keys(root: Path) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for shard in root.glob("*.jsonl"):
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    run_id = record.get("run_id")
                    game_id = record.get("game_id")
                    if (
                        isinstance(run_id, str)
                        and run_id
                        and isinstance(game_id, str)
                        and game_id
                    ):
                        keys.add((run_id, game_id))
        return keys

    validation_inodes = file_identities(validation_dir)
    validation_signatures = shard_signatures(validation_snapshot)
    validation_game_keys = game_keys(validation_dir)
    for raw_training_dir in training_dirs:
        training_dir = Path(raw_training_dir)
        if not _has_jsonl_files(training_dir):
            continue
        if _is_same_path(validation_dir, training_dir):
            raise ValueError("fixed validation source must be separate from training data")
        if validation_inodes.intersection(file_identities(training_dir)):
            raise ValueError("fixed validation source overlaps training data by file identity")
        training_snapshot = _jsonl_stage_snapshot(training_dir)
        if validation_signatures.intersection(shard_signatures(training_snapshot)):
            raise ValueError("fixed validation source contains a copied shard from training data")
        if validation_game_keys and validation_game_keys.intersection(game_keys(training_dir)):
            raise ValueError("fixed validation source overlaps training game provenance")


def _write_merged_stage_manifest(
    merged_dir: Path,
    *,
    src_dirs: Sequence[Path],
) -> Dict[str, Any]:
    output = _jsonl_stage_snapshot(merged_dir)
    if not output["files"] or int(output["records"]) <= 0:
        raise ValueError("replay merge produced no JSONL records")
    manifest: Dict[str, Any] = {
        "version": 1,
        "stage": "merge",
        "sources": _merged_source_provenance(src_dirs),
        "output": output,
    }
    _atomic_write_json_file(merged_dir / _MERGED_STAGE_MANIFEST, manifest)
    return manifest


def _validated_merged_stage_manifest(
    merged_dir: Path,
    *,
    src_dirs: Sequence[Path],
) -> Optional[Dict[str, Any]]:
    try:
        manifest = json.loads(
            (merged_dir / _MERGED_STAGE_MANIFEST).read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            return None
        if manifest.get("version") != 1 or manifest.get("stage") != "merge":
            return None
        if manifest.get("sources") != _merged_source_provenance(src_dirs):
            return None
        output = _jsonl_stage_snapshot(merged_dir)
        if not output["files"] or int(output["records"]) <= 0:
            return None
        if manifest.get("output") != output:
            return None
        return manifest
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _reset_merged_stage(merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    for path in merged_dir.glob("*.jsonl"):
        path.unlink()
    manifest = merged_dir / _MERGED_STAGE_MANIFEST
    manifest.unlink(missing_ok=True)
    manifest.with_suffix(manifest.suffix + ".tmp").unlink(missing_ok=True)


def _build_training_jsonl_dir(
    *,
    out_dir: Path,
    primary_jsonl_dir: Path,
    replay_jsonl_dirs: Optional[Sequence[Path]],
    resume: bool,
) -> Path:
    replay_dirs = [Path(p) for p in (replay_jsonl_dirs or [])]
    unique_replay: List[Path] = []
    for d in replay_dirs:
        if _is_same_path(d, primary_jsonl_dir):
            continue
        if not d.exists() or not _has_jsonl_files(d):
            continue
        if any(_is_same_path(d, ex) for ex in unique_replay):
            continue
        unique_replay.append(d)

    if not unique_replay:
        return primary_jsonl_dir

    merged_dir = out_dir / "jsonl_train"
    src_dirs = [primary_jsonl_dir] + unique_replay
    if resume and _validated_merged_stage_manifest(merged_dir, src_dirs=src_dirs) is not None:
        return merged_dir
    _reset_merged_stage(merged_dir)

    total = 0
    for src_idx, src_dir in enumerate(src_dirs):
        for shard_idx, src in enumerate(sorted(src_dir.glob("*.jsonl"))):
            dst = merged_dir / f"src{src_idx:02d}_shard{shard_idx:06d}.jsonl"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            total += 1
    if total == 0:
        raise ValueError("no JSONL shards found after replay merge")
    _write_merged_stage_manifest(merged_dir, src_dirs=src_dirs)
    return merged_dir


def _resolve_trainer_backend(requested: str, trainer_device: str = "auto") -> str:
    req = (requested or "stub").strip().lower()
    if req not in {"stub", "torch", "auto"}:
        raise ValueError("trainer_backend must be one of: stub, torch, auto")
    if req == "stub":
        return "stub"
    if req == "torch":
        if train_torch is None:
            raise ValueError("trainer_backend=torch requested but torch backend is unavailable")
        if trainer_device.strip().lower() == "cuda":
            try:
                if not bool(train_torch.cuda_available()):  # type: ignore[union-attr]
                    raise ValueError("trainer_backend=torch with trainer_device=cuda but CUDA is unavailable")
            except AttributeError:
                pass
        return "torch"
    # auto
    if train_torch is not None:
        try:
            wants_cuda = trainer_device.strip().lower() == "cuda"
            if wants_cuda and hasattr(train_torch, "cuda_available"):
                if bool(train_torch.cuda_available()):  # type: ignore[union-attr]
                    return "torch"
            elif bool(train_torch.torch_available()):
                return "torch"
        except Exception:
            pass
    return "stub"


def run_pipeline(
    *,
    out_dir: Path,
    jsonl_dir: Optional[Path] = None,
    bin_inputs: Optional[Sequence[Path]] = None,
    piebot_dir: Optional[Path] = None,
    selfplay_games: int = 0,
    selfplay_max_plies: int = 100,
    selfplay_threads: int = 1,
    selfplay_parallel_games: int = 0,
    selfplay_depth: int = 4,
    selfplay_movetime_ms: Optional[int] = None,
    selfplay_seed: int = 42,
    selfplay_use_engine: bool = True,
    selfplay_openings: Optional[Path] = None,
    selfplay_temperature_tau: float = 1.0,
    selfplay_temp_cp_scale: float = 200.0,
    selfplay_dirichlet_alpha: float = 0.3,
    selfplay_dirichlet_epsilon: float = 0.25,
    selfplay_dirichlet_plies: int = 8,
    selfplay_temperature_moves: int = 20,
    selfplay_temperature_tau_final: float = 0.1,
    selfplay_nnue_quant_file: Optional[Path] = None,
    selfplay_nnue_blend_percent: int = 100,
    selfplay_resign_cp: float = 900.0,
    selfplay_resign_plies: int = 8,
    selfplay_no_resign_fraction: float = 0.15,
    selfplay_draw_adj_cp: float = 10.0,
    selfplay_draw_adj_plies: int = 40,
    selfplay_draw_adj_min_ply: int = 80,
    selfplay_actor_tt_mb: int = 0,
    selfplay_policy_node_cap: int = 10_000,
    selfplay_bestmove_node_cap: int = 20_000,
    replay_jsonl_dirs: Optional[Sequence[Path]] = None,
    teacher_relabel_depth: int = 0,
    teacher_relabel_every: int = 4,
    teacher_relabel_threads: int = 1,
    teacher_relabel_hash_mb: int = 64,
    teacher_relabel_max_records: int = 0,
    teacher_relabel_max_nodes: int = 0,
    teacher_relabel_nnue_quant_file: Optional[Path] = None,
    teacher_relabel_nnue_blend_percent: int = 100,
    bin_glob: str = "*.bin*",
    shard_size: int = 200_000,
    top_policy: int = 8,
    max_bin_records: int = 0,
    batch_size: int = 4096,
    max_samples: int = 200_000,
    epochs: int = 8,
    val_split: float = 0.1,
    learning_rate: float = 0.05,
    hidden_dim: int = 16,
    train_arch: str = "v1",
    target_cp: float = 100.0,
    teacher_mix: float = 0.7,
    max_teacher_cp: float = 1500.0,
    outcome_decay: float = 1.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 5.0,
    primary_sample_fraction: float = 0.5,
    teacher_sample_fraction: float = 0.5,
    min_teacher_depth: int = 0,
    loss_kind: str = "mse",
    huber_delta_cp: float = 100.0,
    wdl_scale_cp: float = 400.0,
    validation_jsonl_dir: Optional[Path] = None,
    validation_require_teacher: bool = False,
    max_validation_samples: int = 100_000,
    validation_seed: int = 20_260_802,
    seed: int = 1,
    cp_scale: float = 100.0,
    dense_name: str = "nnue_dense.nnue",
    quant_name: str = "nnue_quant.nnue",
    resume: bool = False,
    trainer_backend: str = "stub",
    trainer_device: str = "auto",
    initial_checkpoint: Optional[Path] = None,
    initial_checkpoint_weights_only: bool = False,
    initial_optimizer_state: Optional[Path] = None,
    continue_optimizer_state: bool = False,
) -> Dict[str, Any]:
    resolved_backend = _resolve_trainer_backend(
        trainer_backend,
        trainer_device=trainer_device,
    )
    initial_checkpoint_weights_only = bool(initial_checkpoint_weights_only)
    if initial_checkpoint_weights_only:
        if resolved_backend != "torch":
            raise ValueError(
                "initial checkpoint weights-only mode is supported only by the torch trainer"
            )
        if initial_checkpoint is None:
            raise ValueError(
                "initial checkpoint weights-only mode requires an initial checkpoint"
            )
        if initial_optimizer_state is not None or continue_optimizer_state:
            raise ValueError(
                "optimizer continuation cannot be combined with weights-only "
                "initial checkpoint mode"
            )
    if initial_checkpoint is not None and not Path(initial_checkpoint).is_file():
        raise ValueError(f"initial checkpoint does not exist: {initial_checkpoint}")
    resolved_initial_optimizer: Optional[Path] = None
    if resolved_backend == "torch":
        if initial_optimizer_state is not None:
            resolved_initial_optimizer = Path(initial_optimizer_state)
            if not resolved_initial_optimizer.is_file():
                raise ValueError(
                    f"initial optimizer state does not exist: {resolved_initial_optimizer}"
                )
        elif continue_optimizer_state and initial_checkpoint is not None:
            checkpoint_parent = Path(initial_checkpoint)
            if not checkpoint_parent.is_file():
                raise ValueError(
                    f"initial checkpoint does not exist: {checkpoint_parent}"
                )
            sibling = checkpoint_parent.parent / "optimizer.pt"
            if not sibling.is_file():
                raise ValueError(
                    "requested optimizer continuation but sibling optimizer state "
                    f"is missing: {sibling}"
                )
            resolved_initial_optimizer = sibling
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0
    selfplay_cmd: Optional[List[str]] = None
    relabel_cmd: Optional[List[str]] = None
    selfplay_rebuilt = False
    if selfplay_games > 0:
        if jsonl_dir is not None or bin_inputs:
            raise ValueError("selfplay generation cannot be combined with jsonl_dir/bin_inputs")
        if piebot_dir is None:
            piebot_dir = Path(__file__).resolve().parents[2] / "PieBot"
        jsonl_dir = out_dir / "selfplay_jsonl"
        selfplay_provenance = _selfplay_stage_provenance(
            piebot_dir=piebot_dir,
            games=selfplay_games,
            max_plies=selfplay_max_plies,
            threads=selfplay_threads,
            parallel_games=selfplay_parallel_games,
            depth=selfplay_depth,
            movetime_ms=selfplay_movetime_ms,
            seed=selfplay_seed,
            max_records_per_shard=shard_size,
            use_engine=selfplay_use_engine,
            openings=selfplay_openings,
                resign_cp=selfplay_resign_cp,
                resign_plies=selfplay_resign_plies,
                no_resign_fraction=selfplay_no_resign_fraction,
                draw_adj_cp=selfplay_draw_adj_cp,
                draw_adj_plies=selfplay_draw_adj_plies,
                draw_adj_min_ply=selfplay_draw_adj_min_ply,
                actor_tt_mb=selfplay_actor_tt_mb,
                policy_node_cap=selfplay_policy_node_cap,
                bestmove_node_cap=selfplay_bestmove_node_cap,
            temperature_tau=selfplay_temperature_tau,
            temp_cp_scale=selfplay_temp_cp_scale,
            dirichlet_alpha=selfplay_dirichlet_alpha,
            dirichlet_epsilon=selfplay_dirichlet_epsilon,
            dirichlet_plies=selfplay_dirichlet_plies,
            temperature_moves=selfplay_temperature_moves,
            temperature_tau_final=selfplay_temperature_tau_final,
            nnue_quant_file=selfplay_nnue_quant_file,
            nnue_blend_percent=selfplay_nnue_blend_percent,
        )
        selfplay_manifest = (
            _validated_jsonl_stage_manifest(
                jsonl_dir,
                "selfplay",
                expected_provenance=selfplay_provenance,
            )
            if resume
            else None
        )
        if selfplay_manifest is not None:
            selfplay_cmd = None
        else:
            _reset_jsonl_stage(jsonl_dir)
            selfplay_cmd = _generate_selfplay_jsonl(
                piebot_dir=piebot_dir,
                jsonl_out=jsonl_dir,
                games=selfplay_games,
                max_plies=selfplay_max_plies,
                threads=selfplay_threads,
                parallel_games=selfplay_parallel_games,
                depth=selfplay_depth,
                movetime_ms=selfplay_movetime_ms,
                seed=selfplay_seed,
                max_records_per_shard=shard_size,
                use_engine=selfplay_use_engine,
                openings=selfplay_openings,
                resign_cp=selfplay_resign_cp,
                resign_plies=selfplay_resign_plies,
                no_resign_fraction=selfplay_no_resign_fraction,
                draw_adj_cp=selfplay_draw_adj_cp,
                draw_adj_plies=selfplay_draw_adj_plies,
                draw_adj_min_ply=selfplay_draw_adj_min_ply,
                actor_tt_mb=selfplay_actor_tt_mb,
                policy_node_cap=selfplay_policy_node_cap,
                bestmove_node_cap=selfplay_bestmove_node_cap,
                temperature_tau=selfplay_temperature_tau,
                temp_cp_scale=selfplay_temp_cp_scale,
                dirichlet_alpha=selfplay_dirichlet_alpha,
                dirichlet_epsilon=selfplay_dirichlet_epsilon,
                dirichlet_plies=selfplay_dirichlet_plies,
                temperature_moves=selfplay_temperature_moves,
                temperature_tau_final=selfplay_temperature_tau_final,
                nnue_quant_file=selfplay_nnue_quant_file,
                nnue_blend_percent=selfplay_nnue_blend_percent,
            )
            selfplay_manifest = _write_jsonl_stage_manifest(
                jsonl_dir,
                "selfplay",
                provenance=selfplay_provenance,
            )
            selfplay_rebuilt = True
        ingested = int(selfplay_manifest["records"])
    elif jsonl_dir is None:
        if not bin_inputs:
            raise ValueError("provide one of: jsonl_dir, bin_inputs, or selfplay_games>0")
        jsonl_dir = out_dir / "jsonl"
        ingested = _ingest_bins_to_jsonl(
            bin_inputs=[Path(p) for p in bin_inputs],
            jsonl_dir=jsonl_dir,
            bin_glob=bin_glob,
            shard_size=shard_size,
            top_policy=top_policy,
            max_bin_records=max_bin_records,
        )

    if teacher_relabel_depth > 0:
        if piebot_dir is None:
            piebot_dir = Path(__file__).resolve().parents[2] / "PieBot"
        relabeled_dir = out_dir / "jsonl_relabel"
        relabel_provenance = _relabel_stage_provenance(
            piebot_dir=piebot_dir,
            jsonl_in=Path(jsonl_dir),
            depth=teacher_relabel_depth,
            every=teacher_relabel_every,
            threads=teacher_relabel_threads,
            hash_mb=teacher_relabel_hash_mb,
            max_records=teacher_relabel_max_records,
            nnue_quant_file=teacher_relabel_nnue_quant_file,
            nnue_blend_percent=teacher_relabel_nnue_blend_percent,
            max_nodes=teacher_relabel_max_nodes,
        )
        relabel_manifest = (
            _validated_jsonl_stage_manifest(
                relabeled_dir,
                "relabel",
                expected_provenance=relabel_provenance,
            )
            if resume and not selfplay_rebuilt
            else None
        )
        if relabel_manifest is not None:
            relabel_cmd = None
        else:
            _reset_jsonl_stage(relabeled_dir)
            relabel_cmd = _relabel_jsonl(
                piebot_dir=piebot_dir,
                jsonl_in=Path(jsonl_dir),
                jsonl_out=relabeled_dir,
                depth=teacher_relabel_depth,
                every=teacher_relabel_every,
                threads=teacher_relabel_threads,
                hash_mb=teacher_relabel_hash_mb,
                max_records=teacher_relabel_max_records,
                nnue_quant_file=teacher_relabel_nnue_quant_file,
                nnue_blend_percent=teacher_relabel_nnue_blend_percent,
                max_nodes=teacher_relabel_max_nodes,
            )
            relabel_manifest = _write_jsonl_stage_manifest(
                relabeled_dir,
                "relabel",
                provenance=relabel_provenance,
            )
        jsonl_dir = relabeled_dir
        ingested = int(relabel_manifest["records"])

    if validation_jsonl_dir is not None:
        _assert_validation_source_disjoint(
            Path(validation_jsonl_dir),
            [Path(jsonl_dir), *(Path(path) for path in (replay_jsonl_dirs or []))],
        )

    train_jsonl_dir = _build_training_jsonl_dir(
        out_dir=out_dir,
        primary_jsonl_dir=Path(jsonl_dir),
        replay_jsonl_dirs=replay_jsonl_dirs,
        resume=resume,
    )

    train_out = out_dir / "train"
    checkpoint_path = train_out / "checkpoint.json"
    metrics_path = train_out / "metrics.json"
    validation_dataset = None
    if validation_jsonl_dir is not None:
        validation_dataset = _merged_source_provenance(
            [Path(validation_jsonl_dir)]
        )[0]
        validation_dataset["source_sha256"] = _sha256_jsonl_source(
            Path(validation_jsonl_dir)
        )
        validation_snapshot_after_hash = _merged_source_provenance(
            [Path(validation_jsonl_dir)]
        )[0]
        if {
            key: value
            for key, value in validation_dataset.items()
            if key != "source_sha256"
        } != validation_snapshot_after_hash:
            raise ValueError("validation source changed while pipeline was snapshotting it")

    initial_optimizer_info = _file_content_identity(resolved_initial_optimizer)
    objective = train_stub.objective_metadata(
        loss_kind=loss_kind,
        target_cp=target_cp,
        teacher_mix=teacher_mix,
        max_teacher_cp=max_teacher_cp,
        outcome_decay=outcome_decay,
        min_teacher_depth=min_teacher_depth,
        huber_delta_cp=huber_delta_cp,
        wdl_scale_cp=wdl_scale_cp,
    )
    train_arch = str(train_arch).lower()
    if train_arch not in {"v1", "v2"}:
        raise ValueError(f"unsupported train_arch: {train_arch!r}")
    training_config: Dict[str, Any] = {
        "batch_size": batch_size,
        "max_samples": max_samples,
        "epochs": epochs,
        "val_split": val_split,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_eps": adam_eps,
        "grad_clip": grad_clip,
        "primary_sample_fraction": primary_sample_fraction,
        "teacher_sample_fraction": teacher_sample_fraction,
        "min_teacher_depth": min_teacher_depth,
        "loss_kind": loss_kind,
        "huber_delta_cp": huber_delta_cp,
        "wdl_scale_cp": wdl_scale_cp,
        "validation_jsonl_dir": (
            _normalized_path(Path(validation_jsonl_dir))
            if validation_jsonl_dir is not None
            else None
        ),
        "validation_require_teacher": validation_require_teacher,
        "max_validation_samples": max_validation_samples,
        "validation_seed": validation_seed,
        "seed": seed,
    }
    if train_arch != "v1":
        # Conditional key preserves the legacy v1 stage identity byte-for-byte.
        training_config["arch"] = train_arch
    initial_checkpoint_info = _initial_checkpoint_provenance(initial_checkpoint)
    training_provenance_config = {
        **training_config,
        "sampling_schema": train_stub.SAMPLING_SCHEMA,
        "validation_sampling_schema": train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
        "reference_validation_sampling_schema": (
            train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA
        ),
        "checkpoint_selection_schema": train_stub.CHECKPOINT_SELECTION_SCHEMA,
        "reference_validation_max_relative_loss_regression": (
            train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_hash_namespace": (
            train_stub.PRIMARY_VALIDATION_HASH_NAMESPACE
        ),
        "objective": objective,
        "initial_checkpoint": initial_checkpoint_info,
        "initial_checkpoint_weights_only": initial_checkpoint_weights_only,
        "initial_optimizer_state": initial_optimizer_info,
        "validation_dataset": validation_dataset,
    }
    training_provenance = _training_provenance(
        train_jsonl_dir=train_jsonl_dir,
        trainer_backend=resolved_backend,
        trainer_device=trainer_device,
        config=training_provenance_config,
    )
    training_manifest = (
        _validated_training_stage_manifest(
            train_out,
            expected_provenance=training_provenance,
        )
        if resume
        else None
    )
    if training_manifest is not None:
        checkpoint, metrics = _load_training_artifacts(train_out)
    else:
        train_out.mkdir(parents=True, exist_ok=True)
        train_manifest_path = train_out / _TRAIN_STAGE_MANIFEST
        train_manifest_path.unlink(missing_ok=True)
        train_manifest_path.with_suffix(train_manifest_path.suffix + ".tmp").unlink(
            missing_ok=True
        )
        train_kwargs = {
            "jsonl_dir": train_jsonl_dir,
            **training_config,
            "out_dir": train_out,
        }
        if train_arch == "v2" and resolved_backend != "torch":
            raise ValueError("train_arch v2 requires the torch trainer backend")
        if resolved_backend == "torch":
            train_torch.train_model(  # type: ignore[union-attr]
                device=trainer_device,
                initial_checkpoint=initial_checkpoint,
                initial_checkpoint_weights_only=initial_checkpoint_weights_only,
                initial_optimizer_state=resolved_initial_optimizer,
                **train_kwargs,
            )
        else:
            train_stub.train_model(
                initial_checkpoint=initial_checkpoint,
                **train_kwargs,
            )
        checkpoint, metrics = _load_training_artifacts(train_out)
        _write_training_stage_manifest(train_out, provenance=training_provenance)

    _validate_training_target_identity(checkpoint, metrics, objective)
    _validate_validation_source_binding(metrics, validation_dataset)

    if initial_checkpoint_info is not None:
        initialized_from = metrics.get("initialized_from")
        if not isinstance(initialized_from, dict):
            raise ValueError("trainer did not record warm-start checkpoint provenance")
        if initialized_from.get("sha256") != initial_checkpoint_info["sha256"]:
            raise ValueError("trainer warm-start checkpoint SHA-256 mismatch")
        if initialized_from.get("path") != initial_checkpoint_info["path"]:
            raise ValueError("trainer warm-start checkpoint path mismatch")
        if bool(metrics.get("initial_checkpoint_weights_only")) != (
            initial_checkpoint_weights_only
        ):
            raise ValueError("trainer warm-start mode provenance mismatch")
        expected_mode = "weights-only" if initial_checkpoint_weights_only else "strict"
        recorded_mode = initialized_from.get("mode")
        if (
            initial_checkpoint_weights_only
            and recorded_mode != expected_mode
        ) or (
            not initial_checkpoint_weights_only
            and recorded_mode not in {None, expected_mode}
        ):
            raise ValueError("trainer warm-start initialization mode mismatch")
    if initial_optimizer_info is not None:
        initialized_optimizer = metrics.get("initialized_optimizer_state")
        if not isinstance(initialized_optimizer, dict):
            raise ValueError("trainer did not record optimizer-state provenance")
        if initialized_optimizer.get("sha256") != initial_optimizer_info["sha256"]:
            raise ValueError("trainer optimizer-state SHA-256 mismatch")
        if initialized_optimizer.get("path") != initial_optimizer_info["path"]:
            raise ValueError("trainer optimizer-state path mismatch")
        if not bool(metrics.get("optimizer_state_restored")):
            raise ValueError("trainer did not restore the requested optimizer state")

    dense_path = out_dir / dense_name
    quant_path = out_dir / quant_name
    export_manifest = (
        _validated_export_stage_manifest(
            out_dir,
            checkpoint_path=checkpoint_path,
            dense_path=dense_path,
            quant_path=quant_path,
            expected_cp_scale=cp_scale,
        )
        if resume
        else None
    )
    if export_manifest is not None:
        export_info = dict(export_manifest["export_info"])
    else:
        export_manifest_path = out_dir / _EXPORT_STAGE_MANIFEST
        export_manifest_path.unlink(missing_ok=True)
        export_manifest_path.with_suffix(export_manifest_path.suffix + ".tmp").unlink(
            missing_ok=True
        )
        dense_tmp = dense_path.with_name(dense_path.name + ".tmp")
        quant_tmp = quant_path.with_name(quant_path.name + ".tmp")
        dense_tmp.unlink(missing_ok=True)
        quant_tmp.unlink(missing_ok=True)
        try:
            export_info = export_checkpoint_as_nnue(
                checkpoint,
                dense_path=dense_tmp,
                quant_path=quant_tmp,
                cp_scale=cp_scale,
            )
            expected_dims = _checkpoint_dimensions(checkpoint)
            _validate_nnue_artifact(
                dense_tmp,
                quantized=False,
                expected_dims=expected_dims,
            )
            _validate_nnue_artifact(
                quant_tmp,
                quantized=True,
                expected_dims=expected_dims,
            )
            os.replace(dense_tmp, dense_path)
            os.replace(quant_tmp, quant_path)
        finally:
            dense_tmp.unlink(missing_ok=True)
            quant_tmp.unlink(missing_ok=True)
        _write_export_stage_manifest(
            out_dir,
            checkpoint_path=checkpoint_path,
            dense_path=dense_path,
            quant_path=quant_path,
            export_info=export_info,
        )

    summary: Dict[str, Any] = {
        "jsonl_dir": str(Path(jsonl_dir)),
        "train_jsonl_dir": str(train_jsonl_dir),
        "replay_jsonl_dirs": [str(Path(p)) for p in (replay_jsonl_dirs or [])],
        "ingested_records": ingested,
        "selfplay_command": selfplay_cmd,
        "relabel_command": relabel_cmd,
        "selfplay_nnue_quant_file": str(selfplay_nnue_quant_file) if selfplay_nnue_quant_file else None,
        "teacher_relabel_nnue_quant_file": str(teacher_relabel_nnue_quant_file)
        if teacher_relabel_nnue_quant_file
        else None,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "initial_checkpoint": initial_checkpoint_info,
        "initial_checkpoint_weights_only": initial_checkpoint_weights_only,
        "initial_optimizer_state": initial_optimizer_info,
        "validation_dataset": validation_dataset,
        "metrics_path": str(metrics_path),
        "optimizer_path": (
            str(train_out / "optimizer.pt")
            if (train_out / "optimizer.pt").is_file()
            else None
        ),
        "optimizer_sha256": (
            _sha256_file(train_out / "optimizer.pt")
            if (train_out / "optimizer.pt").is_file()
            else None
        ),
        "dense_path": str(dense_path),
        "quant_path": str(quant_path),
        "quant_sha256": _sha256_file(quant_path),
        "trainer_backend": resolved_backend,
        "export": export_info,
        "metrics": metrics,
    }
    _atomic_write_json_file(out_dir / "pipeline_summary.json", summary)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output directory for all artifacts")
    ap.add_argument("--jsonl-dir", type=Path, default=None, help="Existing JSONL shard directory")
    ap.add_argument("--piebot-dir", type=Path, default=None, help="Path to PieBot crate for selfplay generation")
    ap.add_argument("--selfplay-games", type=int, default=0, help="Generate this many selfplay games before training")
    ap.add_argument("--selfplay-max-plies", type=int, default=100)
    ap.add_argument("--selfplay-threads", type=int, default=1)
    ap.add_argument("--selfplay-parallel-games", type=int, default=0)
    ap.add_argument("--selfplay-depth", type=int, default=4)
    ap.add_argument("--selfplay-movetime-ms", type=int, default=None)
    ap.add_argument("--selfplay-seed", type=int, default=42)
    ap.add_argument(
        "--selfplay-use-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--selfplay-openings", type=Path, default=None)
    ap.add_argument("--selfplay-nnue-quant-file", type=Path, default=None)
    ap.add_argument("--selfplay-nnue-blend-percent", type=int, default=100)
    ap.add_argument("--replay-jsonl-dirs", nargs="*", type=Path, default=None)
    ap.add_argument("--selfplay-temperature-tau", type=float, default=1.0)
    ap.add_argument("--selfplay-temp-cp-scale", type=float, default=200.0)
    ap.add_argument("--selfplay-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--selfplay-dirichlet-epsilon", type=float, default=0.25)
    ap.add_argument("--selfplay-dirichlet-plies", type=int, default=8)
    ap.add_argument("--selfplay-temperature-moves", type=int, default=20)
    ap.add_argument("--selfplay-temperature-tau-final", type=float, default=0.1)
    ap.add_argument("--teacher-relabel-depth", type=int, default=0)
    ap.add_argument("--teacher-relabel-every", type=int, default=4)
    ap.add_argument("--teacher-relabel-threads", type=int, default=1)
    ap.add_argument("--teacher-relabel-hash-mb", type=int, default=64)
    ap.add_argument("--teacher-relabel-max-nodes", type=int, default=0)
    ap.add_argument("--teacher-relabel-max-records", type=int, default=0)
    ap.add_argument("--teacher-relabel-nnue-quant-file", type=Path, default=None)
    ap.add_argument("--teacher-relabel-nnue-blend-percent", type=int, default=100)
    ap.add_argument(
        "--bin-inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional BIN files/directories/tars to ingest when --jsonl-dir is not set",
    )
    ap.add_argument("--bin-glob", default="*.bin*", help="Glob used when scanning BIN directories")
    ap.add_argument("--shard-size", type=int, default=200_000, help="JSONL records per shard")
    ap.add_argument("--top-policy", type=int, default=8, help="Top policy entries per LC0 sample")
    ap.add_argument("--max-bin-records", type=int, default=0, help="Cap BIN ingest records (0=unlimited)")

    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Compatible float checkpoint used to initialize Torch training",
    )
    ap.add_argument(
        "--initial-checkpoint-weights-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "load only model weights for an explicit objective transition; "
            "cannot be combined with optimizer continuation"
        ),
    )
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--train-arch", choices=("v1", "v2"), default="v1")
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--primary-sample-fraction", type=float, default=0.5)
    ap.add_argument("--teacher-sample-fraction", type=float, default=0.5)
    ap.add_argument("--min-teacher-depth", type=int, default=0)
    ap.add_argument("--loss-kind", choices=["mse", "huber", "wdl"], default="mse")
    ap.add_argument("--huber-delta-cp", type=float, default=100.0)
    ap.add_argument("--wdl-scale-cp", type=float, default=400.0)
    ap.add_argument("--validation-jsonl-dir", type=Path, default=None)
    ap.add_argument(
        "--validation-require-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--max-validation-samples", type=int, default=100_000)
    ap.add_argument("--validation-seed", type=int, default=20_260_802)
    ap.add_argument(
        "--initial-optimizer-state",
        type=Path,
        default=None,
        help="Compatible Torch optimizer.pt used to continue Adam moments",
    )
    ap.add_argument(
        "--continue-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use optimizer.pt beside --initial-checkpoint when available",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--trainer-backend",
        default="stub",
        choices=["stub", "torch", "auto"],
        help="Training backend",
    )
    ap.add_argument(
        "--trainer-device",
        default="auto",
        help="Trainer device for torch backend: auto|cuda|cpu",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume an interrupted pipeline by reusing existing stage artifacts",
    )
    ap.add_argument("--cp-scale", type=float, default=100.0)
    ap.add_argument("--dense-name", default="nnue_dense.nnue")
    ap.add_argument("--quant-name", default="nnue_quant.nnue")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_pipeline(
        out_dir=args.out,
        jsonl_dir=args.jsonl_dir,
        bin_inputs=args.bin_inputs,
        piebot_dir=args.piebot_dir,
        selfplay_games=args.selfplay_games,
        selfplay_max_plies=args.selfplay_max_plies,
        selfplay_threads=args.selfplay_threads,
        selfplay_parallel_games=args.selfplay_parallel_games,
        selfplay_depth=args.selfplay_depth,
        selfplay_movetime_ms=args.selfplay_movetime_ms,
        selfplay_seed=args.selfplay_seed,
        selfplay_use_engine=args.selfplay_use_engine,
        selfplay_openings=args.selfplay_openings,
        selfplay_nnue_quant_file=args.selfplay_nnue_quant_file,
        selfplay_nnue_blend_percent=args.selfplay_nnue_blend_percent,
        replay_jsonl_dirs=args.replay_jsonl_dirs,
        selfplay_temperature_tau=args.selfplay_temperature_tau,
        selfplay_temp_cp_scale=args.selfplay_temp_cp_scale,
        selfplay_dirichlet_alpha=args.selfplay_dirichlet_alpha,
        selfplay_dirichlet_epsilon=args.selfplay_dirichlet_epsilon,
        selfplay_dirichlet_plies=args.selfplay_dirichlet_plies,
        selfplay_temperature_moves=args.selfplay_temperature_moves,
        selfplay_temperature_tau_final=args.selfplay_temperature_tau_final,
        teacher_relabel_depth=args.teacher_relabel_depth,
        teacher_relabel_every=args.teacher_relabel_every,
        teacher_relabel_threads=args.teacher_relabel_threads,
        teacher_relabel_hash_mb=args.teacher_relabel_hash_mb,
        teacher_relabel_max_nodes=args.teacher_relabel_max_nodes,
        teacher_relabel_max_records=args.teacher_relabel_max_records,
        teacher_relabel_nnue_quant_file=args.teacher_relabel_nnue_quant_file,
        teacher_relabel_nnue_blend_percent=args.teacher_relabel_nnue_blend_percent,
        bin_glob=args.bin_glob,
        shard_size=args.shard_size,
        top_policy=args.top_policy,
        max_bin_records=args.max_bin_records,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        train_arch=args.train_arch,
        target_cp=args.target_cp,
        teacher_mix=args.teacher_mix,
        max_teacher_cp=args.max_teacher_cp,
        outcome_decay=args.outcome_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        primary_sample_fraction=args.primary_sample_fraction,
        teacher_sample_fraction=args.teacher_sample_fraction,
        min_teacher_depth=args.min_teacher_depth,
        loss_kind=args.loss_kind,
        huber_delta_cp=args.huber_delta_cp,
        wdl_scale_cp=args.wdl_scale_cp,
        validation_jsonl_dir=args.validation_jsonl_dir,
        validation_require_teacher=args.validation_require_teacher,
        max_validation_samples=args.max_validation_samples,
        validation_seed=args.validation_seed,
        seed=args.seed,
        trainer_backend=args.trainer_backend,
        trainer_device=args.trainer_device,
        initial_checkpoint=args.initial_checkpoint,
        initial_checkpoint_weights_only=args.initial_checkpoint_weights_only,
        initial_optimizer_state=args.initial_optimizer_state,
        continue_optimizer_state=args.continue_optimizer_state,
        resume=args.resume,
        cp_scale=args.cp_scale,
        dense_name=args.dense_name,
        quant_name=args.quant_name,
    )
    print(f"JSONL dir: {summary['jsonl_dir']}")
    print(f"Dense NNUE: {summary['dense_path']}")
    print(f"Quant NNUE: {summary['quant_path']}")
    print(f"Summary: {(Path(args.out) / 'pipeline_summary.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
