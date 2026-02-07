#!/usr/bin/env python3
"""End-to-end NNUE bootstrap pipeline: ingest -> train -> export."""

from __future__ import annotations

import argparse
import json
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


def export_checkpoint_as_nnue(
    checkpoint: Dict[str, Any],
    *,
    dense_path: Path,
    quant_path: Path,
    cp_scale: float = 100.0,
) -> Dict[str, Any]:
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
    return int(total)


def build_selfplay_command(
    *,
    piebot_dir: Path,
    jsonl_out: Path,
    games: int,
    max_plies: int,
    threads: int,
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
) -> List[str]:
    cmd: List[str] = [
        "cargo",
        "run",
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
    if movetime_ms is not None:
        cmd.extend(["--movetime-ms", str(movetime_ms)])
    if openings is not None:
        cmd.extend(["--openings", str(openings)])
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
) -> List[str]:
    cmd: List[str] = [
        "cargo",
        "run",
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
    if max_records > 0:
        cmd.extend(["--max-records", str(max_records)])
    return cmd


def _generate_selfplay_jsonl(
    *,
    piebot_dir: Path,
    jsonl_out: Path,
    games: int,
    max_plies: int,
    threads: int,
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
) -> List[str]:
    jsonl_out.mkdir(parents=True, exist_ok=True)
    cmd = build_selfplay_command(
        piebot_dir=piebot_dir,
        jsonl_out=jsonl_out,
        games=games,
        max_plies=max_plies,
        threads=threads,
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
) -> List[str]:
    jsonl_out.mkdir(parents=True, exist_ok=True)
    cmd = build_relabel_command(
        piebot_dir=piebot_dir,
        jsonl_in=jsonl_in,
        jsonl_out=jsonl_out,
        depth=depth,
        every=every,
        threads=threads,
        hash_mb=hash_mb,
        max_records=max_records,
    )
    subprocess.run(cmd, cwd=str(piebot_dir), check=True)
    return cmd


def _count_jsonl_records(jsonl_dir: Path) -> int:
    total = 0
    for p in sorted(jsonl_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def _has_jsonl_files(jsonl_dir: Path) -> bool:
    return jsonl_dir.exists() and any(jsonl_dir.glob("*.jsonl"))


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
    teacher_relabel_depth: int = 0,
    teacher_relabel_every: int = 4,
    teacher_relabel_threads: int = 1,
    teacher_relabel_hash_mb: int = 64,
    teacher_relabel_max_records: int = 0,
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
    target_cp: float = 100.0,
    teacher_mix: float = 0.7,
    max_teacher_cp: float = 1500.0,
    outcome_decay: float = 1.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 5.0,
    seed: int = 1,
    cp_scale: float = 100.0,
    dense_name: str = "nnue_dense.nnue",
    quant_name: str = "nnue_quant.nnue",
    resume: bool = False,
    trainer_backend: str = "stub",
    trainer_device: str = "auto",
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0
    selfplay_cmd: Optional[List[str]] = None
    relabel_cmd: Optional[List[str]] = None
    if selfplay_games > 0:
        if jsonl_dir is not None or bin_inputs:
            raise ValueError("selfplay generation cannot be combined with jsonl_dir/bin_inputs")
        if piebot_dir is None:
            piebot_dir = Path(__file__).resolve().parents[2] / "PieBot"
        jsonl_dir = out_dir / "selfplay_jsonl"
        if resume and _has_jsonl_files(jsonl_dir):
            selfplay_cmd = None
        else:
            selfplay_cmd = _generate_selfplay_jsonl(
                piebot_dir=piebot_dir,
                jsonl_out=jsonl_dir,
                games=selfplay_games,
                max_plies=selfplay_max_plies,
                threads=selfplay_threads,
                depth=selfplay_depth,
                movetime_ms=selfplay_movetime_ms,
                seed=selfplay_seed,
                max_records_per_shard=shard_size,
                use_engine=selfplay_use_engine,
                openings=selfplay_openings,
                temperature_tau=selfplay_temperature_tau,
                temp_cp_scale=selfplay_temp_cp_scale,
                dirichlet_alpha=selfplay_dirichlet_alpha,
                dirichlet_epsilon=selfplay_dirichlet_epsilon,
                dirichlet_plies=selfplay_dirichlet_plies,
                temperature_moves=selfplay_temperature_moves,
                temperature_tau_final=selfplay_temperature_tau_final,
            )
        ingested = _count_jsonl_records(jsonl_dir)
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
        if resume and _has_jsonl_files(relabeled_dir):
            relabel_cmd = None
        else:
            relabel_cmd = _relabel_jsonl(
                piebot_dir=piebot_dir,
                jsonl_in=Path(jsonl_dir),
                jsonl_out=relabeled_dir,
                depth=teacher_relabel_depth,
                every=teacher_relabel_every,
                threads=teacher_relabel_threads,
                hash_mb=teacher_relabel_hash_mb,
                max_records=teacher_relabel_max_records,
            )
        jsonl_dir = relabeled_dir
        ingested = _count_jsonl_records(jsonl_dir)

    train_out = out_dir / "train"
    checkpoint_path = train_out / "checkpoint.json"
    metrics_path = train_out / "metrics.json"
    resolved_backend = _resolve_trainer_backend(trainer_backend, trainer_device=trainer_device)
    if resume and checkpoint_path.exists() and metrics_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        train_kwargs = dict(
            jsonl_dir=Path(jsonl_dir),
            batch_size=batch_size,
            max_samples=max_samples,
            epochs=epochs,
            val_split=val_split,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            target_cp=target_cp,
            teacher_mix=teacher_mix,
            max_teacher_cp=max_teacher_cp,
            outcome_decay=outcome_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
            grad_clip=grad_clip,
            seed=seed,
            out_dir=train_out,
        )
        if resolved_backend == "torch":
            metrics = train_torch.train_model(  # type: ignore[union-attr]
                device=trainer_device,
                **train_kwargs,
            )
        else:
            metrics = train_stub.train_model(**train_kwargs)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    dense_path = out_dir / dense_name
    quant_path = out_dir / quant_name
    if resume and dense_path.exists() and quant_path.exists():
        old_summary_path = out_dir / "pipeline_summary.json"
        export_info = {}
        if old_summary_path.exists():
            try:
                old = json.loads(old_summary_path.read_text(encoding="utf-8"))
                if isinstance(old.get("export"), dict):
                    export_info = old["export"]
            except Exception:
                export_info = {}
        if not export_info:
            export_info = {
                "input_dim": int(checkpoint.get("input_dim", 0)),
                "hidden_dim": int(checkpoint.get("hidden_dim", 0)),
                "cp_scale": cp_scale,
                "mode": "existing",
            }
    else:
        export_info = export_checkpoint_as_nnue(
            checkpoint,
            dense_path=dense_path,
            quant_path=quant_path,
            cp_scale=cp_scale,
        )

    summary: Dict[str, Any] = {
        "jsonl_dir": str(Path(jsonl_dir)),
        "ingested_records": ingested,
        "selfplay_command": selfplay_cmd,
        "relabel_command": relabel_cmd,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "dense_path": str(dense_path),
        "quant_path": str(quant_path),
        "trainer_backend": resolved_backend,
        "export": export_info,
        "metrics": metrics,
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output directory for all artifacts")
    ap.add_argument("--jsonl-dir", type=Path, default=None, help="Existing JSONL shard directory")
    ap.add_argument("--piebot-dir", type=Path, default=None, help="Path to PieBot crate for selfplay generation")
    ap.add_argument("--selfplay-games", type=int, default=0, help="Generate this many selfplay games before training")
    ap.add_argument("--selfplay-max-plies", type=int, default=100)
    ap.add_argument("--selfplay-threads", type=int, default=1)
    ap.add_argument("--selfplay-depth", type=int, default=4)
    ap.add_argument("--selfplay-movetime-ms", type=int, default=None)
    ap.add_argument("--selfplay-seed", type=int, default=42)
    ap.add_argument(
        "--selfplay-use-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--selfplay-openings", type=Path, default=None)
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
    ap.add_argument("--teacher-relabel-max-records", type=int, default=0)
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
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
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
        selfplay_depth=args.selfplay_depth,
        selfplay_movetime_ms=args.selfplay_movetime_ms,
        selfplay_seed=args.selfplay_seed,
        selfplay_use_engine=args.selfplay_use_engine,
        selfplay_openings=args.selfplay_openings,
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
        teacher_relabel_max_records=args.teacher_relabel_max_records,
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
        target_cp=args.target_cp,
        teacher_mix=args.teacher_mix,
        max_teacher_cp=args.max_teacher_cp,
        outcome_decay=args.outcome_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        trainer_backend=args.trainer_backend,
        trainer_device=args.trainer_device,
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
