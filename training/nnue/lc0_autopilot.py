#!/usr/bin/env python3
"""Resumable, bounded-memory LCZero corpus training with existing PieBot gates.

The corpus is prepared separately by lc0_corpus. This controller never generates
training games or edits the archived self-play campaign. Its state is deliberately
separate from autopilot_state.json: a completed chunk advances the learner, and
only the established paired gameplay gate advances the accepted model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import autopilot, run_pipeline

SCHEMA = "piebot-lc0-autopilot-v1"
CHUNK_LIMIT = 700_000
VALIDATION_LIMIT = 100_000
DAY = 86_400


def _sha(path: Path) -> str:
    return autopilot._sha256_file(path)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-active-model", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--piebot-dir", type=Path, default=Path("PieBot"))
    parser.add_argument("--hours", type=float, default=336)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gate-games", type=int, default=400, choices=(0, 400),
                        help="0 is a promotion-ineligible smoke run")
    parser.add_argument("--gate-parallel-games", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Stop after this many additional chunks; 0 runs until deadline")
    parser.add_argument("--disk-reserve-gib", type=float, default=50)
    return parser.parse_args(argv)


def load_corpus(manifest_path: Path, *, verify_chunks: bool = True) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "piebot-lc0-corpus-v1" or manifest.get("complete") is not True:
        raise ValueError("a complete piebot-lc0-corpus-v1 manifest is required")
    if not manifest.get("corpus_id") or not manifest.get("chunks"):
        raise ValueError("corpus identity and training chunks are required")
    seen = set()
    for row in list(manifest["chunks"]) + [manifest["validation"]]:
        path = Path(row["path"])
        path = (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        row["path"] = str(path)
        if path in seen:
            raise ValueError("corpus training and validation paths must be disjoint and unique")
        seen.add(path)
        limit = VALIDATION_LIMIT if row is manifest["validation"] else CHUNK_LIMIT
        if not 0 < int(row["positions"]) <= limit:
            raise ValueError(f"invalid bounded corpus position count: {path}")
        if not path.is_file():
            raise ValueError(f"missing corpus artifact: {path}")
        if (verify_chunks or row is manifest["validation"]) and _sha(path) != row["sha256"]:
            raise ValueError(f"corpus checksum mismatch: {path}")
    if any(not row["path"].endswith(".jsonl.gz") for row in manifest["chunks"]):
        raise ValueError("training chunks must be compressed .jsonl.gz files")
    if not manifest["validation"]["path"].endswith(".jsonl"):
        raise ValueError("fixed validation must be an uncompressed .jsonl file")
    return manifest


def _source_identity(commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be an explicit 40-character SHA")
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != commit:
        raise ValueError(f"source commit mismatch: requested {commit}, found {actual}")
    digest = hashlib.sha256()
    # Include working-tree contents as well as HEAD: a deployed source archive
    # must never silently resume a lineage with edited trainer or search code.
    for path in sorted(list((repo / "training" / "nnue").rglob("*.py"))
                       + list((repo / "PieBot" / "src").rglob("*.rs"))):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(bytes.fromhex(_sha(path)))
    return {"commit": commit, "digest": digest.hexdigest()}


def _identity(args, corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": _source_identity(args.source_commit),
        "corpus_id": corpus["corpus_id"],
        "manifest_sha256": _sha(args.corpus_manifest),
        "initial_checkpoint": {"path": str(args.initial_checkpoint.resolve()), "sha256": _sha(args.initial_checkpoint)},
        "initial_active_model": {"path": str(args.initial_active_model.resolve()), "sha256": _sha(args.initial_active_model)},
        "arch": "v2", "hidden_dim": 1024, "target_mode": "lc0-q-outcome",
        "teacher_mix": 0.8, "cp_loss_weight": 0.0, "wdl_scale_cp": 400,
        "checkpoint_selection": "latest", "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "hours": args.hours, "gate_games": args.gate_games,
        "gate_parallel_games": args.gate_parallel_games,
        "gate_confirmation_games": 1000, "blend_percent": 75,
        "validation_sha256": corpus["validation"]["sha256"],
    }


def _order(count: int, seed: int, pass_number: int) -> list[int]:
    order = list(range(count))
    random.Random(f"piebot-lc0-pass-v1:{seed}:{pass_number}").shuffle(order)
    return order


def _check_disk(root: Path, reserve_gib: float) -> None:
    if shutil.disk_usage(root).free < reserve_gib * (1024 ** 3):
        # Only the expanded training cache is reproducible and unprotected.
        shutil.rmtree(root / "cache" / "training", ignore_errors=True)
        if shutil.disk_usage(root).free < reserve_gib * (1024 ** 3):
            raise RuntimeError(f"disk reserve below {reserve_gib:g} GiB; protected corpus/checkpoints retained")


def _expand_chunk(chunk: dict[str, Any], destination: Path) -> Path:
    source = Path(chunk["path"])
    if _sha(source) != chunk["sha256"]:
        raise ValueError(f"corpus checksum mismatch: {source}")
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    output = destination / "train.jsonl"
    count = 0
    with gzip.open(source, "rb") as src, output.open("wb") as dst:
        for line in src:
            if not line.strip():
                continue
            count += 1
            if count > CHUNK_LIMIT:
                raise ValueError("expanded training chunk exceeds bounded position limit")
            dst.write(line)
    if count != chunk["positions"]:
        raise ValueError(f"expanded corpus position count mismatch: {count} != {chunk['positions']}")
    return destination


def _train(**kwargs):
    # Keep torch optional for state/manifest checks and operational tests.
    from . import train_torch
    return train_torch.train_model(**kwargs)


def _retain(root: Path, state: dict[str, Any]) -> None:
    recent = state["history"][-2:]
    keep = {Path(item["directory"]).resolve() for item in recent}
    if state.get("best_checkpoint_path"):
        keep.add(Path(state["best_checkpoint_path"]).resolve().parent.parent)
    if state.get("training_checkpoint_path"):
        keep.add(Path(state["training_checkpoint_path"]).resolve().parent.parent)
    for path in (root / "chunks").glob("chunk_*"):
        if path.resolve() in keep:
            continue
        autopilot._require_within(path.resolve(), root / "chunks", label="LC0 retention")
        # Keep metrics, completion manifests and gate evidence indefinitely.
        for file in (path / "train" / "checkpoint.json", path / "train" / "optimizer.pt", path / "candidate.nnue"):
            file.unlink(missing_ok=True)


def _verify_resume_artifacts(state: dict[str, Any]) -> None:
    checks = [(state["active_model_path"], state["active_model_sha256"])]
    if state.get("history"):
        latest = state["history"][-1]
        checks.extend([(state["training_checkpoint_path"], latest["checkpoint_sha256"]),
                       (state["training_optimizer_path"], latest["optimizer_sha256"])])
    if state.get("best_quant_path"):
        checks.extend([(state["best_quant_path"], state["best_quant_sha256"]),
                       (state["best_checkpoint_path"], state["best_checkpoint_sha256"])])
    for raw, sha in checks:
        path = Path(raw)
        if not path.is_file() or _sha(path) != sha:
            raise ValueError(f"committed campaign artifact checksum mismatch: {path}")


def _completed_chunk(output: Path, pending: dict[str, Any]) -> dict[str, Any] | None:
    manifest = output / "complete.json"
    if not manifest.exists():
        return None
    completed = json.loads(manifest.read_text())
    if any(completed.get(key) != value for key, value in pending.items()):
        raise ValueError("completed chunk does not match pending traversal identity")
    for relative, key in (("train/checkpoint.json", "checkpoint_sha256"),
                          ("train/optimizer.pt", "optimizer_sha256"),
                          ("candidate.nnue", "quant_sha256")):
        path = output / relative
        if not path.is_file() or _sha(path) != completed[key]:
            raise ValueError(f"completed chunk artifact checksum mismatch: {path}")
    return completed


def _evaluate_candidate(args, root: Path, state: dict[str, Any], *, stamp: float) -> None:
    candidate = Path(state["best_quant_path"])
    sha = state["best_quant_sha256"]
    if args.gate_games == 0:
        state["last_gate"] = {"accepted": False, "reason": "gate-disabled-promotion-ineligible"}
        return
    gate_identity = {"candidate": sha, "incumbent": state["active_model_sha256"]}
    if sha == state["active_model_sha256"] or gate_identity == state.get("last_gate_identity"):
        return
    gate_dir = root / "gates" / f"chunk_{state['completed_chunks']:08d}_{sha[:12]}"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate = autopilot._run_confirmed_gate_attempt(
        piebot_dir=args.piebot_dir.resolve(), screen_json=gate_dir / "screen.json",
        confirmation_json=gate_dir / "confirmation.json",
        base_quant=Path(state["active_model_path"]), candidate_quant=candidate,
        screen_games=400, confirmation_games=1000, movetime_ms=150,
        noise_plies=12, noise_topk=5, threads=1, seed=args.seed + state["completed_chunks"],
        screen_min_score_delta=0.0, confirmation_min_score_delta=0.0,
        base_blend_percent=75, candidate_blend_percent=75, paired_openings=True,
        incremental_pst_policy="strict-superiority", confidence_level=0.95,
        bootstrap_samples=20_000, parallel_games=args.gate_parallel_games,
    )
    gate.update({"candidate_sha256": sha, "incumbent_sha256": state["active_model_sha256"], "at": stamp})
    autopilot._atomic_write_json(gate_dir / "decision.json", gate)
    if gate.get("accepted"):
        accepted = root / "accepted" / f"{sha}.nnue"
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, accepted)
        if _sha(accepted) != sha:
            raise RuntimeError("accepted model copy checksum mismatch")
        state["active_model_path"] = str(accepted)
        state["active_model_sha256"] = sha
        state["accepted_models"].append({"path": str(accepted), "sha256": sha,
            "blend_percent": 75, "completed_chunks": state["completed_chunks"], "gate": str(gate_dir / "decision.json")})
    state["last_gate"] = gate
    state["last_gate_identity"] = gate_identity


def run(args, *, now: Callable[[], float] = time.time, stop_requested: Callable[[], bool] = lambda: False) -> int:
    root = args.out_root.resolve()
    if not math.isfinite(args.hours) or args.hours <= 0 or args.max_chunks < 0:
        raise ValueError("hours must be positive and max-chunks nonnegative")
    if args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("positive batch size and learning rate required")
    if not math.isfinite(args.disk_reserve_gib) or args.disk_reserve_gib < 0 or args.gate_parallel_games < 1:
        raise ValueError("invalid disk reserve or gate parallelism")
    for protected in (args.initial_checkpoint.resolve(), args.initial_active_model.resolve(), args.corpus_manifest.resolve()):
        if protected == root or root in protected.parents:
            raise ValueError("LC0 output root must be separate from bootstrap and corpus artifacts")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "autopilot_state.json").exists():
        raise ValueError("refusing an existing self-play output root")
    with autopilot._single_instance_lock(root / "lc0.lock"):
        state_path = root / "lc0_state.json"
        state = autopilot._load_state(state_path)
        corpus = load_corpus(args.corpus_manifest, verify_chunks=state is None)
        identity = _identity(args, corpus)
        if state is not None and (state.get("schema") != SCHEMA or state.get("identity") != identity):
            raise ValueError("LC0 resume identity mismatch; create a new output root for changed source/corpus/objective/bootstrap")
        if state is None:
            checkpoint = json.loads(args.initial_checkpoint.read_text())
            if (checkpoint.get("arch"), checkpoint.get("hidden_dim"), checkpoint.get("input_dim")) != ("v2", 1024, 40960):
                raise ValueError("initial checkpoint must be the existing v2 hidden-1024 architecture")
            with args.initial_active_model.open("rb") as handle:
                if handle.read(8) not in (b"PIENNQ01", b"PIENNQ02"):
                    raise ValueError("initial accepted model has invalid NNUE magic")
            del checkpoint
            started = now()
            state = {
                "schema": SCHEMA, "identity": identity, "status": "running", "started_at": started,
                "deadline_at": started + args.hours * 3600, "completed_chunks": 0,
                "pass_number": 0, "cursor": 0, "in_progress": None, "history": [],
                "training_checkpoint_path": None, "training_optimizer_path": None,
                "active_model_path": str(args.initial_active_model.resolve()),
                "active_model_sha256": identity["initial_active_model"]["sha256"],
                "active_model_blend_percent": 75, "accepted_models": [], "last_gate": None,
                "last_evaluation_at": started, "evaluation_pending": False,
                "baseline_validation": None, "best_validation_loss": None,
            }
            autopilot._atomic_write_json(state_path, state)
        if state["status"] == "complete":
            print("LC0 campaign has completed its fixed training budget.", flush=True)
            return 0
        _verify_resume_artifacts(state)
        validation_dir = root / "cache" / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_file = validation_dir / "validation.jsonl"
        if not validation_file.exists() or _sha(validation_file) != corpus["validation"]["sha256"]:
            shutil.copyfile(corpus["validation"]["path"], validation_file)
        if _sha(validation_file) != corpus["validation"]["sha256"]:
            raise ValueError("fixed validation copy checksum mismatch")
        state["status"] = "running"
        state["last_error"] = None
        additional = 0
        try:
            while True:
                stamp = now()
                if stop_requested() or stamp >= state["deadline_at"]:
                    state["status"] = "paused" if stop_requested() else "complete"
                    state["stopped_at"] = stamp
                    autopilot._atomic_write_json(state_path, state)
                    return 0
                # Persist pending evaluation independently from the training
                # cursor. A killed gate must never cause a finished chunk to retrain.
                if state.get("evaluation_pending"):
                    _evaluate_candidate(args, root, state, stamp=stamp)
                    state["evaluation_pending"] = False
                    state["last_evaluation_at"] = now()
                    autopilot._atomic_write_json(state_path, state)
                if args.max_chunks and additional >= args.max_chunks:
                    state["status"] = "paused"
                    autopilot._atomic_write_json(state_path, state)
                    return 0
                _check_disk(root, args.disk_reserve_gib)
                order = _order(len(corpus["chunks"]), args.seed, state["pass_number"])
                index = order[state["cursor"]]
                chunk = corpus["chunks"][index]
                number = state["completed_chunks"]
                output = root / "chunks" / f"chunk_{number:08d}"
                seed = autopilot._derive_cycle_seed(args.seed, number)
                pending = {"number": number, "pass_number": state["pass_number"], "cursor": state["cursor"],
                           "chunk_index": index, "chunk_sha256": chunk["sha256"], "seed": seed}
                if state.get("in_progress") not in (None, pending):
                    raise ValueError("pending chunk identity does not match persisted traversal cursor")
                state["in_progress"] = pending
                autopilot._atomic_write_json(state_path, state)
                print(json.dumps({"event": "lc0-chunk-start", **pending}), flush=True)
                train_jsonl = root / "cache" / "training"
                first = state["training_checkpoint_path"] is None
                checkpoint_path = output / "train" / "checkpoint.json"
                optimizer_path = output / "train" / "optimizer.pt"
                candidate = output / "candidate.nnue"
                completed = _completed_chunk(output, pending)
                if completed is None:
                    _expand_chunk(chunk, train_jsonl)
                    if shutil.disk_usage(root).free < args.disk_reserve_gib * (1024 ** 3):
                        raise RuntimeError("disk reserve exhausted by expanded corpus cache")
                    # Partial outputs are never resumed as checkpoints. Replay
                    # unfinished work from the preceding committed learner.
                    shutil.rmtree(output / "train", ignore_errors=True)
                    metrics = _train(
                        jsonl_dir=train_jsonl, out_dir=output / "train", arch="v2", hidden_dim=1024,
                        batch_size=args.batch_size, max_samples=CHUNK_LIMIT, epochs=1, val_split=0.0,
                        learning_rate=args.learning_rate, target_cp=250.0, cp_loss_weight=0.0,
                        teacher_mix=0.8, min_teacher_depth=0, loss_kind="wdl", wdl_scale_cp=400.0,
                        primary_sample_fraction=1.0, teacher_sample_fraction=0.0,
                        validation_jsonl_dir=validation_dir, max_validation_samples=VALIDATION_LIMIT,
                        validation_seed=args.seed, validation_require_teacher=False,
                        initial_checkpoint=args.initial_checkpoint.resolve() if first else Path(state["training_checkpoint_path"]),
                        initial_checkpoint_weights_only=first,
                        initial_optimizer_state=None if first else Path(state["training_optimizer_path"]),
                        seed=seed, device=args.device, target_mode="lc0-q-outcome", checkpoint_selection="latest",
                    )
                    if not checkpoint_path.is_file() or not optimizer_path.is_file():
                        raise RuntimeError("trainer did not write both latest checkpoint and Adam state")
                    checkpoint = json.loads(checkpoint_path.read_text())
                    run_pipeline._export_v2_checkpoint(checkpoint, quant_path=candidate)
                    del checkpoint
                    with candidate.open("rb") as handle:
                        if handle.read(8) != b"PIENNQ02":
                            raise ValueError("LC0 candidate export must be PIENNQ02")
                    val_loss = float(metrics["selected_reference_val_loss"])
                    initial_loss = float(metrics["initial_reference_val_loss"])
                    if not math.isfinite(val_loss) or not math.isfinite(initial_loss):
                        raise ValueError("non-finite fixed validation loss")
                    completed = {**pending, "directory": str(output), "positions": chunk["positions"],
                        "checkpoint_sha256": _sha(checkpoint_path), "optimizer_sha256": _sha(optimizer_path),
                        "quant_sha256": _sha(candidate), "validation_loss": val_loss,
                        "initial_validation_loss": initial_loss, "finished_at": now()}
                    autopilot._atomic_write_json(output / "complete.json", completed)
                candidate_sha = completed["quant_sha256"]
                val_loss = completed["validation_loss"]
                state["training_checkpoint_path"] = str(checkpoint_path)
                state["training_optimizer_path"] = str(optimizer_path)
                if first:
                    baseline_loss = float(completed["initial_validation_loss"])
                    if not math.isfinite(baseline_loss):
                        raise ValueError("non-finite baseline validation loss")
                    state["baseline_validation"] = {"loss": baseline_loss,
                        "checkpoint_sha256": identity["initial_checkpoint"]["sha256"],
                        "validation_sha256": corpus["validation"]["sha256"]}
                    if baseline_loss <= val_loss:
                        baseline_quant = root / "baseline" / "candidate.nnue"
                        baseline_quant.parent.mkdir(parents=True, exist_ok=True)
                        baseline_checkpoint = json.loads(args.initial_checkpoint.read_text())
                        run_pipeline._export_v2_checkpoint(baseline_checkpoint, quant_path=baseline_quant)
                        del baseline_checkpoint
                        state.update(best_validation_loss=baseline_loss,
                            best_checkpoint_path=str(args.initial_checkpoint.resolve()),
                            best_checkpoint_sha256=identity["initial_checkpoint"]["sha256"],
                            best_quant_path=str(baseline_quant), best_quant_sha256=_sha(baseline_quant))
                if state["best_validation_loss"] is None or val_loss < state["best_validation_loss"]:
                    state.update(best_validation_loss=val_loss, best_checkpoint_path=str(checkpoint_path),
                                 best_checkpoint_sha256=completed["checkpoint_sha256"],
                                 best_quant_path=str(candidate), best_quant_sha256=candidate_sha)
                state["history"].append(completed)
                state["completed_chunks"] += 1
                state["cursor"] += 1
                pass_finished = state["cursor"] == len(order)
                if pass_finished:
                    state["pass_number"] += 1
                    state["cursor"] = 0
                state["in_progress"] = None
                state["evaluation_pending"] = pass_finished or completed["finished_at"] - state["last_evaluation_at"] >= DAY
                autopilot._atomic_write_json(state_path, state)
                _retain(root, state)
                shutil.rmtree(train_jsonl, ignore_errors=True)
                additional += 1
                print(json.dumps({"event": "lc0-chunk-complete", "completed_chunks": state["completed_chunks"],
                    "pass_number": state["pass_number"], "validation_loss": val_loss,
                    "best_validation_loss": state["best_validation_loss"]}), flush=True)
        except Exception as exc:
            # Do not accidentally commit a half-built in-memory cursor when a
            # write/export/gate fails. The last atomic state is authoritative.
            state = autopilot._load_state(state_path) or state
            state["status"] = "error"
            state["last_error"] = {"type": type(exc).__name__, "message": str(exc), "at": now()}
            autopilot._atomic_write_json(state_path, state)
            raise


def main(argv=None) -> int:
    args = _parse_args(argv)
    stop = [False]
    def request_stop(_signum, _frame):
        stop[0] = True
        print("LC0 stop requested; saving at the next chunk boundary.", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        return run(args, stop_requested=lambda: stop[0])
    except Exception as exc:
        print(f"LC0 campaign failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
