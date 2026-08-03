#!/usr/bin/env python3
"""Set-and-forget NNUE training autopilot with crash-safe resume."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Dict, Optional

try:
    from . import run_pipeline
except Exception:
    import run_pipeline  # type: ignore

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore


class _FileLockBackend:
    name = "unknown"

    def lock(self, handle: IO[str]) -> None:
        raise NotImplementedError

    def unlock(self, handle: IO[str]) -> None:
        raise NotImplementedError


class _FcntlFileLockBackend(_FileLockBackend):
    name = "fcntl"

    def lock(self, handle: IO[str]) -> None:
        if fcntl is None:
            raise RuntimeError("fcntl backend unavailable")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(self, handle: IO[str]) -> None:
        if fcntl is None:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _MsvcrtFileLockBackend(_FileLockBackend):
    name = "msvcrt"

    def lock(self, handle: IO[str]) -> None:
        if msvcrt is None:
            raise RuntimeError("msvcrt backend unavailable")
        handle.seek(0)
        handle.write("0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock(self, handle: IO[str]) -> None:
        if msvcrt is None:
            return
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _select_lock_backend() -> _FileLockBackend:
    if fcntl is not None:
        return _FcntlFileLockBackend()
    if msvcrt is not None:
        return _MsvcrtFileLockBackend()
    raise RuntimeError("autopilot locking requires either fcntl or msvcrt support")


def zen5_9755_7d_profile() -> Dict[str, Any]:
    """Defaults tuned for sustained NNUE-v2 training on a Zen5 9755 VM."""
    return {
        "selfplay_games": 8_000,
        "selfplay_max_plies": 160,
        "selfplay_threads": 1,
        "selfplay_parallel_games": 0,
        "selfplay_depth": 2,
        "selfplay_temperature_tau": 1.0,
        "selfplay_temperature_tau_final": 0.1,
        "selfplay_temperature_moves": 24,
        "selfplay_dirichlet_alpha": 0.30,
        "selfplay_dirichlet_epsilon": 0.25,
        "selfplay_dirichlet_plies": 12,
        "selfplay_seed": 42,
        "teacher_relabel_depth": 6,
        "teacher_relabel_every": 4,
        "teacher_relabel_threads": 48,
        "teacher_relabel_hash_mb": 4096,
        "teacher_relabel_max_records": 0,
        "batch_size": 4096,
        "max_samples": 700_000,
        "epochs": 2,
        # The production scalar evaluator benchmarks materially faster at 64;
        # v2 gains capacity from its all-piece HalfKP inputs instead of width.
        "hidden_dim": 64,
        "training_input_dim": 81_920,
        "training_feature_set": "halfkp-all-pieces-v2",
        "training_target_schema": "soft-cp-wdl-v2",
        "training_objective_schema": "nnue-objective-v1",
        "target_cp": 100.0,
        "teacher_mix": 0.8,
        "max_teacher_cp": 1200.0,
        "outcome_decay": 1.0,
        "learning_rate": 0.003,
        "warm_start_learning_rate": 0.001,
        "val_split": 0.1,
        "seed": 1,
        "trainer_backend": "auto",
        "trainer_device": "cuda",
        "resume": True,
        "retain_full_cycles": 0,
        "replay_window_cycles": 6,
        "primary_sample_fraction": 0.5,
        "teacher_sample_fraction": 0.5,
        "teacher_lag_cycles": 0,
        "min_teacher_depth": 6,
        "loss_kind": "wdl",
        "huber_delta_cp": 100.0,
        "wdl_scale_cp": 400.0,
        "validation_jsonl_dir": None,
        "max_validation_samples": 100_000,
        "validation_seed": 20_260_802,
        "validation_require_teacher": True,
        "continue_optimizer_state": True,
        "gate_games": 24,
        "gate_movetime_ms": 150,
        "gate_noise_plies": 12,
        "gate_noise_topk": 5,
        "gate_threads": 1,
        "gate_seed": 1,
        "gate_min_score_delta": 0.0,
        "gate_paired_openings": True,
        "gate_confirmation_games": 96,
        "gate_confirmation_min_score_delta": 2.0,
        "warm_start": True,
        "initial_checkpoint": None,
    }


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _validate_training_lineage_floor(state: Dict[str, Any]) -> int:
    raw = state.get("training_lineage_start_cycle")
    if isinstance(raw, bool):
        raise ValueError("training_lineage_start_cycle must be an integer")
    try:
        floor = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_lineage_start_cycle must be an integer") from exc
    if floor < 1:
        raise ValueError("training_lineage_start_cycle must be >= 1")
    next_cycle = int(state.get("next_cycle", 1))
    if floor > next_cycle:
        raise ValueError(
            f"training_lineage_start_cycle {floor} cannot exceed next_cycle {next_cycle}"
        )
    return floor


def _atomic_reset_training_lineage(
    *,
    state_path: Path,
    state: Dict[str, Any],
    start_cycle: int,
) -> tuple[Dict[str, Any], bool]:
    start = int(start_cycle)
    existing_reset = state.get("training_lineage_reset")
    already_reset = (
        int(state.get("training_lineage_start_cycle", 0) or 0) == start
        and isinstance(existing_reset, dict)
        and int(existing_reset.get("start_cycle", 0) or 0) == start
    )
    if already_reset:
        return state, False

    next_cycle = int(state.get("next_cycle", 1))
    if start < 1 or start != next_cycle:
        raise ValueError(
            "--reset-training-lineage-at-cycle must equal the current "
            f"next_cycle ({next_cycle}); got {start}"
        )

    next_state = copy.deepcopy(state)
    next_state["training_lineage_start_cycle"] = start
    next_state["training_lineage_reset"] = {
        "start_cycle": start,
        "reset_at": time.time(),
        "prior_start_cycle": state.get("training_lineage_start_cycle"),
        "prior_checkpoint_path": state.get("training_checkpoint_path"),
        "prior_checkpoint_sha256": state.get("training_checkpoint_sha256"),
        "prior_model_identity": state.get("training_model_identity"),
    }
    next_state["training_checkpoint_path"] = None
    next_state["training_checkpoint_sha256"] = None
    next_state["training_model_identity"] = None
    _validate_training_lineage_floor(next_state)
    _atomic_write_json(state_path, next_state)
    return next_state, True


def _configured_training_objective(defaults: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    schema = defaults.get("training_objective_schema")
    target_schema = defaults.get("training_target_schema")
    if schema is None or target_schema is None:
        return None
    return {
        "schema": str(schema),
        "target_schema": str(target_schema),
        "loss_kind": str(defaults.get("loss_kind", "mse")),
        "target_cp": float(defaults.get("target_cp", 100.0)),
        "teacher_mix": float(defaults.get("teacher_mix", 0.8)),
        "max_teacher_cp": float(defaults.get("max_teacher_cp", 1200.0)),
        "outcome_decay": float(defaults.get("outcome_decay", 1.0)),
        "min_teacher_depth": int(defaults.get("min_teacher_depth", 0)),
        "huber_delta_cp": float(defaults.get("huber_delta_cp", 100.0)),
        "wdl_scale_cp": float(defaults.get("wdl_scale_cp", 400.0)),
    }


def _validate_training_checkpoint_identity(
    state: Dict[str, Any],
    defaults: Dict[str, Any],
) -> None:
    if not state.get("training_checkpoint_path"):
        return
    identity = state.get("training_model_identity")
    if not isinstance(identity, dict):
        return
    expected = {
        "input_dim": defaults.get("training_input_dim"),
        "hidden_dim": defaults.get("hidden_dim"),
        "feature_set": defaults.get("training_feature_set"),
        "target_schema": defaults.get("training_target_schema"),
        "objective": _configured_training_objective(defaults),
    }
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if value is not None
        and identity.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "training checkpoint identity is incompatible with the configured lineage "
            f"({mismatches}); restart with --reset-training-lineage-at-cycle "
            f"{int(state.get('next_cycle', 1))}"
        )


def _load_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _single_instance_lock(lock_path: Path, *, backend: Optional[_FileLockBackend] = None):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    lock_backend = backend or _select_lock_backend()
    locked = False
    try:
        lock_backend.lock(handle)
        locked = True
        handle.seek(0)
        handle.truncate(0)
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            if locked:
                lock_backend.unlock(handle)
        finally:
            handle.close()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True, help="Root directory for autopilot outputs")
    ap.add_argument("--piebot-dir", type=Path, default=Path(__file__).resolve().parents[2] / "PieBot")
    ap.add_argument("--hours", type=float, default=24.0 * 7.0, help="Runtime budget in hours")
    ap.add_argument("--max-cycles", type=int, default=0, help="Optional max cycles (0 = unlimited)")
    ap.add_argument("--retry-limit", type=int, default=5, help="Retries per cycle before aborting")
    ap.add_argument("--retry-backoff-sec", type=float, default=30.0)
    ap.add_argument(
        "--reset-training-lineage-at-cycle",
        type=int,
        default=None,
        help="Atomically start a fresh checkpoint/replay lineage at the current next cycle",
    )
    ap.add_argument("--profile", default="zen5_9755_7d", choices=["zen5_9755_7d"])
    ap.add_argument("--selfplay-games", type=int, default=None)
    ap.add_argument("--selfplay-depth", type=int, default=None)
    ap.add_argument("--selfplay-threads", type=int, default=None)
    ap.add_argument("--selfplay-parallel-games", type=int, default=None)
    ap.add_argument("--teacher-relabel-depth", type=int, default=None)
    ap.add_argument("--teacher-relabel-every", type=int, default=None)
    ap.add_argument("--teacher-relabel-threads", type=int, default=None)
    ap.add_argument("--teacher-relabel-hash-mb", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--primary-sample-fraction",
        type=float,
        default=None,
        help="Minimum fraction of a capped training sample reserved for the current cycle",
    )
    ap.add_argument("--teacher-sample-fraction", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--min-teacher-depth", type=int, default=None)
    ap.add_argument("--loss-kind", default=None)
    ap.add_argument("--huber-delta-cp", type=float, default=None)
    ap.add_argument("--wdl-scale-cp", type=float, default=None)
    ap.add_argument("--validation-jsonl-dir", type=Path, default=None)
    ap.add_argument("--max-validation-samples", type=int, default=None)
    ap.add_argument("--validation-seed", type=int, default=None)
    ap.add_argument(
        "--validation-require-teacher",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--continue-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Continue optimizer moments alongside a warm-started model checkpoint",
    )
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--warm-start-learning-rate", type=float, default=None)
    ap.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Initialize each training cycle from the latest completed float checkpoint",
    )
    ap.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Bootstrap checkpoint for the first cumulative training cycle",
    )
    ap.add_argument("--trainer-backend", choices=["stub", "torch", "auto"], default=None)
    ap.add_argument("--trainer-device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument(
        "--retain-full-cycles",
        type=int,
        default=None,
        help="Keep newest N cycle directories in full (0 = unlimited)",
    )
    ap.add_argument("--replay-window-cycles", type=int, default=None)
    ap.add_argument("--teacher-lag-cycles", type=int, default=None)
    ap.add_argument("--gate-games", type=int, default=None)
    ap.add_argument("--gate-movetime-ms", type=int, default=None)
    ap.add_argument("--gate-noise-plies", type=int, default=None)
    ap.add_argument("--gate-noise-topk", type=int, default=None)
    ap.add_argument("--gate-threads", type=int, default=None)
    ap.add_argument("--gate-seed", type=int, default=None)
    ap.add_argument("--gate-min-score-delta", type=float, default=None)
    ap.add_argument(
        "--gate-paired-openings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--gate-confirmation-games", type=int, default=None)
    ap.add_argument("--gate-confirmation-min-score-delta", type=float, default=None)
    return ap.parse_args(argv)


def _profile_defaults(name: str) -> Dict[str, Any]:
    if name == "zen5_9755_7d":
        return zen5_9755_7d_profile()
    raise ValueError(f"unknown profile: {name}")


def _derive_cycle_seed(base_seed: int, cycle_idx: int, *, stream: int = 0) -> int:
    """Derive a stable, independent u64 seed for one autopilot cycle."""
    mask = (1 << 64) - 1
    x = int(base_seed) & mask
    x ^= (int(cycle_idx) * 0x9E3779B97F4A7C15) & mask
    x ^= (int(stream) * 0xD1B54A32D192ED03) & mask
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return int(x & mask) or 1


_BLEND_RAMP = (25, 50, 75, 100)
_RUNTIME_IDENTITY_KEYS = (
    "quant_format",
    "quant_version",
    "input_dim",
    "hidden_dim",
    "output_dim",
)
_SEMANTIC_IDENTITY_KEYS = (
    "feature_set",
    "feature_schema",
    "schema_version",
    "target_schema",
    "objective",
)


def _normalized_blend_percent(raw: Any) -> Optional[int]:
    if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        return None
    return max(0, min(100, int(raw)))


def _active_accepted_model(state: Dict[str, Any]) -> tuple[Optional[int], Optional[Dict[str, Any]]]:
    active_path = state.get("active_model_path")
    accepted = state.get("accepted_models")
    if not active_path or not isinstance(accepted, list):
        return None, None
    for idx in range(len(accepted) - 1, -1, -1):
        model = accepted[idx]
        if isinstance(model, dict) and model.get("quant_path") == active_path:
            return idx, model
    return None, None


def _active_model_blend_percent(state: Dict[str, Any]) -> int:
    explicit = _normalized_blend_percent(state.get("active_model_blend_percent"))
    if explicit is not None:
        return explicit

    accepted = state.get("accepted_models")
    idx, active_model = _active_accepted_model(state)
    if idx is not None and active_model is not None:
        return _accepted_model_promoted_blend(active_model, idx)
    if not isinstance(accepted, list) or not accepted:
        return 0
    # Final compatibility fallback for states that predate active_model_path or
    # did not record the promoted blend on an acceptance.
    return int(_BLEND_RAMP[min(len(accepted) - 1, len(_BLEND_RAMP) - 1)])


def _model_identities_same(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if any(key not in left or key not in right for key in _RUNTIME_IDENTITY_KEYS):
        return False
    if any(left[key] != right[key] for key in _RUNTIME_IDENTITY_KEYS):
        return False
    for key in _SEMANTIC_IDENTITY_KEYS:
        left_has_value = key in left and left.get(key) is not None
        right_has_value = key in right and right.get(key) is not None
        if left_has_value != right_has_value:
            return False
        if left_has_value and left.get(key) != right.get(key):
            return False
    return True


def _active_model_identity(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = state.get("active_model_identity")
    if isinstance(explicit, dict):
        return dict(explicit)
    _, active_model = _active_accepted_model(state)
    if isinstance(active_model, dict) and isinstance(active_model.get("model_identity"), dict):
        return dict(active_model["model_identity"])
    active_path = _path_if_exists(state.get("active_model_path"))
    if active_path is not None:
        return _quant_model_identity(active_path)
    return None


def _candidate_model_blend_percent(
    state: Dict[str, Any],
    *,
    candidate_identity: Optional[Dict[str, Any]] = None,
) -> int:
    accepted = state.get("accepted_models")
    has_active = bool(state.get("active_model_path")) or (
        isinstance(accepted, list) and bool(accepted)
    )
    if not has_active:
        return _BLEND_RAMP[0]

    active_identity = _active_model_identity(state)
    if (
        candidate_identity is not None
        and active_identity is not None
        and not _model_identities_same(active_identity, candidate_identity)
    ):
        return _BLEND_RAMP[0]

    active_blend = _active_model_blend_percent(state)
    for blend in _BLEND_RAMP:
        if blend > active_blend:
            return blend
    return _BLEND_RAMP[-1]


def _apply_cli_overrides(defaults: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(defaults)
    mapping = {
        "selfplay_games": args.selfplay_games,
        "selfplay_depth": args.selfplay_depth,
        "selfplay_threads": args.selfplay_threads,
        "selfplay_parallel_games": args.selfplay_parallel_games,
        "teacher_relabel_depth": args.teacher_relabel_depth,
        "teacher_relabel_every": args.teacher_relabel_every,
        "teacher_relabel_threads": args.teacher_relabel_threads,
        "teacher_relabel_hash_mb": args.teacher_relabel_hash_mb,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "primary_sample_fraction": args.primary_sample_fraction,
        "teacher_sample_fraction": args.teacher_sample_fraction,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "min_teacher_depth": args.min_teacher_depth,
        "loss_kind": args.loss_kind,
        "huber_delta_cp": args.huber_delta_cp,
        "wdl_scale_cp": args.wdl_scale_cp,
        "validation_jsonl_dir": args.validation_jsonl_dir,
        "max_validation_samples": args.max_validation_samples,
        "validation_seed": args.validation_seed,
        "validation_require_teacher": args.validation_require_teacher,
        "continue_optimizer_state": args.continue_optimizer_state,
        "learning_rate": args.learning_rate,
        "warm_start_learning_rate": args.warm_start_learning_rate,
        "warm_start": args.warm_start,
        "initial_checkpoint": args.initial_checkpoint,
        "trainer_backend": args.trainer_backend,
        "trainer_device": args.trainer_device,
        "retain_full_cycles": args.retain_full_cycles,
        "replay_window_cycles": args.replay_window_cycles,
        "teacher_lag_cycles": args.teacher_lag_cycles,
        "gate_games": args.gate_games,
        "gate_movetime_ms": args.gate_movetime_ms,
        "gate_noise_plies": args.gate_noise_plies,
        "gate_noise_topk": args.gate_noise_topk,
        "gate_threads": args.gate_threads,
        "gate_seed": args.gate_seed,
        "gate_min_score_delta": args.gate_min_score_delta,
        "gate_paired_openings": args.gate_paired_openings,
        "gate_confirmation_games": args.gate_confirmation_games,
        "gate_confirmation_min_score_delta": args.gate_confirmation_min_score_delta,
    }
    for k, v in mapping.items():
        if v is not None:
            out[k] = v
    return out


def _path_if_exists(raw: Any) -> Optional[Path]:
    if isinstance(raw, str) and raw:
        p = Path(raw)
        if p.exists():
            return p
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _quant_model_identity(
    path: Path,
    *,
    summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"PIENNQ01":
        return None
    try:
        version, input_dim, hidden_dim, output_dim = struct.unpack("<IIII", header[8:24])
    except struct.error:
        return None
    if version <= 0 or input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
        return None

    identity: Dict[str, Any] = {
        "quant_format": "PIENNQ01",
        "quant_version": int(version),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "output_dim": int(output_dim),
    }
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    if isinstance(metrics, dict):
        for key, expected in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
        ):
            reported = metrics.get(key)
            if reported is not None and int(reported) != int(expected):
                raise ValueError(
                    f"candidate {key} mismatch between summary ({reported}) and quant ({expected})"
                )
        for key in ("feature_set", "feature_schema", "schema_version"):
            value = metrics.get(key)
            if isinstance(value, (str, int, float)) and value != "":
                identity[key] = value
        target_schema = metrics.get("target_schema")
        if isinstance(target_schema, str) and target_schema:
            identity["target_schema"] = target_schema
        objective = metrics.get("objective")
        if isinstance(objective, dict):
            identity["objective"] = copy.deepcopy(objective)
    return identity


def _infer_training_model_identity(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = state.get("training_model_identity")
    if isinstance(explicit, dict):
        return dict(explicit)
    if not state.get("training_checkpoint_path"):
        return None
    summary = state.get("last_summary")
    if not isinstance(summary, dict):
        return None
    quant_path = _path_if_exists(summary.get("quant_path"))
    if quant_path is not None:
        identity = _quant_model_identity(quant_path, summary=summary)
        if identity is not None:
            return identity
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    identity = {
        key: metrics[key]
        for key in (
            "input_dim",
            "hidden_dim",
            "feature_set",
            "target_schema",
            "objective",
        )
        if metrics.get(key) is not None
    }
    return identity or None


def _migrate_deployment_state(state: Dict[str, Any]) -> bool:
    """Persist the exact deployed model tuple without discarding audit history."""
    changed = False
    if state.get("deployment_state_version") != 2:
        state["deployment_state_version"] = 2
        changed = True
    if "training_lineage_start_cycle" not in state:
        state["training_lineage_start_cycle"] = 1
        changed = True
    _validate_training_lineage_floor(state)
    if "training_model_identity" not in state or (
        state.get("training_checkpoint_path")
        and not isinstance(state.get("training_model_identity"), dict)
    ):
        training_identity = _infer_training_model_identity(state)
        state["training_model_identity"] = training_identity
        changed = True

    active_raw = state.get("active_model_path")
    if not active_raw:
        defaults = {
            "active_model_sha256": None,
            "active_model_blend_percent": 0,
            "active_model_identity": None,
        }
        for key, value in defaults.items():
            if key not in state:
                state[key] = value
                changed = True
        return changed

    idx, accepted_model = _active_accepted_model(state)
    if _normalized_blend_percent(state.get("active_model_blend_percent")) is None:
        blend = (
            _accepted_model_promoted_blend(accepted_model, idx)
            if idx is not None and accepted_model is not None
            else _active_model_blend_percent(state)
        )
        state["active_model_blend_percent"] = blend
        changed = True
    blend = _active_model_blend_percent(state)

    if not state.get("active_model_sha256"):
        accepted_sha = (
            accepted_model.get("quant_sha256")
            if isinstance(accepted_model, dict)
            else None
        )
        active_path = _path_if_exists(active_raw)
        active_sha = accepted_sha or (
            _sha256_file(active_path) if active_path is not None and active_path.is_file() else None
        )
        if active_sha:
            state["active_model_sha256"] = active_sha
            changed = True

    identity = state.get("active_model_identity")
    if not isinstance(identity, dict):
        accepted_identity = (
            accepted_model.get("model_identity")
            if isinstance(accepted_model, dict)
            else None
        )
        if isinstance(accepted_identity, dict):
            identity = dict(accepted_identity)
        else:
            active_path = _path_if_exists(active_raw)
            identity = (
                _quant_model_identity(active_path) if active_path is not None else None
            )
        if identity is not None:
            state["active_model_identity"] = identity
            changed = True

    if isinstance(accepted_model, dict):
        if accepted_model.get("blend_percent") != blend:
            accepted_model["blend_percent"] = blend
            changed = True
        if isinstance(identity, dict) and accepted_model.get("model_identity") != identity:
            accepted_model["model_identity"] = dict(identity)
            changed = True
    return changed


def _verified_quant_path(
    raw: Any,
    *,
    expected_sha256: Any = None,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if expected_sha256 and _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _resolve_active_quant_path(state: Dict[str, Any]) -> Optional[Path]:
    if "active_model_path" in state:
        raw = state.get("active_model_path")
        if not raw:
            return None
        expected_sha = state.get("active_model_sha256")
        accepted = state.get("accepted_models")
        if not expected_sha and isinstance(accepted, list):
            for model in reversed(accepted):
                if isinstance(model, dict) and model.get("quant_path") == raw:
                    expected_sha = model.get("quant_sha256")
                    break
        return _verified_quant_path(
            raw,
            expected_sha256=expected_sha,
            label="active model",
        )
    # Backward compatibility: older state schema used last_summary only.
    last_summary = state.get("last_summary")
    if isinstance(last_summary, dict):
        raw = last_summary.get("quant_path")
        if raw:
            return _verified_quant_path(
                raw,
                expected_sha256=last_summary.get("quant_sha256"),
                label="legacy active model",
            )
    return None


def _accepted_model_promoted_blend(model: Dict[str, Any], accepted_index: int) -> int:
    gate = model.get("gate")
    raw_blend = (
        gate.get("experimental_blend_percent")
        if isinstance(gate, dict)
        else None
    )
    if raw_blend is None:
        raw_blend = model.get("blend_percent")
    if isinstance(raw_blend, (int, float)) and math.isfinite(float(raw_blend)):
        return max(0, min(100, int(raw_blend)))

    # Older state did not persist a promoted blend. Reconstruct the historical
    # 25/50/75/100 ramp from the acceptance's position in the list.
    ramp = (25, 50, 75, 100)
    return int(ramp[min(max(0, int(accepted_index)), len(ramp) - 1)])


def _resolve_teacher_quant_and_blend(
    state: Dict[str, Any], lag_cycles: int
) -> tuple[Optional[Path], int]:
    lag = max(0, int(lag_cycles))
    accepted = state.get("accepted_models")
    if isinstance(accepted, list) and accepted:
        idx = len(accepted) - 1 - lag
        if idx >= 0 and isinstance(accepted[idx], dict):
            model = accepted[idx]
            return (
                _verified_quant_path(
                    model.get("quant_path"),
                    expected_sha256=model.get("quant_sha256"),
                    label="teacher model",
                ),
                _accepted_model_promoted_blend(model, idx),
            )
    return _resolve_active_quant_path(state), _active_model_blend_percent(state)


def _resolve_teacher_quant_path(state: Dict[str, Any], lag_cycles: int) -> Optional[Path]:
    teacher_quant, _ = _resolve_teacher_quant_and_blend(state, lag_cycles)
    return teacher_quant


def _collect_replay_jsonl_dirs(state: Dict[str, Any], window_cycles: int) -> list[Path]:
    window = max(0, int(window_cycles))
    if window == 0:
        return []
    completed = state.get("completed_cycles")
    if not isinstance(completed, list):
        return []
    lineage_floor = max(0, int(state.get("training_lineage_start_cycle", 0) or 0))
    out: list[Path] = []
    for c in reversed(completed):
        if not isinstance(c, dict):
            continue
        if int(c.get("cycle", 0) or 0) < lineage_floor:
            continue
        # Replay a cycle's fresh/relabelled shards. Its train_jsonl_dir may already
        # contain older replay windows, which would recursively duplicate history.
        p = _path_if_exists(c.get("jsonl_dir")) or _path_if_exists(c.get("train_jsonl_dir"))
        if p is None:
            continue
        if any(x.resolve() == p.resolve() for x in out):
            continue
        out.append(p)
        if len(out) >= window:
            break
    return out


def _resolve_training_checkpoint_path(
    state: Dict[str, Any],
    bootstrap_checkpoint: Optional[Path],
) -> Optional[Path]:
    if "training_checkpoint_path" in state:
        raw = state.get("training_checkpoint_path")
        if raw is None:
            # Current-schema null is an explicit lineage reset. Do not revive an
            # older checkpoint through the legacy completed-cycle fallback.
            return None
        if raw:
            checkpoint = _path_if_exists(raw)
            if checkpoint is None:
                raise ValueError(f"latest training checkpoint is missing: {raw}")
            expected_sha = state.get("training_checkpoint_sha256")
            if expected_sha and _sha256_file(checkpoint) != expected_sha:
                raise ValueError("latest training checkpoint SHA-256 mismatch")
            return checkpoint

    completed = state.get("completed_cycles")
    if isinstance(completed, list) and completed:
        latest = next((item for item in reversed(completed) if isinstance(item, dict)), None)
        candidates = []
        if latest is not None:
            candidates.append(
                (latest.get("checkpoint_path"), latest.get("checkpoint_sha256"))
            )
            out_dir = latest.get("out_dir")
            if isinstance(out_dir, str) and out_dir:
                candidates.append(
                    (
                        str(Path(out_dir) / "train" / "checkpoint.json"),
                        latest.get("checkpoint_sha256"),
                    )
                )
        last_summary = state.get("last_summary")
        if isinstance(last_summary, dict):
            candidates.append(
                (
                    last_summary.get("checkpoint_path"),
                    last_summary.get("checkpoint_sha256"),
                )
            )
        for raw, expected_sha in candidates:
            checkpoint = _path_if_exists(raw)
            if checkpoint is not None:
                if expected_sha and _sha256_file(checkpoint) != expected_sha:
                    raise ValueError("latest completed checkpoint SHA-256 mismatch")
                return checkpoint
        raise ValueError("latest completed cycle has no usable training checkpoint")

    if bootstrap_checkpoint is None:
        return None
    checkpoint = Path(bootstrap_checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"initial checkpoint does not exist: {checkpoint}")
    return checkpoint


def _retention_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"retention {label} path is missing")
    return Path(raw).resolve()


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"retention refuses {label} outside {root}: {path}") from exc


def _prune_cycle_directory(cycle_dir: Path, keep_files: set[Path]) -> bool:
    removed = False

    def prune(directory: Path) -> None:
        nonlocal removed
        for child in list(directory.iterdir()):
            resolved = child.resolve()
            if child.is_symlink():
                if resolved not in keep_files:
                    child.unlink()
                    removed = True
                continue
            if child.is_dir():
                prune(child)
                if not any(child.iterdir()):
                    child.rmdir()
                    removed = True
                continue
            if resolved not in keep_files:
                child.unlink()
                removed = True

    prune(cycle_dir)
    return removed


def _apply_cycle_retention(
    *,
    out_root: Path,
    state: Dict[str, Any],
    retain_full_cycles: int,
) -> Dict[str, Any]:
    """Prune completed cycles only after validating every destructive target."""
    retain = int(retain_full_cycles)
    if retain < 0:
        raise ValueError("retain_full_cycles must be >= 0")
    report: Dict[str, Any] = {
        "deleted_cycles": [],
        "pruned_cycles": [],
        "state_changed": False,
    }
    if retain == 0:
        return report

    completed = state.get("completed_cycles")
    if not isinstance(completed, list) or len(completed) <= retain:
        return report

    out_root_resolved = Path(out_root).resolve()
    cycles_root = out_root_resolved / "cycles"
    configured_cycles = Path(out_root) / "cycles"
    if configured_cycles.is_symlink() or configured_cycles.resolve() != cycles_root:
        raise ValueError(f"retention refuses cycles root outside {cycles_root}")

    entries: Dict[int, Dict[str, Any]] = {}
    cycle_dirs: Dict[int, Path] = {}
    ordered_cycles: list[int] = []
    for raw_entry in completed:
        if not isinstance(raw_entry, dict):
            raise ValueError("retention completed cycle entry is not an object")
        cycle = int(raw_entry.get("cycle", 0))
        if cycle <= 0 or cycle in entries:
            raise ValueError(f"retention invalid or duplicate cycle: {cycle}")
        expected = cycles_root / f"cycle_{cycle:06d}"
        recorded = raw_entry.get("out_dir")
        if recorded is not None:
            recorded_path = _retention_path(recorded, label=f"cycle {cycle}")
            _require_within(recorded_path, cycles_root, label=f"cycle {cycle}")
            if recorded_path != expected:
                raise ValueError(
                    f"retention refuses cycle {cycle} path outside expected directory: {recorded_path}"
                )
        if expected.is_symlink():
            raise ValueError(f"retention refuses symlinked cycle directory: {expected}")
        entries[cycle] = raw_entry
        cycle_dirs[cycle] = expected
        ordered_cycles.append(cycle)

    protected_quant: Dict[int, set[Path]] = {}
    accepted_cycles: set[int] = set()

    def protect_quant(cycle: int, raw_path: Any, *, label: str) -> None:
        if cycle not in entries:
            raise ValueError(f"retention {label} references unknown cycle {cycle}")
        path = _retention_path(raw_path, label=label)
        _require_within(path, cycles_root, label=label)
        _require_within(path, cycle_dirs[cycle], label=label)
        if not path.is_file():
            raise ValueError(f"retention cannot preserve missing {label}: {path}")
        protected_quant.setdefault(cycle, set()).add(path)
        accepted_cycles.add(cycle)

    accepted_models = state.get("accepted_models")
    if accepted_models is not None and not isinstance(accepted_models, list):
        raise ValueError("retention accepted_models is not a list")
    for model in accepted_models or []:
        if not isinstance(model, dict):
            raise ValueError("retention accepted model entry is not an object")
        protect_quant(
            int(model.get("cycle", 0)),
            model.get("quant_path"),
            label="accepted quant model",
        )

    for cycle, entry in entries.items():
        gate = entry.get("gate")
        if isinstance(gate, dict) and gate.get("accepted") and cycle not in accepted_cycles:
            protect_quant(cycle, entry.get("quant_path"), label="accepted cycle quant model")

    active_raw = state.get("active_model_path")
    if active_raw:
        active_path = _retention_path(active_raw, label="active model")
        _require_within(active_path, cycles_root, label="active model")
        relative = active_path.relative_to(cycles_root)
        try:
            active_cycle = int(relative.parts[0].removeprefix("cycle_"))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"retention cannot identify active model cycle: {active_path}") from exc
        protect_quant(active_cycle, active_raw, label="active model")

    full_cycles = set(ordered_cycles[-retain:])
    old_cycles = ordered_cycles[:-retain]

    # Every path is validated before the first removal. Filesystem failures may
    # leave a partial prune, which is safe to repeat on the next startup.
    for cycle in full_cycles:
        entry = entries[cycle]
        if entry.get("retention") not in {"deleted", "model_only"}:
            if entry.get("retention") != "full":
                entry["retention"] = "full"
                report["state_changed"] = True

    for cycle in old_cycles:
        entry = entries[cycle]
        cycle_dir = cycle_dirs[cycle]
        if cycle not in accepted_cycles:
            if cycle_dir.exists():
                shutil.rmtree(cycle_dir)
                report["deleted_cycles"].append(cycle)
            if entry.get("retention") != "deleted":
                entry["retention"] = "deleted"
                report["state_changed"] = True
            for key in (
                "jsonl_dir",
                "train_jsonl_dir",
                "checkpoint_path",
                "checkpoint_sha256",
                "quant_path",
                "summary_path",
            ):
                if entry.get(key) is not None:
                    entry[key] = None
                    report["state_changed"] = True
            continue

        if not cycle_dir.is_dir():
            raise ValueError(f"retention accepted cycle directory is missing: {cycle_dir}")
        if entry.get("retention") != "model_only":
            entry["retention"] = "model_only"
            report["state_changed"] = True
        for key in (
            "jsonl_dir",
            "train_jsonl_dir",
            "checkpoint_path",
            "checkpoint_sha256",
        ):
            if entry.get(key) is not None:
                entry[key] = None
                report["state_changed"] = True

        cycle_acceptances = [
            model
            for model in (accepted_models or [])
            if isinstance(model, dict) and int(model.get("cycle", 0)) == cycle
        ]
        retained_metadata = {
            "version": 1,
            "cycle": entry,
            "accepted_models": cycle_acceptances,
            "active_model": str(state.get("active_model_path") or "")
            in {str(path) for path in protected_quant.get(cycle, set())},
        }
        retained_path = cycle_dir / "retained_cycle.json"
        _atomic_write_json(retained_path, retained_metadata)
        keep_files = set(protected_quant.get(cycle, set()))
        keep_files.update(
            {
                retained_path.resolve(),
                (cycle_dir / "pipeline_summary.json").resolve(),
                (cycle_dir / "gate_compare.json").resolve(),
                (cycle_dir / "gate_compare_confirmation.json").resolve(),
                (cycle_dir / "gate_compare_same_blend.json").resolve(),
                (cycle_dir / "gate_compare_same_blend_confirmation.json").resolve(),
            }
        )
        if _prune_cycle_directory(cycle_dir, keep_files):
            report["pruned_cycles"].append(cycle)

    return report


def _enforce_cycle_retention(
    *,
    out_root: Path,
    state: Dict[str, Any],
    state_path: Path,
    retain_full_cycles: int,
) -> Dict[str, Any]:
    report = _apply_cycle_retention(
        out_root=out_root,
        state=state,
        retain_full_cycles=retain_full_cycles,
    )
    if report["state_changed"] or report["deleted_cycles"] or report["pruned_cycles"]:
        state["last_retention"] = {
            "retain_full_cycles": int(retain_full_cycles),
            "deleted_cycles": report["deleted_cycles"],
            "pruned_cycles": report["pruned_cycles"],
            "ts": time.time(),
        }
        _atomic_write_json(state_path, state)
    return report


def _run_model_gate(
    *,
    piebot_dir: Path,
    out_json: Path,
    base_quant: Optional[Path],
    candidate_quant: Path,
    games: int,
    movetime_ms: int,
    noise_plies: int,
    noise_topk: int,
    threads: int,
    seed: int,
    min_score_delta: float,
    base_blend_percent: int = 100,
    candidate_blend_percent: int = 100,
    paired_openings: bool = True,
) -> Dict[str, Any]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.unlink(missing_ok=True)
    expected_games = max(2, int(games))
    if paired_openings and expected_games % 2 != 0:
        raise ValueError(
            f"paired model gate requires an even game count; got {expected_games}"
        )
    cmd = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "--bin",
        "compare_play",
        "--",
        "--games",
        str(expected_games),
        "--movetime",
        str(max(1, int(movetime_ms))),
        "--noise-plies",
        str(max(0, int(noise_plies))),
        "--noise-topk",
        str(max(1, int(noise_topk))),
        "--threads",
        str(max(1, int(threads))),
        "--seed",
        str(max(1, int(seed))),
        "--json-out",
        str(out_json),
        "--same-search",
        "--exp-eval",
        "nnue",
        "--exp-use-nnue",
        "true",
        "--exp-blend",
        str(max(0, min(100, int(candidate_blend_percent)))),
        "--exp-nnue-quant-file",
        str(candidate_quant),
    ]
    if paired_openings:
        cmd.append("--paired-openings")
    if base_quant is None:
        cmd.extend(
            [
                "--base-eval",
                "pst",
                "--base-use-nnue",
                "false",
                "--base-blend",
                "0",
            ]
        )
    else:
        cmd.extend(
            [
                "--base-eval",
                "nnue",
                "--base-use-nnue",
                "true",
                "--base-blend",
                str(max(0, min(100, int(base_blend_percent)))),
                "--base-nnue-quant-file",
                str(base_quant),
            ]
        )
    subprocess.run(cmd, cwd=str(piebot_dir), check=True)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model gate JSON must be an object")
    points = payload.get("points")
    if not isinstance(points, dict) or "baseline" not in points or "experimental" not in points:
        raise ValueError("model gate JSON is missing baseline/experimental points")
    baseline = float(points["baseline"])
    experimental = float(points["experimental"])
    if not math.isfinite(baseline) or not math.isfinite(experimental):
        raise ValueError("model gate JSON contains non-finite points")
    reported_games = int(payload.get("games", -1))
    if reported_games != expected_games:
        raise ValueError(
            f"model gate JSON reports {reported_games} games, expected {expected_games}"
        )
    delta = experimental - baseline
    accepted = delta >= float(min_score_delta)
    return {
        "accepted": accepted,
        "baseline_points": baseline,
        "experimental_points": experimental,
        "delta_points": delta,
        "games": reported_games,
        "json_path": str(out_json),
        "baseline_blend_percent": 0
        if base_quant is None
        else max(0, min(100, int(base_blend_percent))),
        "experimental_blend_percent": max(
            0, min(100, int(candidate_blend_percent))
        ),
    }


def _run_confirmed_gate_attempt(
    *,
    piebot_dir: Path,
    screen_json: Path,
    confirmation_json: Path,
    base_quant: Optional[Path],
    candidate_quant: Path,
    screen_games: int,
    confirmation_games: int,
    movetime_ms: int,
    noise_plies: int,
    noise_topk: int,
    threads: int,
    seed: int,
    screen_min_score_delta: float,
    confirmation_min_score_delta: float,
    base_blend_percent: int,
    candidate_blend_percent: int,
    paired_openings: bool,
) -> Dict[str, Any]:
    screen = _run_model_gate(
        piebot_dir=piebot_dir,
        out_json=screen_json,
        base_quant=base_quant,
        candidate_quant=candidate_quant,
        games=screen_games,
        movetime_ms=movetime_ms,
        noise_plies=noise_plies,
        noise_topk=noise_topk,
        threads=threads,
        seed=seed,
        min_score_delta=screen_min_score_delta,
        base_blend_percent=base_blend_percent,
        candidate_blend_percent=candidate_blend_percent,
        paired_openings=paired_openings,
    )
    screen = dict(screen)
    screen["baseline_blend_percent"] = (
        0 if base_quant is None else max(0, min(100, int(base_blend_percent)))
    )
    screen["experimental_blend_percent"] = max(
        0, min(100, int(candidate_blend_percent))
    )
    attempt: Dict[str, Any] = {
        "blend_percent": max(0, min(100, int(candidate_blend_percent))),
        "accepted": False,
        "screen": screen,
        "confirmation": None,
    }
    if not bool(screen.get("accepted")):
        attempt["reason"] = "screen-rejected"
        return attempt

    if int(confirmation_games) <= 0:
        attempt["accepted"] = True
        attempt["reason"] = "confirmation-disabled"
        return attempt

    confirmation = _run_model_gate(
        piebot_dir=piebot_dir,
        out_json=confirmation_json,
        base_quant=base_quant,
        candidate_quant=candidate_quant,
        games=confirmation_games,
        movetime_ms=movetime_ms,
        noise_plies=noise_plies,
        noise_topk=noise_topk,
        threads=threads,
        seed=seed + 1_000_003,
        min_score_delta=confirmation_min_score_delta,
        base_blend_percent=base_blend_percent,
        candidate_blend_percent=candidate_blend_percent,
        paired_openings=paired_openings,
    )
    confirmation = dict(confirmation)
    confirmation["baseline_blend_percent"] = (
        0 if base_quant is None else max(0, min(100, int(base_blend_percent)))
    )
    confirmation["experimental_blend_percent"] = max(
        0, min(100, int(candidate_blend_percent))
    )
    attempt["confirmation"] = confirmation
    attempt["accepted"] = bool(confirmation.get("accepted"))
    attempt["reason"] = (
        "confirmation-accepted" if attempt["accepted"] else "confirmation-rejected"
    )
    return attempt


def _gate_from_attempts(attempts: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not attempts:
        return {"accepted": False, "reason": "no-gate-attempt"}
    selected = next((attempt for attempt in attempts if attempt.get("accepted")), attempts[-1])
    result = selected.get("confirmation") or selected.get("screen") or {}
    gate = dict(result) if isinstance(result, dict) else {}
    gate["accepted"] = bool(selected.get("accepted"))
    gate["reason"] = selected.get("reason")
    gate["experimental_blend_percent"] = int(selected["blend_percent"])
    gate["screen"] = selected.get("screen")
    gate["confirmation"] = selected.get("confirmation")
    gate["attempts"] = attempts
    return gate


def _record_acceptance(
    *,
    state: Dict[str, Any],
    cycle_idx: int,
    quant_path: Path,
    quant_sha256: Optional[str],
    gate: Dict[str, Any],
    blend_percent: int,
    model_identity: Optional[Dict[str, Any]],
) -> None:
    blend = max(0, min(100, int(blend_percent)))
    state["active_model_path"] = str(quant_path)
    state["active_model_sha256"] = quant_sha256
    state["active_model_blend_percent"] = blend
    state["active_model_identity"] = (
        dict(model_identity) if isinstance(model_identity, dict) else None
    )
    accepted = state.setdefault("accepted_models", [])
    if not isinstance(accepted, list):
        accepted = []
        state["accepted_models"] = accepted
    accepted.append(
        {
            "cycle": int(cycle_idx),
            "quant_path": str(quant_path),
            "quant_sha256": quant_sha256,
            "blend_percent": blend,
            "model_identity": (
                dict(model_identity) if isinstance(model_identity, dict) else None
            ),
            "accepted_at": time.time(),
            "gate": gate,
        }
    )


def _resolve_bootstrap_quant_path(state: Dict[str, Any]) -> Optional[Path]:
    return _resolve_active_quant_path(state)


def _filter_run_pipeline_kwargs(values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(inspect.signature(run_pipeline.run_pipeline).parameters)
    return {key: value for key, value in values.items() if key in allowed}


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = args.out_root
    state_path = out_root / "autopilot_state.json"
    lock_path = out_root / "autopilot.lock"

    with _single_instance_lock(lock_path):
        state = _load_state(state_path)
        now = time.time()
        if state is None:
            state = {
                "version": 1,
                "profile": args.profile,
                "started_at": now,
                "deadline_ts": now + (args.hours * 3600.0),
                "next_cycle": 1,
                "completed_cycles": [],
                "accepted_models": [],
                "active_model_path": None,
                "active_model_sha256": None,
                "active_model_blend_percent": 0,
                "active_model_identity": None,
                "deployment_state_version": 2,
                "training_lineage_start_cycle": 1,
                "training_model_identity": None,
                "last_error": None,
            }
            _atomic_write_json(state_path, state)

        try:
            if _migrate_deployment_state(state):
                _atomic_write_json(state_path, state)
            if args.reset_training_lineage_at_cycle is not None:
                state, _ = _atomic_reset_training_lineage(
                    state_path=state_path,
                    state=state,
                    start_cycle=args.reset_training_lineage_at_cycle,
                )
            _validate_training_lineage_floor(state)
        except ValueError as exc:
            state["last_error"] = {
                "stage": "training-lineage-reset",
                "error": str(exc),
                "ts": time.time(),
            }
            _atomic_write_json(state_path, state)
            print(f"autopilot refusing training lineage: {exc}", file=sys.stderr)
            return 2

        defaults = _profile_defaults(str(state.get("profile", args.profile)))
        defaults = _apply_cli_overrides(defaults, args)
        try:
            _validate_training_checkpoint_identity(state, defaults)
        except ValueError as exc:
            state["last_error"] = {
                "stage": "training-lineage-validation",
                "error": str(exc),
                "ts": time.time(),
            }
            _atomic_write_json(state_path, state)
            print(f"autopilot refusing incompatible training lineage: {exc}", file=sys.stderr)
            return 2
        retain_full_cycles = int(defaults.get("retain_full_cycles", 0))
        try:
            _enforce_cycle_retention(
                out_root=out_root,
                state=state,
                state_path=state_path,
                retain_full_cycles=retain_full_cycles,
            )
        except Exception as exc:
            state["last_error"] = {
                "stage": "retention",
                "error": str(exc),
                "ts": time.time(),
            }
            _atomic_write_json(state_path, state)
            print(f"autopilot aborting during retention cleanup: {exc}", file=sys.stderr)
            return 2
        completed = int(len(state.get("completed_cycles", [])))

        while True:
            now = time.time()
            if now >= float(state["deadline_ts"]):
                break
            if args.max_cycles > 0 and completed >= args.max_cycles:
                break

            cycle_idx = int(state["next_cycle"])
            cycle_dir = out_root / "cycles" / f"cycle_{cycle_idx:06d}"
            cycle_selfplay_seed = _derive_cycle_seed(
                int(defaults.get("selfplay_seed", 42)),
                cycle_idx,
                stream=0,
            )
            cycle_training_seed = _derive_cycle_seed(
                int(defaults.get("seed", 1)),
                cycle_idx,
                stream=1,
            )
            cycle_state = {
                "cycle": cycle_idx,
                "started_at": now,
                "out_dir": str(cycle_dir),
                "selfplay_seed": cycle_selfplay_seed,
                "training_seed": cycle_training_seed,
                "status": "running",
            }
            state["current_cycle"] = cycle_state
            _atomic_write_json(state_path, state)

            attempt = 0
            while True:
                try:
                    kwargs = _filter_run_pipeline_kwargs(defaults)
                    bootstrap_quant = _resolve_bootstrap_quant_path(state)
                    teacher_quant, teacher_blend = _resolve_teacher_quant_and_blend(
                        state,
                        int(defaults.get("teacher_lag_cycles", 0)),
                    )
                    active_blend = _active_model_blend_percent(state)
                    replay_dirs = _collect_replay_jsonl_dirs(
                        state,
                        int(defaults.get("replay_window_cycles", 0)),
                    )
                    initial_checkpoint = None
                    if bool(defaults.get("warm_start", True)):
                        initial_checkpoint = _resolve_training_checkpoint_path(
                            state,
                            defaults.get("initial_checkpoint"),
                        )
                    cycle_state["initial_checkpoint_path"] = (
                        str(initial_checkpoint) if initial_checkpoint is not None else None
                    )
                    cycle_learning_rate = float(defaults.get("learning_rate", 0.03))
                    if initial_checkpoint is not None:
                        cycle_learning_rate = float(
                            defaults.get("warm_start_learning_rate", cycle_learning_rate)
                        )
                    kwargs.update(
                        {
                            "out_dir": cycle_dir,
                            "piebot_dir": args.piebot_dir,
                            "resume": True,
                            "selfplay_seed": cycle_selfplay_seed,
                            "seed": cycle_training_seed,
                            "selfplay_nnue_quant_file": bootstrap_quant,
                            "selfplay_nnue_blend_percent": active_blend,
                            "teacher_relabel_nnue_quant_file": teacher_quant,
                            "teacher_relabel_nnue_blend_percent": teacher_blend,
                            "replay_jsonl_dirs": replay_dirs,
                            "initial_checkpoint": initial_checkpoint,
                            "learning_rate": cycle_learning_rate,
                        }
                    )
                    summary = run_pipeline.run_pipeline(**kwargs)
                    candidate_checkpoint = (
                        _path_if_exists(summary.get("checkpoint_path"))
                        if isinstance(summary, dict)
                        else None
                    )
                    if candidate_checkpoint is None:
                        raise ValueError("pipeline summary is missing the training checkpoint")
                    candidate_checkpoint_sha = _sha256_file(candidate_checkpoint)
                    reported_checkpoint_sha = (
                        summary.get("checkpoint_sha256")
                        if isinstance(summary, dict)
                        else None
                    )
                    if (
                        reported_checkpoint_sha
                        and reported_checkpoint_sha != candidate_checkpoint_sha
                    ):
                        raise ValueError("candidate training checkpoint SHA-256 mismatch")
                    candidate_quant = (
                        _path_if_exists(summary.get("quant_path")) if isinstance(summary, dict) else None
                    )
                    candidate_quant_sha = None
                    if candidate_quant is not None:
                        candidate_quant_sha = _sha256_file(candidate_quant)
                        reported_quant_sha = (
                            summary.get("quant_sha256")
                            if isinstance(summary, dict)
                            else None
                        )
                        if reported_quant_sha and reported_quant_sha != candidate_quant_sha:
                            raise ValueError("candidate quantized model SHA-256 mismatch")
                    baseline_quant_sha = (
                        _sha256_file(bootstrap_quant)
                        if bootstrap_quant is not None
                        else None
                    )
                    candidate_identity = (
                        _quant_model_identity(candidate_quant, summary=summary)
                        if candidate_quant is not None
                        else None
                    )
                    active_identity = _active_model_identity(state)
                    candidate_blend = _candidate_model_blend_percent(
                        state,
                        candidate_identity=candidate_identity,
                    )
                    same_lineage = _model_identities_same(
                        active_identity,
                        candidate_identity,
                    )
                    candidate_blends = [candidate_blend]
                    if (
                        bootstrap_quant is not None
                        and same_lineage
                        and active_blend > 0
                        and candidate_blend > active_blend
                    ):
                        candidate_blends.append(active_blend)
                    gate_identity = {
                        "quant_sha256": candidate_quant_sha,
                        "baseline_quant_sha256": baseline_quant_sha,
                        "baseline_model_identity": active_identity,
                        "candidate_model_identity": candidate_identity,
                        "baseline_blend_percent": active_blend,
                        "candidate_blend_percents": candidate_blends,
                        "games": int(defaults.get("gate_games", 0)),
                        "paired_openings": bool(
                            defaults.get("gate_paired_openings", True)
                        ),
                        "confirmation_games": int(
                            defaults.get("gate_confirmation_games", 96)
                        ),
                        "confirmation_min_score_delta": float(
                            defaults.get("gate_confirmation_min_score_delta", 2.0)
                        ),
                        "movetime_ms": int(defaults.get("gate_movetime_ms", 150)),
                        "noise_plies": int(defaults.get("gate_noise_plies", 12)),
                        "noise_topk": int(defaults.get("gate_noise_topk", 5)),
                        "threads": int(defaults.get("gate_threads", 1)),
                        "min_score_delta": float(
                            defaults.get("gate_min_score_delta", 0.0)
                        ),
                    }
                    gate_games = int(defaults.get("gate_games", 0))
                    gate_was_run = False
                    if gate_games <= 0:
                        gate = {
                            "accepted": True,
                            "reason": "gate-disabled",
                            "baseline_blend_percent": active_blend,
                            "experimental_blend_percent": candidate_blend,
                            "screen": None,
                            "confirmation": None,
                            "attempts": [],
                        }
                    elif candidate_quant is None:
                        gate = {"accepted": False, "reason": "missing-candidate-model"}
                    elif (
                        baseline_quant_sha
                        and candidate_quant_sha == baseline_quant_sha
                        and candidate_blend == active_blend
                    ):
                        gate = {
                            "accepted": False,
                            "reason": "candidate-identical-to-active-model",
                            "games": 0,
                            "quant_sha256": candidate_quant_sha,
                            "baseline_blend_percent": active_blend,
                            "experimental_blend_percent": candidate_blend,
                        }
                    elif (
                        candidate_quant_sha
                        and state.get("last_gate_identity") == gate_identity
                    ):
                        gate = {
                            "accepted": False,
                            "reason": "unchanged-training-checkpoint",
                            "games": 0,
                            "quant_sha256": candidate_quant_sha,
                            "baseline_blend_percent": active_blend,
                            "experimental_blend_percent": candidate_blend,
                        }
                    else:
                        gate_attempts: list[Dict[str, Any]] = []
                        for blend_idx, blend in enumerate(candidate_blends):
                            if (
                                blend_idx > 0
                                and baseline_quant_sha
                                and candidate_quant_sha == baseline_quant_sha
                                and blend == active_blend
                            ):
                                gate_attempts.append(
                                    {
                                        "blend_percent": blend,
                                        "accepted": False,
                                        "reason": "candidate-identical-to-active-model",
                                        "screen": None,
                                        "confirmation": None,
                                    }
                                )
                                continue
                            fallback = blend_idx > 0
                            stem = "gate_compare_same_blend" if fallback else "gate_compare"
                            gate_was_run = True
                            gate_attempt = _run_confirmed_gate_attempt(
                                piebot_dir=args.piebot_dir,
                                screen_json=cycle_dir / f"{stem}.json",
                                confirmation_json=cycle_dir / f"{stem}_confirmation.json",
                                base_quant=bootstrap_quant,
                                candidate_quant=candidate_quant,
                                screen_games=gate_games,
                                confirmation_games=int(
                                    defaults.get("gate_confirmation_games", 96)
                                ),
                                movetime_ms=int(defaults.get("gate_movetime_ms", 150)),
                                noise_plies=int(defaults.get("gate_noise_plies", 12)),
                                noise_topk=int(defaults.get("gate_noise_topk", 5)),
                                threads=int(defaults.get("gate_threads", 1)),
                                seed=int(defaults.get("gate_seed", 1)) + cycle_idx,
                                screen_min_score_delta=float(
                                    defaults.get("gate_min_score_delta", 0.0)
                                ),
                                confirmation_min_score_delta=float(
                                    defaults.get(
                                        "gate_confirmation_min_score_delta",
                                        2.0,
                                    )
                                ),
                                base_blend_percent=active_blend,
                                candidate_blend_percent=blend,
                                paired_openings=bool(
                                    defaults.get("gate_paired_openings", True)
                                ),
                            )
                            gate_attempts.append(gate_attempt)
                            if gate_attempt.get("accepted"):
                                break
                        gate = _gate_from_attempts(gate_attempts)
                    next_state = copy.deepcopy(state)
                    if gate_was_run and candidate_quant_sha:
                        next_state["last_gate_identity"] = gate_identity
                        next_state["last_gated_quant_sha256"] = candidate_quant_sha
                    if gate.get("accepted") and candidate_quant is not None:
                        _record_acceptance(
                            state=next_state,
                            cycle_idx=cycle_idx,
                            quant_path=candidate_quant,
                            quant_sha256=candidate_quant_sha,
                            gate=gate,
                            blend_percent=int(
                                gate.get("experimental_blend_percent", candidate_blend)
                            ),
                            model_identity=candidate_identity,
                        )
                    completed_cycle_state = dict(cycle_state)
                    completed_cycle_state["status"] = "completed"
                    completed_cycle_state["completed_at"] = time.time()
                    completed_cycle_state["summary_path"] = str(
                        cycle_dir / "pipeline_summary.json"
                    )
                    completed_cycle_state["jsonl_dir"] = (
                        summary.get("jsonl_dir") if isinstance(summary, dict) else None
                    )
                    completed_cycle_state["train_jsonl_dir"] = (
                        summary.get("train_jsonl_dir") if isinstance(summary, dict) else None
                    )
                    completed_cycle_state["checkpoint_path"] = str(
                        candidate_checkpoint
                    )
                    completed_cycle_state["checkpoint_sha256"] = (
                        candidate_checkpoint_sha
                    )
                    completed_cycle_state["quant_path"] = (
                        summary.get("quant_path") if isinstance(summary, dict) else None
                    )
                    completed_cycle_state["gate"] = gate
                    next_state["current_cycle"] = completed_cycle_state
                    next_state.setdefault("completed_cycles", []).append(
                        completed_cycle_state
                    )
                    next_state["next_cycle"] = cycle_idx + 1
                    next_state["last_error"] = None
                    next_state["last_summary"] = summary
                    next_state["last_gate"] = gate
                    next_state["training_checkpoint_path"] = str(candidate_checkpoint)
                    next_state["training_checkpoint_sha256"] = candidate_checkpoint_sha
                    next_state["training_model_identity"] = (
                        dict(candidate_identity)
                        if isinstance(candidate_identity, dict)
                        else None
                    )
                    _atomic_write_json(state_path, next_state)
                    state = next_state
                    completed += 1
                    break
                except Exception as exc:
                    attempt += 1
                    state["last_error"] = {
                        "cycle": cycle_idx,
                        "attempt": attempt,
                        "error": str(exc),
                        "ts": time.time(),
                    }
                    _atomic_write_json(state_path, state)
                    if attempt >= args.retry_limit:
                        print(
                            f"autopilot aborting: cycle {cycle_idx} failed after {attempt} attempts: {exc}",
                            file=sys.stderr,
                        )
                        return 2
                    time.sleep(max(1.0, args.retry_backoff_sec))

            # Cycle completion is durably recorded before any artifact removal.
            # A crash during cleanup is reconciled by the startup pass above.
            try:
                _enforce_cycle_retention(
                    out_root=out_root,
                    state=state,
                    state_path=state_path,
                    retain_full_cycles=retain_full_cycles,
                )
            except Exception as exc:
                state["last_error"] = {
                    "cycle": cycle_idx,
                    "stage": "retention",
                    "error": str(exc),
                    "ts": time.time(),
                }
                _atomic_write_json(state_path, state)
                print(f"autopilot aborting during retention cleanup: {exc}", file=sys.stderr)
                return 2

        state["finished_at"] = time.time()
        state["status"] = "complete"
        _atomic_write_json(state_path, state)
        print(f"Autopilot finished. Completed cycles: {len(state.get('completed_cycles', []))}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
