#!/usr/bin/env python3
"""Set-and-forget NNUE training autopilot with crash-safe resume."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
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
    """Defaults tuned for a 7-day unattended run on Zen5 9755."""
    return {
        "selfplay_games": 12_000,
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
        "teacher_relabel_depth": 5,
        "teacher_relabel_every": 8,
        "teacher_relabel_threads": 48,
        "teacher_relabel_hash_mb": 4096,
        "teacher_relabel_max_records": 0,
        "batch_size": 4096,
        "max_samples": 350_000,
        "epochs": 2,
        "hidden_dim": 64,
        "target_cp": 100.0,
        "teacher_mix": 0.8,
        "max_teacher_cp": 1200.0,
        "learning_rate": 0.03,
        "val_split": 0.1,
        "seed": 1,
        "trainer_backend": "auto",
        "trainer_device": "cuda",
        "resume": True,
        "retain_full_cycles": 0,
        "replay_window_cycles": 6,
        "teacher_lag_cycles": 1,
        "gate_games": 24,
        "gate_movetime_ms": 150,
        "gate_noise_plies": 12,
        "gate_noise_topk": 5,
        "gate_threads": 1,
        "gate_seed": 1,
        "gate_min_score_delta": 0.0,
    }


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


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
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
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


def _active_model_blend_percent(state: Dict[str, Any]) -> int:
    accepted = state.get("accepted_models")
    if not isinstance(accepted, list):
        return 0
    accepted_count = len(accepted)
    if accepted_count <= 0:
        return 0
    ramp = (25, 50, 75, 100)
    return int(ramp[min(accepted_count - 1, len(ramp) - 1)])


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
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
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


def _resolve_active_quant_path(state: Dict[str, Any]) -> Optional[Path]:
    if "active_model_path" in state:
        return _path_if_exists(state.get("active_model_path"))
    # Backward compatibility: older state schema used last_summary only.
    last_summary = state.get("last_summary")
    if isinstance(last_summary, dict):
        return _path_if_exists(last_summary.get("quant_path"))
    return None


def _resolve_teacher_quant_path(state: Dict[str, Any], lag_cycles: int) -> Optional[Path]:
    lag = max(0, int(lag_cycles))
    accepted = state.get("accepted_models")
    if isinstance(accepted, list) and accepted:
        idx = len(accepted) - 1 - lag
        if idx >= 0 and isinstance(accepted[idx], dict):
            teacher = _path_if_exists(accepted[idx].get("quant_path"))
            if teacher is not None:
                return teacher
    return _resolve_active_quant_path(state)


def _collect_replay_jsonl_dirs(state: Dict[str, Any], window_cycles: int) -> list[Path]:
    window = max(0, int(window_cycles))
    if window == 0:
        return []
    completed = state.get("completed_cycles")
    if not isinstance(completed, list):
        return []
    out: list[Path] = []
    for c in reversed(completed):
        if not isinstance(c, dict):
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
            for key in ("jsonl_dir", "train_jsonl_dir", "quant_path", "summary_path"):
                if entry.get(key) is not None:
                    entry[key] = None
                    report["state_changed"] = True
            continue

        if not cycle_dir.is_dir():
            raise ValueError(f"retention accepted cycle directory is missing: {cycle_dir}")
        if entry.get("retention") != "model_only":
            entry["retention"] = "model_only"
            report["state_changed"] = True
        for key in ("jsonl_dir", "train_jsonl_dir"):
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
) -> Dict[str, Any]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.unlink(missing_ok=True)
    expected_games = max(2, int(games))
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
        "100",
        "--exp-nnue-quant-file",
        str(candidate_quant),
    ]
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
                "100",
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
    }


def _record_acceptance(
    *,
    state: Dict[str, Any],
    cycle_idx: int,
    quant_path: Path,
    gate: Dict[str, Any],
) -> None:
    state["active_model_path"] = str(quant_path)
    accepted = state.setdefault("accepted_models", [])
    if not isinstance(accepted, list):
        accepted = []
        state["accepted_models"] = accepted
    accepted.append(
        {
            "cycle": int(cycle_idx),
            "quant_path": str(quant_path),
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
                "last_error": None,
            }
            _atomic_write_json(state_path, state)

        defaults = _profile_defaults(str(state.get("profile", args.profile)))
        defaults = _apply_cli_overrides(defaults, args)
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
                    teacher_quant = _resolve_teacher_quant_path(
                        state,
                        int(defaults.get("teacher_lag_cycles", 0)),
                    )
                    active_blend = _active_model_blend_percent(state)
                    replay_dirs = _collect_replay_jsonl_dirs(
                        state,
                        int(defaults.get("replay_window_cycles", 0)),
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
                            "teacher_relabel_nnue_blend_percent": active_blend,
                            "replay_jsonl_dirs": replay_dirs,
                        }
                    )
                    summary = run_pipeline.run_pipeline(**kwargs)
                    candidate_quant = (
                        _path_if_exists(summary.get("quant_path")) if isinstance(summary, dict) else None
                    )
                    gate_games = int(defaults.get("gate_games", 0))
                    if gate_games <= 0:
                        gate = {"accepted": True, "reason": "gate-disabled"}
                    elif candidate_quant is None:
                        gate = {"accepted": False, "reason": "missing-candidate-model"}
                    else:
                        gate = _run_model_gate(
                            piebot_dir=args.piebot_dir,
                            out_json=cycle_dir / "gate_compare.json",
                            base_quant=bootstrap_quant,
                            candidate_quant=candidate_quant,
                            games=gate_games,
                            movetime_ms=int(defaults.get("gate_movetime_ms", 150)),
                            noise_plies=int(defaults.get("gate_noise_plies", 12)),
                            noise_topk=int(defaults.get("gate_noise_topk", 5)),
                            threads=int(defaults.get("gate_threads", 1)),
                            seed=int(defaults.get("gate_seed", 1)) + cycle_idx,
                            min_score_delta=float(defaults.get("gate_min_score_delta", 0.0)),
                        )
                    if gate.get("accepted") and candidate_quant is not None:
                        _record_acceptance(
                            state=state,
                            cycle_idx=cycle_idx,
                            quant_path=candidate_quant,
                            gate=gate,
                        )
                    cycle_state["status"] = "completed"
                    cycle_state["completed_at"] = time.time()
                    cycle_state["summary_path"] = str(cycle_dir / "pipeline_summary.json")
                    cycle_state["jsonl_dir"] = summary.get("jsonl_dir") if isinstance(summary, dict) else None
                    cycle_state["train_jsonl_dir"] = (
                        summary.get("train_jsonl_dir") if isinstance(summary, dict) else None
                    )
                    cycle_state["quant_path"] = summary.get("quant_path") if isinstance(summary, dict) else None
                    cycle_state["gate"] = gate
                    state.setdefault("completed_cycles", []).append(cycle_state)
                    state["next_cycle"] = cycle_idx + 1
                    state["last_error"] = None
                    state["last_summary"] = summary
                    state["last_gate"] = gate
                    completed += 1
                    _atomic_write_json(state_path, state)
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
