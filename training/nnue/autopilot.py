#!/usr/bin/env python3
"""Set-and-forget NNUE training autopilot with crash-safe resume."""

from __future__ import annotations

import argparse
import json
import os
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
        "teacher_relabel_depth": 9,
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
        "trainer_backend": "auto",
        "trainer_device": "cuda",
        "resume": True,
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
    active = _path_if_exists(state.get("active_model_path"))
    if active is not None:
        return active
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
        p = _path_if_exists(c.get("train_jsonl_dir")) or _path_if_exists(c.get("jsonl_dir"))
        if p is None:
            continue
        if any(x.resolve() == p.resolve() for x in out):
            continue
        out.append(p)
        if len(out) >= window:
            break
    return out


def _run_model_gate(
    *,
    piebot_dir: Path,
    out_json: Path,
    base_quant: Path,
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
    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "compare_play",
        "--",
        "--games",
        str(max(2, int(games))),
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
        "--base-eval",
        "nnue",
        "--exp-eval",
        "nnue",
        "--base-use-nnue",
        "true",
        "--exp-use-nnue",
        "true",
        "--base-blend",
        "100",
        "--exp-blend",
        "100",
        "--base-nnue-quant-file",
        str(base_quant),
        "--exp-nnue-quant-file",
        str(candidate_quant),
    ]
    subprocess.run(cmd, cwd=str(piebot_dir), check=True)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    points = payload.get("points", {}) if isinstance(payload, dict) else {}
    baseline = float(points.get("baseline", 0.0))
    experimental = float(points.get("experimental", 0.0))
    delta = experimental - baseline
    accepted = delta >= float(min_score_delta)
    return {
        "accepted": accepted,
        "baseline_points": baseline,
        "experimental_points": experimental,
        "delta_points": delta,
        "games": int(payload.get("games", max(2, int(games)))) if isinstance(payload, dict) else max(2, int(games)),
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
        completed = int(len(state.get("completed_cycles", [])))

        while True:
            now = time.time()
            if now >= float(state["deadline_ts"]):
                break
            if args.max_cycles > 0 and completed >= args.max_cycles:
                break

            cycle_idx = int(state["next_cycle"])
            cycle_dir = out_root / "cycles" / f"cycle_{cycle_idx:06d}"
            cycle_state = {
                "cycle": cycle_idx,
                "started_at": now,
                "out_dir": str(cycle_dir),
                "status": "running",
            }
            state["current_cycle"] = cycle_state
            _atomic_write_json(state_path, state)

            attempt = 0
            while True:
                try:
                    kwargs = dict(defaults)
                    bootstrap_quant = _resolve_bootstrap_quant_path(state)
                    teacher_quant = _resolve_teacher_quant_path(
                        state,
                        int(defaults.get("teacher_lag_cycles", 0)),
                    )
                    replay_dirs = _collect_replay_jsonl_dirs(
                        state,
                        int(defaults.get("replay_window_cycles", 0)),
                    )
                    kwargs.update(
                        {
                            "out_dir": cycle_dir,
                            "piebot_dir": args.piebot_dir,
                            "resume": True,
                            "selfplay_nnue_quant_file": bootstrap_quant,
                            "teacher_relabel_nnue_quant_file": teacher_quant,
                            "replay_jsonl_dirs": replay_dirs,
                        }
                    )
                    summary = run_pipeline.run_pipeline(**kwargs)
                    candidate_quant = (
                        _path_if_exists(summary.get("quant_path")) if isinstance(summary, dict) else None
                    )
                    gate_games = int(defaults.get("gate_games", 0))
                    if bootstrap_quant is None:
                        gate: Dict[str, Any] = {"accepted": True, "reason": "bootstrap-first-model"}
                    elif gate_games <= 0:
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

        state["finished_at"] = time.time()
        state["status"] = "complete"
        _atomic_write_json(state_path, state)
        print(f"Autopilot finished. Completed cycles: {len(state.get('completed_cycles', []))}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
