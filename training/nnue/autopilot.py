#!/usr/bin/env python3
"""Set-and-forget NNUE training autopilot with crash-safe resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import run_pipeline
except Exception:
    import run_pipeline  # type: ignore

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def zen5_9755_7d_profile() -> Dict[str, Any]:
    """Defaults tuned for a 7-day unattended run on Zen5 9755."""
    return {
        "selfplay_games": 12_000,
        "selfplay_max_plies": 160,
        "selfplay_threads": 24,
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
def _single_instance_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(str(os.getpid()))
        fd.flush()
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()


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
    }
    for k, v in mapping.items():
        if v is not None:
            out[k] = v
    return out


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
                    kwargs.update(
                        {
                            "out_dir": cycle_dir,
                            "piebot_dir": args.piebot_dir,
                            "resume": True,
                        }
                    )
                    summary = run_pipeline.run_pipeline(**kwargs)
                    cycle_state["status"] = "completed"
                    cycle_state["completed_at"] = time.time()
                    cycle_state["summary_path"] = str(cycle_dir / "pipeline_summary.json")
                    state.setdefault("completed_cycles", []).append(cycle_state)
                    state["next_cycle"] = cycle_idx + 1
                    state["last_error"] = None
                    state["last_summary"] = summary
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
