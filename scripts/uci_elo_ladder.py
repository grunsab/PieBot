#!/usr/bin/env python3
"""Parallel multi-rung Stockfish anchor ladder built on uci_elo_arena.py.

Runs one arena process per rung (each rung internally sequential, so the
per-game calibration conditions are preserved) and pools the per-rung
results into a single maximum-likelihood strength estimate with a
bootstrap confidence interval.

The pooled estimate solves the classic performance-rating equation: find R
such that the expected total score against the rung opponents equals the
observed total score. Expected per-game score against a rung at Elo E is
the logistic 1 / (1 + 10^((E - R) / 400)).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ARENA_SCRIPT = Path(__file__).resolve().parent / "uci_elo_arena.py"
_SOLVE_LO = -4000.0
_SOLVE_HI = 8000.0


def _expected_total_score(rating: float, rungs: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    for rung in rungs:
        games = float(rung["games"])
        total += games / (1.0 + math.pow(10.0, (float(rung["elo"]) - rating) / 400.0))
    return total


def _solve_performance_rating(
    rungs: Sequence[Dict[str, Any]], observed_score: float
) -> float:
    lo, hi = _SOLVE_LO, _SOLVE_HI
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _expected_total_score(mid, rungs) < observed_score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def pooled_elo_estimate(
    rungs: Sequence[Dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Pool per-rung results into one strength estimate.

    Each rung dict needs: ``elo`` (anchor UCI_Elo), ``score_points``
    (total points scored), ``games`` (game count). Returns a dict with
    ``estimate`` (None when the pooled score is degenerate 0% or 100%),
    ``ci_95`` and ``degenerate``.
    """
    if not rungs:
        raise ValueError("pooled estimate requires at least one rung")
    total_games = sum(float(r["games"]) for r in rungs)
    total_score = sum(float(r["score_points"]) for r in rungs)
    if total_games <= 0:
        raise ValueError("pooled estimate requires played games")
    if total_score <= 0.0 or total_score >= total_games:
        return {"estimate": None, "ci_95": [None, None], "degenerate": True}

    estimate = _solve_performance_rating(rungs, total_score)

    # Bootstrap: resample each rung's per-game scores as Bernoulli draws at
    # the rung's observed rate (draws contribute through the rate itself),
    # re-solve, and take the empirical 2.5/97.5 percentiles.
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(bootstrap_samples):
        resampled_score = 0.0
        for rung in rungs:
            games = int(rung["games"])
            rate = float(rung["score_points"]) / float(rung["games"])
            wins = sum(1 for _ in range(games) if rng.random() < rate)
            resampled_score += float(wins)
        resampled_score = min(max(resampled_score, 0.5), total_games - 0.5)
        samples.append(_solve_performance_rating(rungs, resampled_score))
    samples.sort()
    lo_index = int(0.025 * (len(samples) - 1))
    hi_index = int(0.975 * (len(samples) - 1))
    return {
        "estimate": estimate,
        "ci_95": [samples[lo_index], samples[hi_index]],
        "degenerate": False,
    }


def build_rung_commands(args: argparse.Namespace) -> List[List[str]]:
    """One uci_elo_arena.py invocation per rung with a distinct results file."""
    commands: List[List[str]] = []
    for rung in args.rungs:
        results = Path(args.out_dir) / f"rung_{rung}.json"
        commands.append(
            [
                sys.executable,
                str(_ARENA_SCRIPT),
                "--piebot-command",
                str(args.piebot_command),
                "--piebot-nnue",
                str(args.piebot_nnue),
                "--piebot-blend",
                str(args.piebot_blend),
                "--stockfish-command",
                str(args.stockfish_command),
                "--stockfish-elo",
                str(rung),
                "--games",
                str(args.games),
                "--time-control",
                str(args.time_control),
                "--seed",
                str(int(args.seed) + int(rung)),
                "--results",
                str(results),
            ]
        )
    return commands


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piebot-command", required=True)
    parser.add_argument("--piebot-nnue", required=True)
    parser.add_argument("--piebot-blend", type=int, default=100)
    parser.add_argument("--stockfish-command", default="stockfish")
    parser.add_argument(
        "--rungs",
        type=lambda raw: [int(item) for item in raw.split(",") if item],
        required=True,
        help="comma-separated anchor UCI_Elo rungs, e.g. 1320,1500",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-control", default="60+0.5")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = build_rung_commands(args)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    processes = [subprocess.Popen(command) for command in commands]
    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        print(f"rung processes failed: {exit_codes}", file=sys.stderr)
        return 1

    rungs: List[Dict[str, Any]] = []
    for rung, command in zip(args.rungs, commands):
        results_path = Path(command[command.index("--results") + 1])
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        rungs.append(
            {
                "elo": rung,
                "score_points": summary["score_points"],
                "games": summary["games"],
                "per_rung_elo_difference": summary.get("elo_difference"),
                "per_rung_ci": summary.get("elo_95_ci"),
            }
        )

    pooled = pooled_elo_estimate(rungs, seed=args.seed)
    report = {
        "schema": "piebot-uci-elo-ladder-v1",
        "rungs": rungs,
        "pooled": pooled,
    }
    report_path = args.out_dir / "ladder_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
