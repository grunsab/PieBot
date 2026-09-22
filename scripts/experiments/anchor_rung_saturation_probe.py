#!/usr/bin/env python3
"""Play one strength-limited Stockfish rung against another.

The anchored ladder derives PieBot's rating by subtracting its Elo deficit
from a rung's nominal ``UCI_Elo``. That arithmetic assumes the rungs are
actually as far apart as their labels claim. The 2026-08-16 ladder suggests
they are not: PieBot scored 15.5% against a nominal 3000 and 14.5% against a
nominal 3190, implying those opponents differ by ~13.6 Elo rather than 190.

That inference runs through PieBot, so it could in principle be an artefact of
how PieBot fails rather than of how Stockfish is limited. This probe removes
PieBot from the measurement: it plays the two rungs directly against each
other. If they score near 50%, the limiter is saturated and the ladder's scale
is fiction. If the stronger rung scores near 75%, the labels are honest and the
ladder's disagreement needs another explanation.

Deliberately mirrors ``scripts/uci_elo_arena.py``: same UCI options, same
whole-game clock semantics, same paired openings played twice with colours
reversed. Anything that differs would reopen the question this is meant to
close.

Evaluation only. Nothing here produces training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine

MAX_PLIES = 400

# python-chess drives these itself and `configure()` rejects them. The ladder
# filters the same two names for the same reason; keep the lists identical.
PYTHON_CHESS_MANAGED_OPTIONS = {"ponder", "multipv"}


def configurable(options: dict) -> dict:
    return {
        name: value
        for name, value in options.items()
        if name.casefold() not in PYTHON_CHESS_MANAGED_OPTIONS
    }


def rung_options(elo: int, hash_mb: int) -> dict:
    """UCI options for one rung, identical to the ladder's anchor settings."""
    if elo <= 0:
        raise ValueError("UCI_Elo must be positive")
    if hash_mb <= 0:
        raise ValueError("hash size must be positive")
    return {
        "Threads": 1,
        "Hash": hash_mb,
        "Ponder": False,
        "MultiPV": 1,
        "SyzygyProbeLimit": 0,
        "UCI_LimitStrength": True,
        "UCI_Elo": elo,
    }


def play_game(engines, board, initial_s, increment_s, high_is_white):
    """Play one game. Returns (score_for_high_rung, termination)."""
    clocks = {chess.WHITE: initial_s, chess.BLACK: initial_s}
    plies = 0

    while plies < MAX_PLIES:
        if board.is_game_over(claim_draw=True):
            break
        turn = board.turn
        high_to_move = (turn == chess.WHITE) == high_is_white
        engine = engines["high"] if high_to_move else engines["low"]

        if clocks[turn] <= 0:
            return (0.0 if high_to_move else 1.0), "time_forfeit"

        limit = chess.engine.Limit(
            white_clock=clocks[chess.WHITE],
            black_clock=clocks[chess.BLACK],
            white_inc=increment_s,
            black_inc=increment_s,
        )
        started = time.monotonic()
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineError:
            return (0.0 if high_to_move else 1.0), "engine_error"
        elapsed = time.monotonic() - started
        clocks[turn] = clocks[turn] - elapsed + increment_s
        if clocks[turn] <= 0:
            return (0.0 if high_to_move else 1.0), "time_forfeit"
        if result.move is None:
            break
        board.push(result.move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5, "ply_cap"
    if outcome.winner is None:
        return 0.5, outcome.termination.name.lower()
    high_won = (outcome.winner == chess.WHITE) == high_is_white
    return (1.0 if high_won else 0.0), outcome.termination.name.lower()


def play_pair(task):
    """Play one opening twice with colours reversed. Returns two results."""
    command, high_elo, low_elo, hash_mb, fen, initial_s, increment_s = task
    results = []
    high = chess.engine.SimpleEngine.popen_uci(command)
    low = chess.engine.SimpleEngine.popen_uci(command)
    try:
        high.configure(configurable(rung_options(high_elo, hash_mb)))
        low.configure(configurable(rung_options(low_elo, hash_mb)))
        engines = {"high": high, "low": low}
        for high_is_white in (True, False):
            board = chess.Board(fen)
            score, termination = play_game(
                engines, board, initial_s, increment_s, high_is_white
            )
            results.append(
                {
                    "fen": fen,
                    "high_is_white": high_is_white,
                    "high_score": score,
                    "termination": termination,
                }
            )
    finally:
        high.quit()
        low.quit()
    return results


def elo_from_score(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="path to the Stockfish binary")
    ap.add_argument("--high-elo", type=int, required=True)
    ap.add_argument("--low-elo", type=int, required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--time-control", default="60+0.5")
    ap.add_argument("--book", required=True, help="FEN-per-line opening book")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    initial_raw, increment_raw = args.time_control.split("+")
    initial_s, increment_s = float(initial_raw), float(increment_raw)
    if args.games % 2:
        raise SystemExit("--games must be even (openings are played in pairs)")

    openings = [
        line.strip()
        for line in Path(args.book).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not openings:
        raise SystemExit("opening book is empty")

    rng = random.Random(args.seed)
    pairs = args.games // 2
    chosen = [rng.choice(openings) for _ in range(pairs)]
    tasks = [
        (
            args.engine,
            args.high_elo,
            args.low_elo,
            args.hash_mb,
            fen,
            initial_s,
            increment_s,
        )
        for fen in chosen
    ]

    started = time.time()
    games: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(play_pair, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            games.extend(future.result())
            if done % 10 == 0:
                partial = sum(g["high_score"] for g in games) / len(games)
                print(
                    f"pairs {done}/{pairs}  games={len(games)}  "
                    f"high_score={partial:.4f}",
                    flush=True,
                )

    points = sum(g["high_score"] for g in games)
    n = len(games)
    score = points / n
    wins = sum(1 for g in games if g["high_score"] == 1.0)
    draws = sum(1 for g in games if g["high_score"] == 0.5)
    losses = sum(1 for g in games if g["high_score"] == 0.0)

    mean_sq = sum(g["high_score"] ** 2 for g in games) / n
    variance = max(mean_sq - score * score, 0.0)
    stderr = math.sqrt(variance / n)
    lo, hi = score - 1.96 * stderr, score + 1.96 * stderr

    summary = {
        "high_elo": args.high_elo,
        "low_elo": args.low_elo,
        "nominal_gap_elo": args.high_elo - args.low_elo,
        "games": n,
        "wins_high": wins,
        "draws": draws,
        "losses_high": losses,
        "high_score_rate": score,
        "score_95_ci": [lo, hi],
        "measured_gap_elo": elo_from_score(score),
        "measured_gap_95_ci_elo": [elo_from_score(lo), elo_from_score(hi)],
        "time_control": args.time_control,
        "hash_mb": args.hash_mb,
        "seed": args.seed,
        "wall_time_s": time.time() - started,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "games": games}, indent=2))

    print(json.dumps(summary, indent=2))
    measured = summary["measured_gap_elo"]
    if measured is not None:
        print(
            f"\nnominal gap {summary['nominal_gap_elo']} Elo -> "
            f"measured {measured:.1f} Elo"
        )


if __name__ == "__main__":
    main()
