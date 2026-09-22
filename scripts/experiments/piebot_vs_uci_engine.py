#!/usr/bin/env python3
"""Play PieBot against an arbitrary CCRL-listed UCI engine.

This is the measurement the campaign has never had. Every rating estimate to
date came from strength-limited Stockfish, and 2026-08-16 established that
instrument is unusable for an engine this far below it: `UCI_LimitStrength`
injects error without regard to position type, so it squanders won positions
and gifted PieBot ~30% of games as repetition draws -- 19 of 60 from positions
with a forced mate available. Draws collapsed 30% -> 3.3% once the limiter was
removed. See evidence/ladder_draws_are_unconverted_wins_20260816.json.

A real opponent with a published CCRL rating has no such failure mode. Playing
one puts PieBot on a scale other people can check.

Deliberately generic about the opponent: it sends ONLY the UCI options that
engine advertises in its own `uci` response, because assuming Stockfish's
option set is what made scripts/uci_elo_arena.py unusable here (Blunder has no
Threads, no SyzygyProbeLimit, no UCI_LimitStrength).

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
PYTHON_CHESS_MANAGED = {"ponder", "multipv"}


def advertised_options(command: list[str]) -> set[str]:
    """Option names the engine actually declares, casefolded."""
    eng = chess.engine.SimpleEngine.popen_uci(command)
    try:
        return {name.casefold() for name in eng.options}
    finally:
        eng.quit()


def safe_configure(engine, desired: dict, supported: set[str]) -> dict:
    """Apply only options the engine advertises and python-chess allows."""
    applied = {
        k: v
        for k, v in desired.items()
        if k.casefold() in supported and k.casefold() not in PYTHON_CHESS_MANAGED
    }
    if applied:
        engine.configure(applied)
    return applied


def play_pair(task):
    (
        piebot_cmd,
        opp_cmd,
        piebot_opts,
        opp_opts,
        piebot_supported,
        opp_supported,
        fen,
        initial_s,
        increment_s,
    ) = task
    results = []
    piebot = chess.engine.SimpleEngine.popen_uci(piebot_cmd)
    opp = chess.engine.SimpleEngine.popen_uci(opp_cmd)
    try:
        safe_configure(piebot, piebot_opts, piebot_supported)
        safe_configure(opp, opp_opts, opp_supported)
        for piebot_is_white in (True, False):
            board = chess.Board(fen)
            clocks = {chess.WHITE: initial_s, chess.BLACK: initial_s}
            plies = 0
            score = None
            termination = None
            while plies < MAX_PLIES:
                if board.is_game_over(claim_draw=True):
                    break
                turn = board.turn
                pie_to_move = (turn == chess.WHITE) == piebot_is_white
                engine = piebot if pie_to_move else opp
                if clocks[turn] <= 0:
                    score, termination = (0.0 if pie_to_move else 1.0), "time_forfeit"
                    break
                limit = chess.engine.Limit(
                    white_clock=clocks[chess.WHITE],
                    black_clock=clocks[chess.BLACK],
                    white_inc=increment_s,
                    black_inc=increment_s,
                )
                started = time.monotonic()
                try:
                    res = engine.play(board, limit)
                except chess.engine.EngineError:
                    score, termination = (0.0 if pie_to_move else 1.0), "engine_error"
                    break
                clocks[turn] = clocks[turn] - (time.monotonic() - started) + increment_s
                if clocks[turn] <= 0:
                    score, termination = (0.0 if pie_to_move else 1.0), "time_forfeit"
                    break
                if res.move is None:
                    break
                board.push(res.move)
                plies += 1
            if score is None:
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    score, termination = 0.5, "ply_cap"
                elif outcome.winner is None:
                    score, termination = 0.5, outcome.termination.name.lower()
                else:
                    won = (outcome.winner == chess.WHITE) == piebot_is_white
                    score = 1.0 if won else 0.0
                    termination = outcome.termination.name.lower()
            results.append(
                {
                    "fen": fen,
                    "piebot_is_white": piebot_is_white,
                    "piebot_score": score,
                    "termination": termination,
                    "plies": plies,
                }
            )
    finally:
        piebot.quit()
        opp.quit()
    return results


def elo_from_score(s: float) -> float | None:
    if s <= 0.0 or s >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--piebot-command", required=True)
    ap.add_argument("--piebot-nnue", type=Path, required=True)
    ap.add_argument("--piebot-blend", type=int, default=75)
    ap.add_argument("--piebot-threads", type=int, default=1)
    ap.add_argument("--piebot-hash", type=int, default=64)
    ap.add_argument("--opponent-command", required=True)
    ap.add_argument("--opponent-name", required=True)
    ap.add_argument("--opponent-ccrl-elo", type=int, required=True)
    ap.add_argument("--opponent-hash", type=int, default=64)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--time-control", default="60+0.5")
    ap.add_argument("--book", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.games % 2:
        raise SystemExit("--games must be even (openings are played in pairs)")
    initial_s, increment_s = (float(x) for x in args.time_control.split("+"))

    piebot_cmd = args.piebot_command.split()
    opp_cmd = args.opponent_command.split()

    nnue = args.piebot_nnue.expanduser().resolve(strict=True)
    # Insertion order matters: load the model before enabling NNUE.
    piebot_opts = {
        "Threads": args.piebot_threads,
        "Hash": args.piebot_hash,
        "NNUEQuantFile": str(nnue),
        "UseNNUE": True,
        "EvalBlend": args.piebot_blend,
    }
    opp_opts = {"Hash": args.opponent_hash, "Threads": 1}

    piebot_supported = advertised_options(piebot_cmd)
    opp_supported = advertised_options(opp_cmd)
    print(f"PieBot advertises: {sorted(piebot_supported)}")
    print(f"{args.opponent_name} advertises: {sorted(opp_supported)}")
    print(f"applying to opponent: {sorted(k for k in opp_opts if k.casefold() in opp_supported)}")

    openings = [
        ln.strip()
        for ln in Path(args.book).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    rng = random.Random(args.seed)
    pairs = args.games // 2
    tasks = [
        (
            piebot_cmd,
            opp_cmd,
            piebot_opts,
            opp_opts,
            piebot_supported,
            opp_supported,
            rng.choice(openings),
            initial_s,
            increment_s,
        )
        for _ in range(pairs)
    ]

    started = time.time()
    games: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(play_pair, t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), start=1):
            games.extend(fut.result())
            if done % 5 == 0:
                s = sum(g["piebot_score"] for g in games) / len(games)
                print(f"pairs {done}/{pairs}  games={len(games)}  piebot={s:.4f}", flush=True)

    n = len(games)
    pts = sum(g["piebot_score"] for g in games)
    score = pts / n
    wins = sum(1 for g in games if g["piebot_score"] == 1.0)
    draws = sum(1 for g in games if g["piebot_score"] == 0.5)
    losses = sum(1 for g in games if g["piebot_score"] == 0.0)

    mean_sq = sum(g["piebot_score"] ** 2 for g in games) / n
    stderr = math.sqrt(max(mean_sq - score * score, 0.0) / n)
    lo, hi = score - 1.96 * stderr, score + 1.96 * stderr

    def rating(s):
        d = elo_from_score(s)
        return None if d is None else args.opponent_ccrl_elo + d

    summary = {
        "opponent": args.opponent_name,
        "opponent_ccrl_elo": args.opponent_ccrl_elo,
        "games": n,
        "piebot_wins": wins,
        "draws": draws,
        "piebot_losses": losses,
        "piebot_score_rate": score,
        "score_95_ci": [lo, hi],
        "elo_difference": elo_from_score(score),
        "piebot_ccrl_estimate": rating(score),
        "piebot_ccrl_95_ci": [rating(lo), rating(hi)],
        "time_control": args.time_control,
        "piebot_threads": args.piebot_threads,
        "seed": args.seed,
        "wall_time_s": time.time() - started,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "games": games}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
