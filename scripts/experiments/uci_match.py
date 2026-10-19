#!/usr/bin/env python3
"""Parallel paired match between two arbitrary UCI engines.

Each opening is played twice with colours reversed, one engine thread per
side unless told otherwise, on a real Fischer clock. A game is drawn only on
a threefold that actually occurred, the fifty-move rule, stalemate or
insufficient material -- never because a repetition was merely claimable (see
documents/history/agent_handoffs_2026.md, "arena threefold draws before this
date never happened").

Results are written after every finished pair, so a run can be watched while
it plays. Evaluation only; nothing here produces training labels.

Example::

    python3 scripts/experiments/uci_match.py \
        --a "out/bin_new/uci" --a-opt EvalBlend=100 \
        --b "out/bin_old/uci" --b-opt EvalBlend=100 \
        --games 400 --tc 10+0.1 --workers 14 --book books/openings_v1.fen \
        --out out/match.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine

PYTHON_CHESS_MANAGED = {"ponder", "multipv", "uci_chess960"}


def parse_opts(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
    return out


def game_over(board: chess.Board) -> chess.Outcome | None:
    """End the game only on a position that has really occurred three times."""
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return outcome
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    if board.is_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return None


def elo_from_score(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def pentanomial_stats(pair_scores: list[float]) -> dict:
    """Score, Elo and a 95% CI from per-pair points (0, 0.5, ..., 2)."""
    n = len(pair_scores)
    if n == 0:
        return {"pairs": 0}
    per = [p / 2.0 for p in pair_scores]
    mean = sum(per) / n
    var = sum((x - mean) ** 2 for x in per) / max(n - 1, 1)
    se = math.sqrt(var / n)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    counts = [0] * 5
    for p in pair_scores:
        counts[int(round(p * 2))] += 1
    return {
        "pairs": n,
        "score": mean,
        "elo": elo_from_score(mean),
        "elo_95_ci": [elo_from_score(lo), elo_from_score(hi)],
        "pentanomial": counts,
    }


def play_pair(task: dict) -> dict:
    a = chess.engine.SimpleEngine.popen_uci(task["a_cmd"])
    b = chess.engine.SimpleEngine.popen_uci(task["b_cmd"])
    games = []
    try:
        for eng, opts in ((a, task["a_opts"]), (b, task["b_opts"])):
            usable = {k: v for k, v in opts.items() if k.casefold() not in PYTHON_CHESS_MANAGED}
            if usable:
                eng.configure(usable)
        for a_is_white in (True, False):
            for eng in (a, b):
                # A fresh game: clear hash and histories as a GUI would.
                eng.protocol.send_line("ucinewgame")
            board = chess.Board(task["fen"])
            clocks = {chess.WHITE: task["init"], chess.BLACK: task["init"]}
            inc = task["inc"]
            result = None
            termination = None
            times = []
            while board.ply() < task["max_plies"]:
                outcome = game_over(board)
                if outcome is not None:
                    termination = outcome.termination.name.lower()
                    if outcome.winner is None:
                        result = 0.5
                    else:
                        result = 1.0 if (outcome.winner == chess.WHITE) == a_is_white else 0.0
                    break
                turn = board.turn
                a_to_move = (turn == chess.WHITE) == a_is_white
                eng = a if a_to_move else b
                limit = chess.engine.Limit(
                    white_clock=max(clocks[chess.WHITE], 0.001),
                    black_clock=max(clocks[chess.BLACK], 0.001),
                    white_inc=inc,
                    black_inc=inc,
                )
                started = time.monotonic()
                try:
                    res = eng.play(board, limit, game=task["game_id"] * 2 + int(a_is_white))
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                    result, termination = (0.0 if a_to_move else 1.0), "engine_error"
                    break
                spent = time.monotonic() - started
                times.append(round(spent, 3))
                clocks[turn] -= spent
                if clocks[turn] < -task["grace"]:
                    result, termination = (0.0 if a_to_move else 1.0), "time_forfeit"
                    break
                clocks[turn] += inc
                if res.move is None or res.move not in board.legal_moves:
                    result, termination = (0.0 if a_to_move else 1.0), "illegal_move"
                    break
                board.push(res.move)
            if result is None:
                result, termination = 0.5, "ply_cap"
            games.append(
                {
                    "a_is_white": a_is_white,
                    "a_score": result,
                    "termination": termination,
                    "plies": board.ply(),
                    "moves": " ".join(m.uci() for m in board.move_stack),
                    "move_times": times,
                }
            )
    finally:
        for eng in (a, b):
            try:
                eng.quit()
            except Exception:
                pass
    return {"fen": task["fen"], "pair": task["game_id"], "games": games}


def summarize(pairs: list[dict]) -> dict:
    games = [g for p in pairs for g in p["games"]]
    w = sum(1 for g in games if g["a_score"] == 1.0)
    d = sum(1 for g in games if g["a_score"] == 0.5)
    l = sum(1 for g in games if g["a_score"] == 0.0)
    terms: dict = {}
    for g in games:
        terms[g["termination"]] = terms.get(g["termination"], 0) + 1
    stats = pentanomial_stats([sum(g["a_score"] for g in p["games"]) for p in pairs])
    return {"games": len(games), "a_wins": w, "draws": d, "a_losses": l,
            "a_win_or_draw_rate": (w + d) / len(games) if games else None,
            "terminations": terms, **stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="engine A command")
    ap.add_argument("--b", required=True, help="engine B command")
    ap.add_argument("--a-opt", action="append", default=[], help="UCI option NAME=VALUE for A")
    ap.add_argument("--b-opt", action="append", default=[], help="UCI option NAME=VALUE for B")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--tc", default="10+0.1", help="INITIAL+INC seconds")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--book", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--grace", type=float, default=0.25, help="seconds past zero before a loss on time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="keep finished pairs already in --out")
    args = ap.parse_args()
    if args.games % 2:
        raise SystemExit("--games must be even")
    init, inc = (float(x) for x in args.tc.split("+"))
    openings = [ln.strip() for ln in Path(args.book).read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    rng = random.Random(args.seed)
    chosen = rng.sample(openings, k=min(args.games // 2, len(openings)))
    while len(chosen) < args.games // 2:
        chosen.append(rng.choice(openings))
    tasks = [
        {"a_cmd": shlex.split(args.a), "b_cmd": shlex.split(args.b),
         "a_opts": parse_opts(args.a_opt), "b_opts": parse_opts(args.b_opt),
         "fen": fen, "init": init, "inc": inc, "game_id": i,
         "max_plies": args.max_plies, "grace": args.grace}
        for i, fen in enumerate(chosen)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    started = time.time()
    pairs: list[dict] = []
    if args.resume and out.exists():
        # Keep finished pairs from an earlier run with the same openings and
        # settings (worker count may differ); replay only the rest.
        prev = json.loads(out.read_text())
        keys = ("a", "b", "a_opt", "b_opt", "games", "tc", "book", "seed", "max_plies")
        if any(prev["config"].get(k) != config[k] for k in keys):
            raise SystemExit("--resume: existing result file has different settings")
        pairs = prev["pairs"]
        done = {p["pair"] for p in pairs}
        tasks = [t for t in tasks if t["game_id"] not in done]
        print(f"resuming: {len(pairs)} pairs kept, {len(tasks)} to play", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(play_pair, t) for t in tasks]
        for fut in as_completed(futures):
            pairs.append(fut.result())
            summary = summarize(pairs)
            summary["wall_s"] = time.time() - started
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"config": config, "summary": summary, "pairs": pairs}))
            tmp.replace(out)
            if len(pairs) % 10 == 0:
                print(f"pairs {len(pairs)}/{len(tasks)} W{summary['a_wins']} D{summary['draws']} "
                      f"L{summary['a_losses']} elo {summary['elo']:+.1f} "
                      f"[{summary['elo_95_ci'][0]:+.1f}, {summary['elo_95_ci'][1]:+.1f}]", flush=True)
    print(json.dumps(summarize(pairs), indent=2))


if __name__ == "__main__":
    main()
