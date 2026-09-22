#!/usr/bin/env python3
"""Ask what PieBot's ladder draws are actually made of.

PieBot draws ~30% of its ladder games against opponents 300-500 Elo stronger,
and that rate barely moves when the opponent gets 190 Elo stronger. That fixed
block of draws sets a score floor and inflates every ladder-derived rating
(evidence/ladder_draw_floor_20260816.json). Before deciding what it means, it
is worth knowing what kind of position each draw came from.

Two very different stories fit the same score:

  RESCUE   PieBot reaches a lost position and escapes by repetition. Then the
           draws are real defensive value -- worth keeping -- but they are also
           exactly what a strength-limited opponent might hand over and a
           full-strength one would not.

  LEVEL    The positions are genuinely balanced when the repetition happens.
           Then PieBot is simply holding, and the draw rate is honest.

This replays each drawn game and evaluates the final position with
FULL-STRENGTH Stockfish, scored from PieBot's point of view. Losses are
sampled the same way at the same point as a control, so "the evaluation was
bad" can be compared against games PieBot actually lost.

Evaluation only. Nothing here produces training labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import chess
import chess.engine

MANAGED = {"ponder", "multipv"}


def full_strength_options(hash_mb: int) -> dict:
    return {"Threads": 1, "Hash": hash_mb, "SyzygyProbeLimit": 0, "UCI_LimitStrength": False}


def replay(game: dict, plies_before_end: int = 0) -> chess.Board | None:
    """Rebuild the position `plies_before_end` plies from the end.

    A lost game ends in checkmate, which is terminal and cannot be evaluated.
    Backing off a few plies gives draws and losses a COMMON, non-terminal
    sampling point, which is what makes the two groups comparable at all.
    """
    board = chess.Board(game["opening_fen"])
    moves = game.get("moves", [])
    if plies_before_end:
        moves = moves[: max(0, len(moves) - plies_before_end)]
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
    return board


def evaluate_cp(engine, board: chess.Board, depth: int, piebot_white: bool) -> int | None:
    """Score in centipawns from PieBot's point of view. Mates clamp to +-10000."""
    if board.is_game_over(claim_draw=False):
        return None
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white() if piebot_white else info["score"].black()
    return score.score(mate_score=10000)


def summarise(label: str, values: list[int]) -> dict:
    if not values:
        return {"label": label, "n": 0}
    values = sorted(values)
    return {
        "label": label,
        "n": len(values),
        "mean_cp": round(statistics.mean(values), 1),
        "median_cp": values[len(values) // 2],
        "p10_cp": values[max(0, len(values) // 10)],
        "p90_cp": values[min(len(values) - 1, 9 * len(values) // 10)],
        "share_losing_below_-200cp": round(
            sum(1 for v in values if v < -200) / len(values), 3
        ),
        "share_losing_below_-500cp": round(
            sum(1 for v in values if v < -500) / len(values), 3
        ),
        "share_balanced_within_100cp": round(
            sum(1 for v in values if abs(v) <= 100) / len(values), 3
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="rung_*.json files")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--max-losses", type=int, default=40, help="control sample size")
    ap.add_argument(
        "--plies-before-end",
        type=int,
        default=0,
        help=(
            "sample this many plies before the end. Use a non-zero value to "
            "compare draws against losses: lost games end in checkmate, a "
            "terminal position that cannot be evaluated, so 0 yields no losses."
        ),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure(
        {k: v for k, v in full_strength_options(args.hash_mb).items() if k.casefold() not in MANAGED}
    )

    draw_cps: list[int] = []
    loss_cps: list[int] = []
    unreplayable = 0
    per_game: list[dict] = []

    try:
        for path in args.results:
            payload = json.loads(Path(path).read_text())
            for game in payload.get("games", []):
                score = game.get("piebot_score")
                termination = game.get("termination", "")
                is_draw = score == 0.5
                is_loss = score == 0.0
                if not (is_draw or is_loss):
                    continue
                if is_loss and len(loss_cps) >= args.max_losses:
                    continue

                board = replay(game, args.plies_before_end)
                if board is None:
                    unreplayable += 1
                    continue
                piebot_white = game.get("piebot_color") == "white"
                cp = evaluate_cp(engine, board, args.depth, piebot_white)
                if cp is None:
                    continue
                (draw_cps if is_draw else loss_cps).append(cp)
                per_game.append(
                    {
                        "source": Path(path).name,
                        "result": "draw" if is_draw else "loss",
                        "termination": termination,
                        "final_cp_for_piebot": cp,
                        "plies": game.get("plies"),
                    }
                )
    finally:
        engine.quit()

    report = {
        "depth": args.depth,
        "plies_before_end": args.plies_before_end,
        "unreplayable_games": unreplayable,
        "draws": summarise("draws (final position)", draw_cps),
        "losses_control": summarise("losses (final position, control)", loss_cps),
    }
    Path(args.out).write_text(json.dumps({"summary": report, "games": per_game}, indent=2))
    print(json.dumps(report, indent=2))

    d, l = report["draws"], report["losses_control"]
    if d.get("n") and l.get("n"):
        print(
            f"\ndrawn games median {d['median_cp']} cp vs lost games median "
            f"{l['median_cp']} cp (PieBot's point of view)"
        )
        print(
            f"share of draws already losing by >200cp: "
            f"{d['share_losing_below_-200cp']:.1%}"
        )


if __name__ == "__main__":
    main()
