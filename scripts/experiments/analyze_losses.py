#!/usr/bin/env python3
"""Detailed engine analysis of PieBot losses against Stockfish 3000."""
from __future__ import annotations

import json
from pathlib import Path
import chess
import chess.engine

ARENA_RESULTS = Path("evidence/stockfish_30_arena_m4pro_8t.json")
OUTPUT_REPORT = Path("evidence/losses_analysis_stockfish3000.json")
STOCKFISH_BIN = "/opt/homebrew/bin/stockfish"


def score_to_cp(score: chess.engine.Score, pov_color: chess.Color) -> int:
    pov_score = score.pov(pov_color)
    if pov_score.is_mate():
        mate = pov_score.mate()
        if mate > 0:
            return 30000 - mate * 100
        else:
            return -30000 - mate * 100
    return pov_score.score(mate_score=30000)


def analyze_game(engine: chess.engine.SimpleEngine, game_record: dict) -> dict:
    game_idx = game_record["game_index"] + 1
    piebot_color = chess.WHITE if game_record["piebot_color"] == "white" else chess.BLACK
    fen = game_record.get("opening_fen", chess.STARTING_FEN)
    board = chess.Board(fen)
    moves = game_record.get("moves", [])

    blunders = []
    mistakes = []
    turning_point = None
    prior_score = 0
    decisive_seen = False

    # First analyze initial board
    init_info = engine.analyse(board, chess.engine.Limit(depth=12))
    current_eval = score_to_cp(init_info["score"], piebot_color)

    for ply_idx, uci_move in enumerate(moves):
        is_piebot_turn = (board.turn == piebot_color)
        move = chess.Move.from_uci(uci_move)
        san_move = board.san(move)

        if is_piebot_turn:
            # Analyze position before move
            pre_info = engine.analyse(board, chess.engine.Limit(depth=12))
            best_move = pre_info.get("pv", [None])[0]
            best_san = board.san(best_move) if best_move else ""
            eval_before = score_to_cp(pre_info["score"], piebot_color)

            # Push PieBot's move
            board.push(move)

            # Analyze position after move
            post_info = engine.analyse(board, chess.engine.Limit(depth=12))
            eval_after = score_to_cp(post_info["score"], piebot_color)

            loss = eval_before - eval_after
            move_num = (board.fullmove_number - 1) if board.turn == chess.WHITE else board.fullmove_number

            entry = {
                "ply": ply_idx + 1,
                "move_number": move_num,
                "fen_before": board.board_fen(),
                "piebot_move": san_move,
                "piebot_uci": uci_move,
                "best_move": best_san,
                "best_uci": best_move.uci() if best_move else "",
                "eval_before_cp": eval_before,
                "eval_after_cp": eval_after,
                "cp_drop": loss,
            }

            if loss >= 150:
                blunders.append(entry)
                if not decisive_seen and eval_after <= -200:
                    turning_point = entry
                    decisive_seen = True
            elif loss >= 80:
                mistakes.append(entry)
                if not decisive_seen and eval_after <= -200:
                    turning_point = entry
                    decisive_seen = True
        else:
            board.push(move)

    return {
        "game_number": game_idx,
        "piebot_color": game_record["piebot_color"],
        "plies": len(moves),
        "termination": game_record.get("termination"),
        "opening_fen": fen,
        "blunders_count": len(blunders),
        "mistakes_count": len(mistakes),
        "turning_point": turning_point,
        "top_blunders": sorted(blunders, key=lambda x: x["cp_drop"], reverse=True)[:3],
        "top_mistakes": sorted(mistakes, key=lambda x: x["cp_drop"], reverse=True)[:3],
    }


def main():
    if not ARENA_RESULTS.exists():
        print(f"File {ARENA_RESULTS} does not exist.")
        return

    with open(ARENA_RESULTS) as f:
        data = json.load(f)

    losses = [g for g in data.get("games", []) if g.get("piebot_score") == 0.0]
    print(f"Found {len(losses)} lost games to analyze.")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    engine.configure({"Threads": 4, "Hash": 256})

    reports = []
    for g in losses:
        gid = g["game_index"] + 1
        print(f"Analyzing Game {gid} ({g['piebot_color']})...")
        rep = analyze_game(engine, g)
        reports.append(rep)
        tp = rep.get("turning_point")
        if tp:
            print(f"  Game {gid} Turning Point: Move {tp['move_number']} ({tp['piebot_move']}), best was {tp['best_move']}, drop: {tp['cp_drop']}cp (Eval: {tp['eval_before_cp']} -> {tp['eval_after_cp']})")
        else:
            print(f"  Game {gid}: Gradual positional squeeze (no single blunder >=80cp)")

    engine.quit()

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nWrote detailed analysis to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
