#!/usr/bin/env python3
"""Where does PieBot lose? For each lost game in a uci_match.py result file,
evaluate the position after every PieBot move with Stockfish at fixed depth
and report the first move where PieBot's side drops below a threshold, with
the swing that move caused, the ply and the material on the board.
Evaluation only."""

import argparse
import json
import statistics as st

import chess
import chess.engine

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(b):
    return sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--sf", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threshold", type=int, default=-150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.result))
    sf = chess.engine.SimpleEngine.popen_uci(a.sf)
    sf.configure({"Threads": 1, "Hash": 64})
    rows = []
    for p in d["pairs"]:
        for g in p["games"]:
            if g["a_score"] != 0.0:
                continue
            b = chess.Board(p["fen"])
            pie_white = g["a_is_white"]
            prev = None
            found = None
            for ply, u in enumerate(g["moves"].split()):
                mover_is_pie = (b.turn == chess.WHITE) == pie_white
                b.push_uci(u)
                if not mover_is_pie:
                    continue
                info = sf.analyse(b, chess.engine.Limit(depth=a.depth))
                sc = info["score"].pov(chess.WHITE if pie_white else chess.BLACK).score(mate_score=10000)
                if prev is not None and sc < a.threshold and prev >= a.threshold:
                    found = {"ply": ply, "before": prev, "after": sc, "swing": prev - sc,
                             "material": material(b), "fen_after": b.fen()}
                    break
                prev = sc
            if found:
                rows.append(found)
    sf.quit()
    json.dump(rows, open(a.out, "w"), indent=1)
    if rows:
        print("lost games with a crossing:", len(rows))
        print("median ply", st.median(r["ply"] for r in rows), "median material", st.median(r["material"] for r in rows))
        print("median swing", st.median(r["swing"] for r in rows),
              "share swing>=200:", sum(r["swing"] >= 200 for r in rows) / len(rows))
        for lo, hi in [(0, 40), (40, 80), (80, 140), (140, 999)]:
            print(f"ply {lo}-{hi}:", sum(lo <= r["ply"] < hi for r in rows))


if __name__ == "__main__":
    main()
