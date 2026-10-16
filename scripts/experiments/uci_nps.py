#!/usr/bin/env python3
"""Wall-clock time for each engine to search a fixed node count on a few
positions, alternating engines to share whatever load the machine has."""
import argparse
import time

import chess
import chess.engine

FENS = [
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
    "r2q1rk1/1b2bppp/p2p1n2/1p2p3/3PP3/1BN2N1P/PP3PP1/R2Q1RK1 w - - 0 14",
    "8/5pk1/6p1/3R4/7P/6P1/r4PK1/8 w - - 0 40",
    "8/8/4k3/3p4/3P1K2/8/5P2/8 w - - 0 50",
    "r4rk1/pp3ppp/2n1b3/q1pp4/3P4/P1P1PN2/2Q1BPPP/R4RK1 w - - 0 14",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--nodes", type=int, default=2_000_000)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    opts = {}
    for o in a.opt:
        k, _, v = o.partition("=")
        opts[k] = int(v) if v.lstrip("-").isdigit() else v
    engines = [chess.engine.SimpleEngine.popen_uci(e) for e in a.engines]
    for e in engines:
        e.configure(opts)
    totals = [[0.0] * len(FENS) for _ in engines]
    for _ in range(a.rounds):
        for fi, fen in enumerate(FENS):
            for ei, e in enumerate(engines):
                e.protocol.send_line("ucinewgame")
                b = chess.Board(fen)
                t = time.monotonic()
                e.play(b, chess.engine.Limit(nodes=a.nodes))
                totals[ei][fi] += time.monotonic() - t
    for name, row in zip(a.engines, totals):
        knps = [a.nodes * a.rounds / s / 1000 for s in row]
        print(f"{name}: knps per position {[round(k) for k in knps]}  mean {sum(knps)/len(knps):.0f}")
    for e in engines:
        e.quit()


if __name__ == "__main__":
    main()
