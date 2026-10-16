#!/usr/bin/env python3
"""Asynchronous SPSA tuning of PieBot's experimental search constants.

Each iteration perturbs every tunable by +-c_k, plays one opening twice
(colours reversed) between theta+ and theta-, and moves theta along the
perturbation by the pair's score difference. This is the fishtest
formulation: c_k = c_end * (N/k)^0.101, a_k = a / (A + k)^0.602 with
a = r_end * c_end^2 * (A + N)^0.602 and A = N / 10.

The engine must be the experimental UCI build started with PIEBOT_TUNE=1,
which advertises every tunable as a spin option. State is checkpointed after
every iteration and a run resumes from --state.

Example::

    PIEBOT_TUNE=1 python3 scripts/experiments/spsa.py \
        --engine out/spsa/uci --opt NNUEQuantFile=$PWD/models/x.nnue \
        --iterations 30000 --tc 6+0.06 --workers 14 --state out/spsa/state.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uci_match import game_over, parse_opts  # noqa: E402


def read_tunables(engine_cmd: str) -> dict:
    eng = chess.engine.SimpleEngine.popen_uci(engine_cmd.split())
    try:
        out = {}
        for name, opt in eng.options.items():
            if opt.type == "spin" and name not in ("Threads", "Hash", "EvalBlend"):
                out[name] = {"default": opt.default, "min": opt.min, "max": opt.max}
        return out
    finally:
        eng.quit()


def c_end_for(spec: dict) -> float:
    span = spec["max"] - spec["min"]
    return max(0.5, abs(spec["default"]) * 0.1, span / 40.0)


def play_game(white, black, fen, init, inc, max_plies=300) -> float:
    """Return the score for White."""
    board = chess.Board(fen)
    clocks = {chess.WHITE: init, chess.BLACK: init}
    while board.ply() < max_plies:
        outcome = game_over(board)
        if outcome is not None:
            return 0.5 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else 0.0)
        eng = white if board.turn == chess.WHITE else black
        t = time.monotonic()
        try:
            res = eng.play(board, chess.engine.Limit(
                white_clock=max(clocks[chess.WHITE], 0.001), black_clock=max(clocks[chess.BLACK], 0.001),
                white_inc=inc, black_inc=inc))
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            return 0.0 if board.turn == chess.WHITE else 1.0
        clocks[board.turn] -= time.monotonic() - t
        if clocks[board.turn] < -0.25:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clocks[board.turn] += inc
        if res.move is None or res.move not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(res.move)
    return 0.5


class Spsa:
    def __init__(self, args, tunables: dict):
        self.args = args
        self.lock = threading.Lock()
        self.state_path = Path(args.state)
        self.N = args.iterations
        self.A = self.N / 10.0
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            self.params = st["params"]
            self.k = st["k"]
            self.history = st.get("history", [])
            self.wins = st.get("pair_scores", [])
        else:
            names = sorted(tunables) if not args.only else [n for n in args.only.split(",")]
            self.params = {}
            for n in names:
                spec = tunables[n]
                self.params[n] = {
                    "theta": float(spec["default"]),
                    "min": spec["min"],
                    "max": spec["max"],
                    "c_end": c_end_for(spec),
                    "r_end": args.r_end,
                    "start": spec["default"],
                }
            self.k = 0
            self.history = []
            self.wins = []

    def coeffs(self, p: dict, k: int):
        c_k = p["c_end"] * (self.N / max(k, 1)) ** 0.101
        a_end = p["r_end"] * p["c_end"] ** 2
        a = a_end * (self.A + self.N) ** 0.602
        a_k = a / (self.A + k) ** 0.602
        return c_k, a_k

    def draw(self):
        with self.lock:
            if self.k >= self.N:
                return None
            self.k += 1
            k = self.k
            plus, minus, delta = {}, {}, {}
            for n, p in self.params.items():
                c_k, _ = self.coeffs(p, k)
                d = random.choice((-1, 1))
                delta[n] = d
                plus[n] = self.round(p["theta"] + c_k * d, p)
                minus[n] = self.round(p["theta"] - c_k * d, p)
            return k, plus, minus, delta

    @staticmethod
    def round(x, p):
        # Stochastic rounding keeps small-c parameters moving on average.
        lo, hi = p["min"], p["max"]
        x = min(max(x, lo), hi)
        f = math.floor(x)
        return int(f + (1 if random.random() < x - f else 0))

    def update(self, k, delta, result):
        """result: theta+ points minus theta- points over the pair (-2..2)."""
        with self.lock:
            for n, p in self.params.items():
                c_k, a_k = self.coeffs(p, k)
                p["theta"] += a_k * result * delta[n] / c_k
                p["theta"] = min(max(p["theta"], p["min"]), p["max"])
            self.wins.append(result)
            if self.k % 50 == 0 or self.k >= self.N:
                self.history.append({"k": self.k, "theta": {n: round(p["theta"], 2) for n, p in self.params.items()}})
            self.save()

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"k": self.k, "params": self.params, "history": self.history[-400:],
                                   "pair_scores": self.wins[-20000:]}, indent=1))
        tmp.replace(self.state_path)


def worker(spsa: Spsa, args, openings, base_opts, wid):
    cmd = args.engine.split()
    a = chess.engine.SimpleEngine.popen_uci(cmd)
    b = chess.engine.SimpleEngine.popen_uci(cmd)
    for e in (a, b):
        e.configure(base_opts)
    init, inc = (float(x) for x in args.tc.split("+"))
    rng = random.Random(args.seed * 1000 + wid)
    try:
        while True:
            job = spsa.draw()
            if job is None:
                return
            k, plus, minus, delta = job
            a.configure(plus)
            b.configure(minus)
            fen = rng.choice(openings)
            for e in (a, b):
                e.protocol.send_line("ucinewgame")
            s1 = play_game(a, b, fen, init, inc)          # a (theta+) is White
            for e in (a, b):
                e.protocol.send_line("ucinewgame")
            s2 = 1.0 - play_game(b, a, fen, init, inc)    # a is Black
            plus_points = s1 + s2
            spsa.update(k, delta, plus_points - (2.0 - plus_points))
    finally:
        for e in (a, b):
            try:
                e.quit()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--tc", default="6+0.06")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--book", default="books/openings_v1.fen")
    ap.add_argument("--state", required=True)
    ap.add_argument("--r-end", type=float, default=0.002)
    ap.add_argument("--only", default="", help="comma-separated subset of tunables")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    tunables = read_tunables(args.engine)
    if not tunables:
        raise SystemExit("engine advertises no tunables; start it with PIEBOT_TUNE=1")
    spsa = Spsa(args, tunables)
    spsa.save()
    openings = [l.strip() for l in Path(args.book).read_text().splitlines() if l.strip() and not l.startswith("#")]
    base_opts = parse_opts(args.opt)
    threads = [threading.Thread(target=worker, args=(spsa, args, openings, base_opts, i), daemon=True)
               for i in range(args.workers)]
    for t in threads:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - last > 300:
            last = time.time()
            with spsa.lock:
                w = spsa.wins[-500:]
                print(f"k={spsa.k} mean pair diff (last {len(w)}) {sum(w)/max(len(w),1):+.3f}", flush=True)
    print("done", spsa.k)


if __name__ == "__main__":
    main()
