"""Paired NPS benchmark: h64 cycle-98 vs its function-identical h128 twin.

Interactive UCI (engine aborts search on pending stdin, so commands are fed
one at a time, waiting for bestmove).
Phase 1 (identity): fixed-depth search per position, compare nodes+bestmove.
Phase 2 (speed): movetime searches, interleaved A/B rounds to cancel drift.
"""
import subprocess, statistics, sys, time

ENGINE = "PieBot/target/release/uci"
H64 = "models/cycle_000098_quant.nnue"
H128 = sys.argv[1]

FENS = [
    "startpos",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1N1PPP/R2QKB1R w KQ - 0 8",
    "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN5/PPP2PPP/R2Q1RK1 w - - 0 12",
    "2r2rk1/1bqnbppp/p2pp3/1p6/3NPP2/2N1B3/PPPQ2PP/2KR3R w - - 0 14",
    "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RK2R w K - 0 17",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

class Uci:
    def __init__(self, net):
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        for opt in ("Threads value 1", "Hash value 256", "UseNNUE value true",
                    f"NNUEQuantFile value {net}", "EvalBlend value 100"):
            self.send(f"setoption name {opt.split(' value')[0]} value {opt.split('value ')[1]}")
        self.send("isready"); self.wait("readyok")

    def send(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            ln = self.p.stdout.readline()
            if not ln:
                raise RuntimeError(f"engine died waiting for {token}: {lines[-3:]}")
            ln = ln.strip(); lines.append(ln)
            if ln.startswith(token):
                return lines

    def search(self, fen, go):
        self.send("position startpos" if fen == "startpos" else f"position fen {fen}")
        self.send(go)
        lines = self.wait("bestmove")
        nodes = nps = None
        for ln in lines:
            if ln.startswith("info") and " nodes " in ln:
                t = ln.split()
                nodes = int(t[t.index("nodes") + 1])
                if "nps" in t:
                    nps = int(t[t.index("nps") + 1])
        return nodes, nps, lines[-1].split()[1]

    def quit(self):
        self.send("quit"); self.p.wait(timeout=10)

# Phase 1: identity at fixed depth 9
e64, e128 = Uci(H64), Uci(H128)
print("identity check (depth 9):")
same = 0
for f in FENS:
    na, _, ma = e64.search(f, "go depth 9")
    nb, _, mb = e128.search(f, "go depth 9")
    ok = na == nb and ma == mb
    same += ok
    print(f"  nodes {na:>10} vs {nb:>10}  best {ma:>6} vs {mb:>6}  {'OK' if ok else 'DIFF'}")
print(f"identical trees: {same}/{len(FENS)}")
e64.quit(); e128.quit()

# Phase 2: movetime 3000, fresh engine per round, interleaved
rounds = 2
h64_nps, h128_nps = [], []
for r in range(rounds):
    for net, sink in ((H64, h64_nps), (H128, h128_nps)):
        e = Uci(net)
        for f in FENS:
            t0 = time.perf_counter()
            nodes, _, _ = e.search(f, "go movetime 3000")
            sink.append(nodes / (time.perf_counter() - t0))
        e.quit()
m64 = statistics.median(h64_nps)
m128 = statistics.median(h128_nps)
print(f"\nh64  nps: median {m64:>9.0f}  mean {statistics.mean(h64_nps):>9.0f}")
print(f"h128 nps: median {m128:>9.0f}  mean {statistics.mean(h128_nps):>9.0f}")
print(f"h128/h64 median ratio: {m128/m64:.3f}  (slowdown {100*(1-m128/m64):.1f}%)")
