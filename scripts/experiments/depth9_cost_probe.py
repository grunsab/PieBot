"""Measure per-position node cost of a depth-9 teacher search (cycle-98, blend 25),
to size RELABEL_MAX_NODES for the v5 deep-teacher design. Sample of book positions."""
import subprocess, statistics, random, json

ENGINE = "PieBot/target/release/uci"
NET = "models/cycle_000098_quant.nnue"
random.seed(20260807)
fens = [l.strip() for l in open("books/openings_v1.fen") if l.strip()]
sample = random.sample(fens, 150)

p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
def wait(tok):
    lines = []
    while True:
        ln = p.stdout.readline().strip()
        lines.append(ln)
        if ln.startswith(tok): return lines

send("uci"); wait("uciok")
for o in ["Threads value 1", "Hash value 1024", "UseNNUE value true",
          f"NNUEQuantFile value {NET}", "EvalBlend value 25"]:
    send(f"setoption name {o.rsplit(' value ', 1)[0]} value {o.rsplit(' value ', 1)[1]}")
send("isready"); wait("readyok")

nodes_all = []
for i, fen in enumerate(sample):
    send(f"position fen {fen}")
    send("go depth 9")
    lines = wait("bestmove")
    nodes = None
    for ln in lines:
        if ln.startswith("info") and " nodes " in ln:
            t = ln.split(); nodes = int(t[t.index("nodes") + 1])
    nodes_all.append(nodes)
    if (i + 1) % 25 == 0:
        print(f"{i+1}/150 median so far {statistics.median(nodes_all):.0f}", flush=True)
send("quit")

nodes_all.sort()
out = {
    "positions": len(nodes_all),
    "depth": 9,
    "blend": 25,
    "median_nodes": statistics.median(nodes_all),
    "mean_nodes": statistics.mean(nodes_all),
    "p90_nodes": nodes_all[int(0.90 * len(nodes_all))],
    "p95_nodes": nodes_all[int(0.95 * len(nodes_all))],
    "max_nodes": nodes_all[-1],
}
print(json.dumps(out, indent=2))
