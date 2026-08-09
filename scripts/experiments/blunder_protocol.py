"""Pure-network blunder protocol: measure a net's move quality in isolation.

Plays N games with the network as the ONLY evaluation (blend 100, greedy,
book openings), then judges every move with the traditional PST engine at a
fixed depth. Move quality is consecutive-eval centipawn loss:

    loss_i = max(0, V_i + V_{i+1})     (both evals side-to-move relative,
                                        so a sign flip means the mover lost
                                        ground), clamped to +/-1500.

This is the campaign's external instrument for "is the learner actually
getting better", independent of the promotion gate's blend dilution.
Established 2026-08-07; committed 2026-08-08 after the original ephemeral
copy was lost with a session scratchpad.

Reference results (300 games, depth 3, seed 20260821, PST depth-5 judge):
  cycle-98 v1 incumbent : ACPL 34.0, 1.77 blunders/game, 81 zero-blunder games
  campaign_v4 cycle-22  : ACPL 36.6, 2.15 blunders/game, 64 zero-blunder games

Usage:
  python3 scripts/experiments/blunder_protocol.py --net models/x.nnue \
      --label "cycle-98" --games 300 --out /tmp/blunder_x
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELFPLAY = REPO / "PieBot/target/release/selfplay"
UCI = REPO / "PieBot/target/release/uci"
BOOK = REPO / "books/openings_v1.fen"
CLAMP = 1500


def generate(net: Path, out_dir: Path, games: int, depth: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = out_dir / "games"
    cmd = [
        str(SELFPLAY), "--games", str(games), "--depth", str(depth),
        "--threads", "1", "--parallel-games", "8", "--seed", str(seed),
        "--jsonl-out", str(jsonl_dir), "--skip-bin", "--use-engine",
        "--nnue-quant-file", str(net), "--nnue-blend-percent", "100",
        "--openings", str(BOOK),
        # Greedy: no exploration noise, so we measure the net's own choices.
        "--temperature-moves", "0", "--dirichlet-epsilon", "0",
        "--max-plies", "200",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return jsonl_dir


class Judge:
    """Traditional PST engine at fixed depth, stm-relative scores."""

    def __init__(self, depth: int):
        self.depth = depth
        self.p = subprocess.Popen([str(UCI)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("uci"); self._wait("uciok")
        for opt in (("Threads", "1"), ("Hash", "128"), ("UseNNUE", "false")):
            self._send(f"setoption name {opt[0]} value {opt[1]}")
        self._send("isready"); self._wait("readyok")

    def _send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, tok):
        lines = []
        while True:
            ln = self.p.stdout.readline()
            if not ln:
                raise RuntimeError(f"judge died waiting for {tok}")
            ln = ln.strip(); lines.append(ln)
            if ln.startswith(tok):
                return lines

    def score(self, fen: str) -> int:
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        cp = 0
        for ln in self._wait("bestmove"):
            if ln.startswith("info") and " score cp " in ln:
                t = ln.split()
                cp = int(t[t.index("cp") + 1])
            elif ln.startswith("info") and " score mate " in ln:
                t = ln.split()
                m = int(t[t.index("mate") + 1])
                cp = CLAMP if m > 0 else -CLAMP
        return max(-CLAMP, min(CLAMP, cp))

    def close(self):
        self._send("quit"); self.p.wait(timeout=10)


def analyze(jsonl_dir: Path, judge_depth: int, label: str) -> dict:
    # Records are one row per position; group into games and order by ply.
    by_game: dict = {}
    for shard in sorted(Path(jsonl_dir).glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_game.setdefault(row["game_id"], []).append(row)
    for rows in by_game.values():
        rows.sort(key=lambda r: r.get("ply", 0))

    judge = Judge(judge_depth)
    losses, per_game = [], []
    for gi, rows in enumerate(by_game.values()):
        if len(rows) < 2:
            continue
        game_losses = []
        prev = judge.score(rows[0]["fen"])
        for row in rows[1:]:
            cur = judge.score(row["fen"])
            # Both scores are side-to-move relative, so the player who just
            # moved lost ground iff prev + cur > 0.
            game_losses.append(max(0, min(CLAMP, prev + cur)))
            prev = cur
        losses.extend(game_losses)
        per_game.append(game_losses)
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(by_game)} games analyzed", flush=True)
    judge.close()

    n = max(1, len(losses))
    blunders = [l for l in losses if l >= 300]
    mistakes = [l for l in losses if 100 <= l < 300]
    inacc = [l for l in losses if 50 <= l < 100]
    zero_blunder = sum(1 for gl in per_game if not any(l >= 300 for l in gl))
    srt = sorted(losses)
    return {
        "model": label,
        "games": len(per_game),
        "moves_scored": len(losses),
        "avg_centipawn_loss": round(sum(losses) / n, 1),
        "median_cp_loss": srt[n // 2] if srt else 0,
        "blunders_300cp": {"count": len(blunders),
                           "per_game": round(len(blunders) / max(1, len(per_game)), 2)},
        "mistakes_100_299cp": {"count": len(mistakes),
                               "per_game": round(len(mistakes) / max(1, len(per_game)), 2)},
        "inaccuracies_50_99cp": {"count": len(inacc),
                                 "per_game": round(len(inacc) / max(1, len(per_game)), 2)},
        "games_with_zero_blunders": zero_blunder,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--judge-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"generating {args.games} games at depth {args.depth} (blend 100, greedy)...",
          flush=True)
    jsonl_dir = generate(args.net, args.out, args.games, args.depth, args.seed)
    print("analyzing...", flush=True)
    report = analyze(jsonl_dir, args.judge_depth, args.label)
    report["protocol"] = {
        "play": f"depth {args.depth}, NNUE-only (blend 100), greedy, book openings",
        "analyst": f"traditional PST engine, depth {args.judge_depth}, evals clamped +/-{CLAMP}",
        "seed": args.seed,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
