#!/usr/bin/env python3
"""Run a queue of A/B matches one after another (14 workers each).

The queue is a JSON list; each entry is
  {"name": ..., "a": cmd, "b": cmd, "a_opts": {...}, "b_opts": {...},
   "games": 400, "tc": "10+0.1", "seed": 1}
Entries whose result file already exists are skipped, so the queue can be
appended to while it runs (it is re-read after every match) and resumed
after an interruption. Results go to out/abq/<name>.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    queue_path = Path(sys.argv[1])
    out_dir = ROOT / "out" / "abq"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    while True:
        queue = json.loads(queue_path.read_text())
        todo = [e for e in queue if not (out_dir / f"{e['name']}.json").exists()
                or json.loads((out_dir / f"{e['name']}.json").read_text())["summary"]["games"] < e.get("games", 400)]
        if not todo:
            print("queue empty", flush=True)
            return
        e = todo[0]
        cmd = [sys.executable, str(ROOT / "scripts/experiments/uci_match.py"), "--a", e["a"], "--b", e["b"],
               "--games", str(e.get("games", 400)), "--tc", e.get("tc", "10+0.1"), "--workers", str(workers),
               "--book", str(ROOT / "books/openings_v1.fen"), "--seed", str(e.get("seed", 1)),
               "--out", str(out_dir / f"{e['name']}.json")]
        for k, v in e.get("a_opts", {}).items():
            cmd += ["--a-opt", f"{k}={v}"]
        for k, v in e.get("b_opts", {}).items():
            cmd += ["--b-opt", f"{k}={v}"]
        t = time.time()
        with open(out_dir / f"{e['name']}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        s = json.loads((out_dir / f"{e['name']}.json").read_text())["summary"]
        print(f"{e['name']}: games {s['games']} W{s['a_wins']} D{s['draws']} L{s['a_losses']} "
              f"elo {s['elo']:+.1f} [{s['elo_95_ci'][0]:+.1f}, {s['elo_95_ci'][1]:+.1f}] "
              f"({(time.time()-t)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
