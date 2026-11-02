#!/usr/bin/env python3
"""Watch the w2048 bullet run: check, export and gate its checkpoints.

For every checkpoint `OUT/<net-id>-<n>/` bullet finishes, it:

1. checks the float weights in `quantised.bin` (the right size, all finite,
   no magnitude a trained net cannot have). This box has no error-correcting
   memory, and a flipped exponent bit shows up here as an absurd value;
2. exports the checkpoint to `NETS/<net-id>-<n>.nnue` and records the hashes
   and the exporter's worst-case accumulator in `NETS/manifest.jsonl`;
3. every 200 superbatches, plays the net against the incumbent with
   `uci_match.py` on the P-cores (200 games, 10+0.1), holding
   `/workspace/pcores.lock` so it never overlaps another P-core match.

A failed check writes `/workspace/w2048_ALERT` and is never exported.
The defaults are the plain w2048 run; `--threats` (with its own `--nets`,
`--gates`, `--alert`, `--engine` and `--incumbent`) watches a threat-input
run, whose checkpoints carry the threat rows after the hidden stack.
`latest-good` prints the newest complete checkpoint that passes the check,
which `deploy/vast/run_w2048.sh` resumes from.

Evaluation only; nothing here produces training labels.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUTPUT_BUCKETS = 8
MAX_ABS = 64.0  # trained weights are clipped near ±2; anything this large is corruption
GATE_EVERY = 200
SHAPE = dict(buckets=16, hidden=2048, l1=16, l2=32)

REPO = Path(__file__).resolve().parents[2]
BULLET = REPO / "training/bullet/target/release/piebot-bullet"
ENGINE = Path("/workspace/bin/uci_w2048")
INCUMBENT = REPO / "models/v3pw_bt4_1500.nnue"
NETS = Path("/workspace/w2048_nets")
GATES = Path("/workspace/w2048_gates")
ALERT = Path("/workspace/w2048_ALERT")
LOCK = Path("/workspace/pcores.lock")
CPU_PAIRS = "0:2,4:6,8:10,12:14"
N_THREAT_FEATURES = 59808  # training/nnue/threats.py


@dataclasses.dataclass
class RunConfig:
    """Where a run's exports and gates go, and what its gates play against.
    The defaults are the plain w2048 run; a threats run (file version 6) sets
    `threats` and its own directories and incumbent."""
    threats: bool = False
    engine: Path = ENGINE
    incumbent: Path = INCUMBENT
    incumbent_label: str = "pw1500"
    nets: Path = NETS
    gates: Path = GATES
    alert: Path = ALERT

    def gate_result(self, net_id: str, n: int) -> Path:
        return self.gates / f"{net_id}-{n}_vs_{self.incumbent_label}_200g.json"


def expected_floats(buckets: int, hidden: int, l1: int, l2: int, threats: bool = False) -> int:
    nb = OUTPUT_BUCKETS
    return (buckets * 768 * hidden + hidden + nb * l1 * hidden + nb * l1
            + nb * l2 * l1 + nb * l2 + nb * l2 + nb + (N_THREAT_FEATURES * hidden if threats else 0))


def check_checkpoint(d: Path, buckets: int, hidden: int, l1: int, l2: int,
                     threats: bool = False) -> tuple[bool, str]:
    n = expected_floats(buckets, hidden, l1, l2, threats)
    f = d / "quantised.bin"
    if not f.exists() or f.stat().st_size < 4 * n or f.stat().st_size - 4 * n >= 64:
        return False, f"size {f.stat().st_size if f.exists() else 0} != {4 * n} (+<64 padding)"
    x = np.fromfile(f, dtype="<f4", count=n)
    if not np.isfinite(x).all():
        return False, f"non-finite weights: {int((~np.isfinite(x)).sum())}"
    m = float(np.abs(x).max())
    if m > MAX_ABS:
        return False, f"magnitude {m:.3g} > {MAX_ABS} at float {int(np.abs(x).argmax())}"
    return True, "ok"


def checkpoints(out: Path, net_id: str) -> list[tuple[int, Path]]:
    pat = re.compile(rf"^{re.escape(net_id)}-(\d+)$")
    found = []
    if out.exists():
        for d in out.iterdir():
            m = pat.match(d.name)
            if m and d.is_dir():
                found.append((int(m.group(1)), d))
    return sorted(found)


def complete(d: Path) -> bool:
    return all((d / p).exists() for p in ("raw.bin", "optimiser_state", "quantised.bin"))


def latest_good(out: Path, net_id: str, threats: bool = False, **shape) -> Path | None:
    for _, d in reversed(checkpoints(out, net_id)):
        if complete(d) and check_checkpoint(d, **shape, threats=threats)[0]:
            return d
    return None


def gate_due(n: int) -> bool:
    return n % GATE_EVERY == 0


def done_set(manifest: Path) -> set[int]:
    if not manifest.exists():
        return set()
    return {json.loads(ln)["n"] for ln in manifest.read_text().splitlines() if ln.strip()}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def settled(d: Path, age: float = 60.0) -> bool:
    """Bullet writes a checkpoint's files one after another; wait until all are old enough."""
    now = time.time()
    return complete(d) and all(now - p.stat().st_mtime > age for p in d.rglob("*"))


def process(n: int, d: Path, net_id: str, manifest: Path, cfg: RunConfig = RunConfig()) -> dict:
    ok, why = check_checkpoint(d, **SHAPE, threats=cfg.threats)
    rec = {"n": n, "dir": str(d), "ok": ok, "check": why, "checked_at": time.time(),
           "quantised_sha256": sha256(d / "quantised.bin")}
    if ok:
        net = cfg.nets / f"{net_id}-{n}.nnue"
        r = subprocess.run([str(BULLET), "export", "--input", str(d / "quantised.bin"), "--output", str(net)],
                           capture_output=True, text=True)
        rec["export_rc"] = r.returncode
        m = re.search(r"worst-case accumulator (\d+)", r.stdout)
        rec["worst_case_accumulator"] = int(m.group(1)) if m else None
        rec["export_warning"] = r.stderr.strip()[-400:] or None
        if r.returncode == 0:
            rec["nnue"] = str(net)
            rec["nnue_sha256"] = sha256(net)
        else:
            rec["ok"] = False
    if not rec["ok"]:
        cfg.alert.write_text(json.dumps(rec) + "\n")
    with open(manifest, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def gate_cmd(net: Path, out: Path, cfg: RunConfig = RunConfig()) -> list[str]:
    opts = ["UseNNUE=true", "EvalBlend=100", "Hash=64"]
    cmd = ["flock", str(LOCK), sys.executable, str(REPO / "scripts/experiments/uci_match.py"),
           "--a", str(cfg.engine), "--b", str(cfg.engine),
           "--games", "200", "--tc", "10+0.1", "--cpu-pairs", CPU_PAIRS,
           "--book", str(REPO / "books/openings_v1.fen"), "--seed", "77",
           "--out", str(out), "--resume"]
    for o in [f"NNUEQuantFile={net}", *opts]:
        cmd += ["--a-opt", o]
    for o in [f"NNUEQuantFile={cfg.incumbent}", *opts]:
        cmd += ["--b-opt", o]
    return cmd


def watch(out: Path, net_id: str, poll: float, cfg: RunConfig = RunConfig()) -> None:
    cfg.nets.mkdir(parents=True, exist_ok=True)
    cfg.gates.mkdir(parents=True, exist_ok=True)
    manifest = cfg.nets / "manifest.jsonl"
    gate_queue: list[tuple[int, Path]] = []
    gate_proc: subprocess.Popen | None = None
    # Gates whose result file is missing or unfinished are (re)queued on start.
    for ln in (manifest.read_text().splitlines() if manifest.exists() else []):
        rec = json.loads(ln)
        res = cfg.gate_result(net_id, rec["n"])
        if rec.get("ok") and gate_due(rec["n"]):
            done = res.exists() and json.loads(res.read_text())["summary"].get("games", 0) >= 200
            if not done:
                gate_queue.append((rec["n"], Path(rec["nnue"])))
    while True:
        done = done_set(manifest)
        for n, d in checkpoints(out, net_id):
            if n in done or not settled(d):
                continue
            rec = process(n, d, net_id, manifest, cfg)
            print(json.dumps({k: rec.get(k) for k in ("n", "ok", "check", "worst_case_accumulator")}), flush=True)
            if rec["ok"] and gate_due(n):
                gate_queue.append((n, Path(rec["nnue"])))
        if gate_proc is not None and gate_proc.poll() is not None:
            print(f"gate finished rc={gate_proc.returncode}", flush=True)
            gate_proc = None
        if gate_proc is None and gate_queue:
            n, net = gate_queue.pop(0)
            res = cfg.gate_result(net_id, n)
            log = open(cfg.gates / f"{net_id}-{n}.log", "a")
            gate_proc = subprocess.Popen(gate_cmd(net, res, cfg), stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
            print(f"gate started for {net_id}-{n}", flush=True)
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["watch", "latest-good"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--net-id", default="w2048")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--threats", action="store_true", help="file version 6 checkpoints (threat inputs)")
    ap.add_argument("--nets", type=Path, default=NETS)
    ap.add_argument("--gates", type=Path, default=GATES)
    ap.add_argument("--alert", type=Path, default=ALERT)
    ap.add_argument("--engine", type=Path, default=ENGINE)
    ap.add_argument("--incumbent", type=Path, default=INCUMBENT)
    ap.add_argument("--incumbent-label", default="pw1500")
    ap.add_argument("--hidden", type=int, default=SHAPE["hidden"])
    a = ap.parse_args()
    SHAPE["hidden"] = a.hidden
    cfg = RunConfig(threats=a.threats, engine=a.engine, incumbent=a.incumbent, incumbent_label=a.incumbent_label,
                    nets=a.nets, gates=a.gates, alert=a.alert)
    if a.cmd == "latest-good":
        d = latest_good(Path(a.out), a.net_id, threats=a.threats, **SHAPE)
        if d is not None:
            print(d)
        return
    watch(Path(a.out), a.net_id, a.poll, cfg)


if __name__ == "__main__":
    main()
