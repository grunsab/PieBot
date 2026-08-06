#!/usr/bin/env python3
"""Teacher-agreement diagnostic: compare two PieBot models' fixed-depth labels.

For every FEN in a frozen probe file, search the position with the same
PieBot UCI binary under model A and model B (same depth, same blend) and
report best-move agreement and the centipawn delta distribution. Near-total
agreement between the active teacher and a newer candidate is the
self-distillation fixed-point signature (the campaign's tripwire).

The probe file is one FEN per line (# comments and blank lines skipped).
Output is a JSON report; positions are processed in file order and the run
is deterministic for a fixed (binary, models, depth, probe) tuple.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_probe_fens(path: Path) -> List[str]:
    fens: List[str] = []
    for line_index, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line.split()) < 4:
            raise ValueError(f"invalid FEN at {path} line {line_index}: {line!r}")
        fens.append(line)
    if not fens:
        raise ValueError(f"probe file {path} contains no FENs")
    return fens


class UciLabeler:
    """Drive one PieBot UCI process to label positions at a fixed depth."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        nnue_quant_file: str,
        blend: int,
        hash_mb: int = 256,
        startup_timeout_s: float = 30.0,
    ) -> None:
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok", startup_timeout_s)
        for name, value in (
            ("Hash", hash_mb),
            ("NNUEQuantFile", nnue_quant_file),
            ("UseNNUE", "true"),
            ("EvalBlend", blend),
            ("Threads", 1),
        ):
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        transcript = self._read_until("readyok", startup_timeout_s)
        for line in transcript:
            if "failed to load" in line.lower():
                raise RuntimeError(f"model failed to load: {line}")

    def _send(self, line: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _read_until(self, marker: str, timeout_s: float) -> List[str]:
        assert self._process.stdout is not None
        deadline = time.monotonic() + timeout_s
        transcript: List[str] = []
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"UCI engine exited before {marker}; last: {transcript[-5:]}"
                )
            transcript.append(line.strip())
            if line.strip().casefold().startswith(marker.casefold()):
                return transcript
        raise RuntimeError(f"timed out waiting for {marker}")

    def label(self, fen: str, depth: int, timeout_s: float = 60.0) -> Dict[str, Any]:
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        assert self._process.stdout is not None
        deadline = time.monotonic() + timeout_s
        last_cp: Optional[int] = None
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("UCI engine exited during search")
            text = line.strip()
            match = re.search(r"\bscore cp (-?\d+)", text)
            if match:
                last_cp = int(match.group(1))
            if text.startswith("bestmove"):
                parts = text.split()
                best = parts[1] if len(parts) > 1 else None
                return {"best_move": best, "score_cp": last_cp}
        raise RuntimeError(f"timed out labeling {fen}")

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()


def agreement_report(
    labels_a: Sequence[Dict[str, Any]], labels_b: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize best-move agreement and cp deltas between two label sets."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label sets must be non-empty and equal length")
    same_move = 0
    cp_deltas: List[float] = []
    for a, b in zip(labels_a, labels_b):
        if a.get("best_move") is not None and a.get("best_move") == b.get("best_move"):
            same_move += 1
        if a.get("score_cp") is not None and b.get("score_cp") is not None:
            cp_deltas.append(abs(float(a["score_cp"]) - float(b["score_cp"])))
    report: Dict[str, Any] = {
        "positions": len(labels_a),
        "best_move_agreement": same_move / len(labels_a),
    }
    if cp_deltas:
        cp_deltas_sorted = sorted(cp_deltas)
        report["cp_delta_mean"] = statistics.fmean(cp_deltas)
        report["cp_delta_median"] = statistics.median(cp_deltas_sorted)
        report["cp_delta_p90"] = cp_deltas_sorted[
            min(len(cp_deltas_sorted) - 1, math.floor(0.9 * len(cp_deltas_sorted)))
        ]
        report["cp_delta_max"] = cp_deltas_sorted[-1]
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piebot-command", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--blend", type=int, default=25)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all probe FENs")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    fens = load_probe_fens(args.probe)
    if args.limit > 0:
        fens = fens[: args.limit]
    command = shlex.split(args.piebot_command)

    labels: Dict[str, List[Dict[str, Any]]] = {}
    for key, model in (("a", args.model_a), ("b", args.model_b)):
        labeler = UciLabeler(command, nnue_quant_file=model, blend=args.blend)
        try:
            labels[key] = [labeler.label(fen, args.depth) for fen in fens]
        finally:
            labeler.close()

    report = {
        "schema": "piebot-teacher-agreement-v1",
        "depth": args.depth,
        "blend": args.blend,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "probe": str(args.probe),
        "summary": agreement_report(labels["a"], labels["b"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
