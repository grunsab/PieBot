#!/usr/bin/env python3
"""Write the threat-input (PIENNQ03 version 6) parity fixture used by
PieBot/tests/nnue_threats.rs.

The net is the version-5 fixture (v3pw_fixture.nnue, width 64) plus threat
weights from a fixed formula, `threat_weight(f, j)`, which the Rust test
recomputes, so no large binary is committed. The JSON holds, per position,
both perspectives' threat features (threats.threat_features) and
quant_eval_fen's evaluation. Positions come from random games, plus a few
chosen for castling, en passant and promotions.

    python3 training/nnue/make_v3t_fixture.py PieBot/tests/data
"""
import json
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_v3p_fixture  # noqa: E402
import threats  # noqa: E402
import train_v3 as tv3  # noqa: E402

EXTRA = [
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r1b1k2r/pP3ppp/2n5/8/8/8/1p3PPP/R1B1K2R b KQkq - 0 12",
    "1Q6/7k/2q5/8/3BN3/2R1b3/5n2/r3K3 w - - 0 40",
]


def threat_weight(f, j):
    """i8 weight of threat feature f, accumulator lane j (also in the Rust test)."""
    return ((31 * f + 17 * j) % 41) - 20


def main(out_dir: str) -> None:
    out = Path(out_dir)
    src = out / "v3pw_fixture.nnue"
    hidden = tv3.load_quant(src)["hidden"]
    f = np.arange(threats.N_THREAT_FEATURES, dtype=np.int64)[:, None]
    j = np.arange(hidden, dtype=np.int64)[None, :]
    wt = threat_weight(f, j).astype(np.int8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v3t.nnue"
        threats.write_threat_net(src, path, wt)
        net = tv3.load_quant(path)
    cases = []
    for fen in make_v3p_fixture.positions(120, 13) + EXTRA:
        b = chess.Board(fen)
        cases.append({"fen": fen, "white": threats.threat_features(b, 0),
                      "black": threats.threat_features(b, 1), "eval": tv3.quant_eval_fen(net, fen)})
    (out / "v3t_fixture.json").write_text(json.dumps(cases))
    evals = [c["eval"] for c in cases]
    most = max(max(len(c["white"]), len(c["black"])) for c in cases)
    print(f"{len(cases)} cases; eval range {min(evals)}..{max(evals)}; most threats {most}")


if __name__ == "__main__":
    main(sys.argv[1])
