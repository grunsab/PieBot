#!/usr/bin/env python3
"""Write the PIENNQ03 version-4 parity fixture used by PieBot/tests/nnue_arch_v3.rs.

A small random net (width 32, 4 factorised material experts) with output
weights spread over the whole i16 range, plus quant_eval_fen's integer
evaluations on positions from random games (all material phases). The engine
must reproduce every evaluation exactly.

    python3 training/nnue/make_v3p_fixture.py PieBot/tests/data
"""
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v3 as tv3  # noqa: E402


def positions(n, seed):
    rng = np.random.default_rng(seed)
    out, b = [], chess.Board()
    while len(out) < n:
        if b.is_game_over() or b.ply() > 220:
            b = chess.Board()
        moves = list(b.legal_moves)
        b.push(moves[int(rng.integers(len(moves)))])
        if rng.random() < 0.15:
            out.append(b.fen())
    return out


def main(out_dir: str) -> None:
    out = Path(out_dir)
    tv3.set_experts(4, (26, 17, 11))
    torch.manual_seed(404)
    model = tv3.build_model(32, factorised=True)
    with torch.no_grad():
        model.ft.normal_(0.0, 0.05)
        model.ft[tv3.N_FEATURES].zero_()
        model.out_w.uniform_(-tv3.W2_LIMIT, tv3.W2_LIMIT)
        model.out_b.uniform_(-1.0, 1.0)
    tv3.export(model, out / "v3p_fixture.nnue", 32, version=tv3.FILE_VERSION_PRECISE)
    net = tv3.load_quant(out / "v3p_fixture.nnue")
    cases = [(fen, tv3.quant_eval_fen(net, fen)) for fen in positions(80, 9)]
    (out / "v3p_fixture_evals.json").write_text(json.dumps(cases))
    print(f"{len(cases)} cases; eval range {min(c[1] for c in cases)}..{max(c[1] for c in cases)}")


if __name__ == "__main__":
    main(sys.argv[1])
