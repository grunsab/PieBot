#!/usr/bin/env python3
"""Write the PIENNQ03 version-5 (pairwise hidden stack) parity fixture used by
PieBot/tests/nnue_arch_v3.rs.

A small random net (width 64, so each pairwise half is 32 wide and the
engine's SIMD activation runs, L1 16, L2 32) and quant_eval_fen's
evaluations on positions from random games. The f32 layers after L1 may
round differently from numpy, so the engine must agree within 1 cp.

    python3 training/nnue/make_v3pw_fixture.py PieBot/tests/data
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_v3p_fixture  # noqa: E402
import train_v3 as tv3  # noqa: E402


def main(out_dir: str) -> None:
    out = Path(out_dir)
    torch.manual_seed(505)
    model = tv3.build_model(64, 16, 32, pairwise=True)
    with torch.no_grad():
        model.ft.normal_(0.0, 0.15)
        model.ft[tv3.N_FEATURES].zero_()
        model.ft_bias.uniform_(-0.2, 0.8)
        model.h1_w.mul_(8.0)
        model.h1_b.uniform_(0.0, 0.5)
        model.h2_b.uniform_(0.0, 0.5)
    tv3.export(model, out / "v3pw_fixture.nnue", 64)
    net = tv3.load_quant(out / "v3pw_fixture.nnue")
    assert net["version"] == tv3.FILE_VERSION_PAIRWISE
    cases = [(fen, tv3.quant_eval_fen(net, fen)) for fen in make_v3p_fixture.positions(80, 11)]
    (out / "v3pw_fixture_evals.json").write_text(json.dumps(cases))
    print(f"{len(cases)} cases; eval range {min(c[1] for c in cases)}..{max(c[1] for c in cases)}")


if __name__ == "__main__":
    main(sys.argv[1])
