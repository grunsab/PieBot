"""Generate the committed arch-v2 cross-language gold fixture.

Builds a deterministic tiny v2 float checkpoint, quantizes it through the
production `_export_v2_checkpoint` path to PIENNQ02, computes expected
white-POV evals for the feature-fixture FENs with an independent integer
reference, and writes both artifacts for the Rust test to assert against.

Run from the repo root:  python3 scripts/experiments/gen_arch_v2_gold.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.nnue import features_v2  # noqa: E402
from training.nnue.run_pipeline import _export_v2_checkpoint  # noqa: E402

HIDDEN = 4
QA, QB, SCALE = 255, 64, 400
INPUT = features_v2.PER_PERSPECTIVE_DIM
OUT = Path("PieBot/tests/data")


def lcg_stream(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield state


def main():
    rng = lcg_stream(0xA5C4_20260808)
    # Small float weights so accumulators land in the SCReLU active region.
    w1 = [((next(rng) % 2001) - 1000) / 25000.0 for _ in range(HIDDEN * INPUT)]
    b1 = [((next(rng) % 2001) - 1000) / 5000.0 for _ in range(HIDDEN)]
    w2 = [((next(rng) % 2001) - 1000) / 1000.0 for _ in range(2 * HIDDEN)]
    b2 = 0.0625
    checkpoint = {
        "arch": "v2",
        "input_dim": INPUT,
        "hidden_dim": HIDDEN,
        "quant_qa": QA,
        "quant_qb": QB,
        "wdl_scale_cp": float(SCALE),
        "w1": w1,  # row-major [hidden][input]
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }
    quant_path = OUT / "arch_v2_gold.nnue"
    meta = _export_v2_checkpoint(checkpoint, quant_path=quant_path)
    assert meta["quant_format"] == "PIENNQ02"

    # Independent integer reference (must mirror the Rust head exactly).
    w1_q = [
        max(-32768, min(32767, round(w1[h * INPUT + i] * QA)))
        for i in range(INPUT)
        for h in range(HIDDEN)
    ]  # feature-major
    b1_q = [max(-32768, min(32767, round(v * QA))) for v in b1]
    w2_q = [max(-128, min(127, round(v * QB))) for v in w2]
    b2_q = round(b2 * QA * QA * QB)

    fixture = json.loads((OUT / "halfkp_dp_fixture.json").read_text())
    expected = []
    for row in fixture["positions"]:
        fen = row["fen"]
        white_p, black_p, stm_white = features_v2.active_indices(fen)
        accs = []
        for idxs in (white_p, black_p):
            acc = list(b1_q)
            for idx in idxs:
                for h in range(HIDDEN):
                    acc[h] += w1_q[idx * HIDDEN + h]
            accs.append(acc)
        first, second = (accs[0], accs[1]) if stm_white else (accs[1], accs[0])
        total = 0
        for h in range(HIDDEN):
            v = max(0, min(QA, first[h]))
            total += v * v * w2_q[h]
        for h in range(HIDDEN):
            v = max(0, min(QA, second[h]))
            total += v * v * w2_q[HIDDEN + h]
        out_stm = (total + b2_q) * SCALE // (QA * QA * QB)
        # Python // floors; Rust i64 division truncates toward zero.
        num = (total + b2_q) * SCALE
        den = QA * QA * QB
        out_stm = abs(num) // den * (1 if num >= 0 else -1)
        white_pov = out_stm if stm_white else -out_stm
        expected.append({"fen": fen, "eval_white_pov_cp": int(white_pov)})

    (OUT / "arch_v2_gold_expected.json").write_text(
        json.dumps({"hidden": HIDDEN, "qa": QA, "qb": QB, "scale": SCALE,
                    "positions": expected}, indent=1) + "\n"
    )
    print("wrote", quant_path, "and expected evals:")
    for row in expected:
        print(f"  {row['eval_white_pov_cp']:>7} cp  {row['fen']}")


if __name__ == "__main__":
    main()
