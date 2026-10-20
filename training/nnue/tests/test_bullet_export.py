"""Parity between the bullet trainer (training/bullet) and train_v3.py.

1. The Rust exporter must write the same PIENNQ03 bytes as train_v3.export
   for the same float weights.
2. For a bullet checkpoint (set PIEBOT_BULLET_CHECKPOINT; needs a GPU build):
   bullet's own float eval must equal a numpy forward pass of the exported
   floats through train_v3's feature map (layout, features, perspectives),
   and the integer eval (train_v3.quant_eval_fen, which the engine matches)
   must equal the same pass on the dequantised weights within 1 cp. The gap
   between float and integer is the format's quantisation error (mostly the
   i8 output weights at QB = 64) and is printed, not asserted.

The binary is training/bullet/target/release/piebot-bullet, or the path in
PIEBOT_BULLET_BIN; tests skip when it (or torch/chess) is missing.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
REPO = HERE.parents[3]
BIN = Path(os.environ.get("PIEBOT_BULLET_BIN", REPO / "training/bullet/target/release/piebot-bullet"))

try:
    import chess
    import torch
except ImportError:  # pragma: no cover
    torch = None


def random_fens(n, seed=5):
    rng = np.random.default_rng(seed)
    out, b = [], chess.Board()
    while len(out) < n:
        if b.is_game_over() or b.ply() > 120:
            b = chess.Board()
        moves = list(b.legal_moves)
        b.push(moves[int(rng.integers(len(moves)))])
        if not b.is_check() and b.ply() > 6:
            out.append(b.fen())
    return out


class _Model:
    """Duck-typed stand-in for train_v3's model, holding fixed float weights."""

    l1 = 0

    def __init__(self, ft, fb, ow, ob):
        self._ft = torch.tensor(ft)
        self.ft_bias = torch.tensor(fb)
        self.out_w = torch.tensor(ow)
        self.out_b = torch.tensor(ob)

    def effective_ft(self):
        return self._ft


@unittest.skipIf(torch is None or not BIN.exists(), "needs torch, chess and the piebot-bullet binary")
class BulletExportTests(unittest.TestCase):
    def test_exporter_matches_train_v3_byte_for_byte(self):
        import train_v3 as tv3
        hidden, buckets = 32, 16
        tv3.set_king_buckets(buckets)
        rng = np.random.default_rng(1)
        nf = buckets * 768
        ft = rng.normal(0, 0.05, (nf, hidden)).astype(np.float32)
        fb = rng.normal(0, 0.2, hidden).astype(np.float32)
        ow = rng.uniform(-1.9, 1.9, (8, 2 * hidden)).astype(np.float32)
        ob = rng.normal(0, 0.1, 8).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw = np.concatenate([ft.ravel(), fb, ow.ravel(), ob]).astype("<f4").tobytes()
            raw += b"\0" * ((64 - len(raw) % 64) % 64)  # bullet pads files to 64 bytes
            (d / "quantised.bin").write_bytes(raw)
            for version in (4, 1):
                subprocess.run([str(BIN), "export", "--input", str(d / "quantised.bin"), "--output", str(d / "rust.nnue"),
                                "--buckets", str(buckets), "--hidden", str(hidden), "--file-version", str(version)],
                               check=True, capture_output=True)
                tv3.export(_Model(ft, fb, ow, ob), d / "py.nnue", hidden, version=version)
                self.assertEqual((d / "rust.nnue").read_bytes(), (d / "py.nnue").read_bytes(), f"version {version}")
                net = tv3.load_quant(d / "rust.nnue")
                self.assertEqual((net["version"], net["nkb"], net["hidden"], net["nout"]), (version, 16, hidden, 8))

    @unittest.skipUnless(os.environ.get("PIEBOT_BULLET_CHECKPOINT"), "set PIEBOT_BULLET_CHECKPOINT")
    def test_bullet_eval_matches_train_v3_features_and_integer_path(self):
        """Bullet's float eval == a numpy forward pass over train_v3's feature map
        on the exported floats (layout, features, perspectives); the engine's
        integer eval == the same pass on dequantised weights (format)."""
        import train_v3 as tv3
        ckpt = Path(os.environ["PIEBOT_BULLET_CHECKPOINT"])
        hidden = int(os.environ.get("PIEBOT_BULLET_HIDDEN", "1024"))
        buckets = int(os.environ.get("PIEBOT_BULLET_BUCKETS", "16"))
        tv3.set_king_buckets(buckets)
        fens = random_fens(60)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "fens.txt").write_text("\n".join(fens) + "\n")
            out = subprocess.run([str(BIN), "evalfens", "--checkpoint", str(ckpt), "--fens", str(d / "fens.txt"),
                                  "--save", str(d / "net.nnue"), "--hidden", str(hidden), "--buckets", str(buckets)],
                                 check=True, capture_output=True, text=True).stdout
            rows = json.loads(out[out.index("[["):])
            net = tv3.load_quant(d / "net.nnue")
        nf = buckets * 768
        raw = np.fromfile(ckpt / "quantised.bin", dtype="<f4")
        sizes = [nf * hidden, hidden, 8 * 2 * hidden, 8]
        parts, off = [], 0
        for n in sizes:
            parts.append(raw[off:off + n].astype(np.float64)); off += n
        w1, b1, w2, b2 = parts[0].reshape(nf, hidden), parts[1], parts[2].reshape(8, 2 * hidden), parts[3]
        dq = (net["w1"] / tv3.QA, net["b1"] / tv3.QA, net["w2"] / tv3.QB, net["b2"] / (tv3.QA * tv3.QA * tv3.QB))

        def forward(W1, B1, W2, B2, fen):
            b = chess.Board(fen)
            pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in b.piece_map().items()]
            fw, fbk = tv3.piece_features(pairs, b.king(chess.WHITE), b.king(chess.BLACK))
            acc = [B1 + W1[fw].sum(0), B1 + W1[fbk].sum(0)]
            stm, nstm = (acc[0], acc[1]) if b.turn else (acc[1], acc[0])
            h = np.concatenate([np.clip(stm, 0, 1), np.clip(nstm, 0, 1)]) ** 2
            k = tv3.output_bucket(len(pairs))
            return float((h * W2[k]).sum() + B2[k]) * tv3.EVAL_SCALE

        float_err, int_err, quant_err = [], [], []
        for fen, bullet_cp in rows:
            ref = forward(w1, b1, w2, b2, fen)
            float_err.append(abs(ref - bullet_cp))
            int_err.append(abs(tv3.quant_eval_fen(net, fen) - forward(*dq, fen)))
            quant_err.append(tv3.quant_eval_fen(net, fen) - bullet_cp)
        quant_err = np.array(quant_err)
        print(f"\nbullet vs numpy float: max {max(float_err):.3f} cp; integer vs dequantised: max {max(int_err):.3f} cp; "
              f"quantisation error: mean {quant_err.mean():+.2f}, max |{np.abs(quant_err).max():.2f}| cp over {len(rows)} FENs")
        self.assertLess(max(float_err), 0.05)
        self.assertLessEqual(max(int_err), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
