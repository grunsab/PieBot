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

3. Factorised material experts (--experts 4): the exporter matches
   train_v3.export with experts byte for byte; bullet's expert features
   match train_v3.piece_features on real binpack positions (set
   PIEBOT_BINPACK_SAMPLE); and a net started with --init-from a one-expert
   checkpoint evaluates exactly like that checkpoint (zero deltas).

4. Pairwise hidden stack (--pairwise, file version 5): the exporter matches
   train_v3.export byte for byte, and for a pairwise checkpoint (set
   PIEBOT_BULLET_PAIRWISE_CHECKPOINT) bullet's float eval equals train_v3's
   torch model (build_model(..., pairwise=True)) loaded with the same floats.

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
        qb = net["qb"]  # 4096 for file version 4 (the default), 64 for version 1
        dq = (net["w1"] / tv3.QA, net["b1"] / tv3.QA, net["w2"] / qb, net["b2"] / (tv3.QA * tv3.QA * qb))

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


CUTS = (26, 17, 11)


@unittest.skipIf(torch is None or not BIN.exists(), "needs torch, chess and the piebot-bullet binary")
class BulletExpertsTests(unittest.TestCase):
    def tearDown(self):
        import train_v3 as tv3
        tv3.set_experts(1)

    def test_experts_exporter_matches_train_v3_byte_for_byte(self):
        import train_v3 as tv3
        hidden, buckets = 32, 16
        tv3.set_king_buckets(buckets)
        tv3.set_experts(4, cuts=CUTS)
        rng = np.random.default_rng(2)
        nf = 4 * buckets * 768
        ft = rng.normal(0, 0.05, (nf, hidden)).astype(np.float32)
        fb = rng.normal(0, 0.2, hidden).astype(np.float32)
        ow = rng.uniform(-1.9, 1.9, (8, 2 * hidden)).astype(np.float32)
        ob = rng.normal(0, 0.1, 8).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ck = d / "run" / "net-20"
            ck.mkdir(parents=True)
            raw = np.concatenate([ft.ravel(), fb, ow.ravel(), ob]).astype("<f4").tobytes()
            raw += b"\0" * ((64 - len(raw) % 64) % 64)
            (ck / "quantised.bin").write_bytes(raw)
            # The run directory's piebot_net.txt supplies the shape.
            (d / "run" / "piebot_net.txt").write_text(f"buckets={buckets}\nhidden={hidden}\nexperts=4\ncuts=26,17,11\n")
            for version, want in ((4, 4), (1, 3)):
                subprocess.run([str(BIN), "export", "--input", str(ck / "quantised.bin"), "--output", str(d / "rust.nnue"),
                                "--file-version", str(version)], check=True, capture_output=True)
                tv3.export(_Model(ft, fb, ow, ob), d / "py.nnue", hidden, version=version)
                self.assertEqual((d / "rust.nnue").read_bytes(), (d / "py.nnue").read_bytes(), f"version {version}")
                net = tv3.load_quant(d / "rust.nnue")
                self.assertEqual((net["version"], net["n_experts"]), (want, 4))
                self.assertEqual(net["expert_table"], tv3.experts_table(4, CUTS))
            # Explicit options give the same file as the sidecar.
            subprocess.run([str(BIN), "export", "--input", str(ck / "quantised.bin"), "--output", str(d / "cli.nnue"),
                            "--buckets", "16", "--hidden", str(hidden), "--experts", "4", "--expert-cuts", "26,17,11"],
                           check=True, capture_output=True)
            tv3.export(_Model(ft, fb, ow, ob), d / "py.nnue", hidden, version=4)
            self.assertEqual((d / "cli.nnue").read_bytes(), (d / "py.nnue").read_bytes())

    @unittest.skipUnless(os.environ.get("PIEBOT_BINPACK_SAMPLE"), "set PIEBOT_BINPACK_SAMPLE to a .binpack file")
    def test_expert_features_match_train_v3_on_real_positions(self):
        import train_v3 as tv3
        tv3.set_king_buckets(16)
        tv3.set_experts(4, cuts=CUTS)
        with tempfile.TemporaryDirectory() as d:
            dump = Path(d) / "f.tsv"
            subprocess.run([str(BIN), "features", "--data", os.environ["PIEBOT_BINPACK_SAMPLE"], "--experts", "4",
                            "--expert-cuts", "26,17,11", "--positions", "20000", "--dump", str(dump)],
                           check=True, capture_output=True)
            lines = dump.read_text().splitlines()
        self.assertGreaterEqual(len(lines), 20000)
        experts_seen = set()
        for line in lines:
            fen, feats = line.split("\t")
            b = chess.Board(fen)
            pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in b.piece_map().items()]
            fw, fbk = tv3.piece_features(pairs, b.king(chess.WHITE), b.king(chess.BLACK))
            stm = fw if b.turn else fbk
            self.assertEqual(sorted(map(int, feats.split())), sorted(stm), fen)
            experts_seen.add(tv3.EXPERT_OF_PIECES[len(pairs)])
        self.assertEqual(experts_seen, {0, 1, 2, 3})

    @unittest.skipUnless(os.environ.get("PIEBOT_BULLET_CHECKPOINT"), "set PIEBOT_BULLET_CHECKPOINT (a one-expert checkpoint)")
    def test_zero_deltas_evaluate_like_the_base_checkpoint(self):
        import train_v3 as tv3
        ckpt = os.environ["PIEBOT_BULLET_CHECKPOINT"]
        hidden = os.environ.get("PIEBOT_BULLET_HIDDEN", "1024")
        fens = random_fens(60, seed=9)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "fens.txt").write_text("\n".join(fens) + "\n")

            def run(*extra, save):
                out = subprocess.run([str(BIN), "evalfens", "--fens", str(d / "fens.txt"), "--hidden", hidden,
                                      "--save", str(d / save), *extra], check=True, capture_output=True, text=True).stdout
                return json.loads(out[out.index("[["):])

            base = run("--checkpoint", ckpt, save="base.nnue")
            moe = run("--experts", "4", "--expert-cuts", "26,17,11", "--init-from", ckpt, save="moe.nnue")
            nb, nm = tv3.load_quant(d / "base.nnue"), tv3.load_quant(d / "moe.nnue")
        self.assertEqual(nm["n_experts"], 4)
        for (fen, a), (_, b) in zip(base, moe):
            self.assertAlmostEqual(a, b, delta=0.01, msg=fen)
            self.assertEqual(tv3.quant_eval_fen(nb, fen), tv3.quant_eval_fen(nm, fen), fen)


PW_HIDDEN, PW_L1, PW_L2 = 32, 16, 32


def pairwise_floats(raw, hidden, l1, l2, buckets=16):
    """Split a pairwise checkpoint's float quantised.bin (bullet's save order)."""
    nf, nb = buckets * 768, 8
    sizes = [("ft", (nf, hidden)), ("fb", (hidden,)), ("h1_w", (nb * l1, hidden)), ("h1_b", (nb * l1,)),
             ("h2_w", (nb * l2, l1)), ("h2_b", (nb * l2,)), ("h3_w", (nb, l2)), ("h3_b", (nb,))]
    out, off = {}, 0
    for name, shape in sizes:
        n = int(np.prod(shape))
        out[name] = raw[off:off + n].reshape(shape)
        off += n
    return out


def pairwise_model(w, hidden, l1, l2):
    """train_v3's own pairwise model holding the given floats (linear head zero)."""
    import train_v3 as tv3
    m = tv3.build_model(hidden, l1, l2, pairwise=True)
    with torch.no_grad():
        m.ft.zero_()
        m.ft[:tv3.N_FEATURES] = torch.tensor(w["ft"])
        m.ft_bias.copy_(torch.tensor(w["fb"]))
        m.out_w.zero_()
        m.out_b.zero_()
        for k in ("h1_w", "h1_b", "h2_w", "h2_b", "h3_w", "h3_b"):
            getattr(m, k).copy_(torch.tensor(w[k]))
    return m


@unittest.skipIf(torch is None or not BIN.exists(), "needs torch, chess and the piebot-bullet binary")
class BulletPairwiseTests(unittest.TestCase):
    def test_pairwise_exporter_matches_train_v3_byte_for_byte(self):
        import train_v3 as tv3
        tv3.set_king_buckets(16)
        rng = np.random.default_rng(3)
        nf, nb = 16 * 768, 8
        w = {
            "ft": rng.normal(0, 0.05, (nf, PW_HIDDEN)), "fb": rng.normal(0.3, 0.2, PW_HIDDEN),
            "h1_w": rng.uniform(-1.9, 1.9, (nb * PW_L1, PW_HIDDEN)), "h1_b": rng.normal(0, 0.3, nb * PW_L1),
            "h2_w": rng.normal(0, 0.5, (nb * PW_L2, PW_L1)), "h2_b": rng.normal(0, 0.3, nb * PW_L2),
            "h3_w": rng.normal(0, 0.5, (nb, PW_L2)), "h3_b": rng.normal(0, 0.3, nb),
        }
        w = {k: v.astype(np.float32) for k, v in w.items()}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ck = d / "run" / "pw-20"
            ck.mkdir(parents=True)
            order = ("ft", "fb", "h1_w", "h1_b", "h2_w", "h2_b", "h3_w", "h3_b")
            raw = np.concatenate([w[k].ravel() for k in order]).astype("<f4").tobytes()
            raw += b"\0" * ((64 - len(raw) % 64) % 64)
            (ck / "quantised.bin").write_bytes(raw)
            tv3.export(pairwise_model(w, PW_HIDDEN, PW_L1, PW_L2), d / "py.nnue", PW_HIDDEN)
            subprocess.run([str(BIN), "export", "--input", str(ck / "quantised.bin"), "--output", str(d / "cli.nnue"),
                            "--buckets", "16", "--hidden", str(PW_HIDDEN), "--pairwise", "--l1", str(PW_L1),
                            "--l2", str(PW_L2)], check=True, capture_output=True)
            self.assertEqual((d / "cli.nnue").read_bytes(), (d / "py.nnue").read_bytes())
            # The run's piebot_net.txt supplies the same shape.
            (d / "run" / "piebot_net.txt").write_text(
                f"buckets=16\nhidden={PW_HIDDEN}\nexperts=1\ncuts=\npairwise_l1={PW_L1}\npairwise_l2={PW_L2}\n")
            subprocess.run([str(BIN), "export", "--input", str(ck / "quantised.bin"), "--output", str(d / "side.nnue")],
                           check=True, capture_output=True)
            self.assertEqual((d / "side.nnue").read_bytes(), (d / "py.nnue").read_bytes())
            net = tv3.load_quant(d / "side.nnue")
        self.assertEqual((net["version"], net["l1"], net["l2"], net["pairwise"]), (5, PW_L1, PW_L2, True))

    @unittest.skipUnless(os.environ.get("PIEBOT_BULLET_PAIRWISE_CHECKPOINT"), "set PIEBOT_BULLET_PAIRWISE_CHECKPOINT")
    def test_pairwise_bullet_eval_matches_train_v3_model(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        ckpt = Path(os.environ["PIEBOT_BULLET_PAIRWISE_CHECKPOINT"])
        hidden = int(os.environ.get("PIEBOT_BULLET_HIDDEN", "1024"))
        l1 = int(os.environ.get("PIEBOT_BULLET_L1", "16"))
        l2 = int(os.environ.get("PIEBOT_BULLET_L2", "32"))
        tv3.set_king_buckets(16)
        fens = random_fens(60, seed=21)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "fens.txt").write_text("\n".join(fens) + "\n")
            out = subprocess.run([str(BIN), "evalfens", "--checkpoint", str(ckpt), "--fens", str(d / "fens.txt"),
                                  "--save", str(d / "net.nnue"), "--hidden", str(hidden), "--pairwise",
                                  "--l1", str(l1), "--l2", str(l2)],
                                 check=True, capture_output=True, text=True).stdout
            rows = json.loads(out[out.index("[["):])
            net = tv3.load_quant(d / "net.nnue")
            if os.environ.get("PIEBOT_SAVE_PAIRWISE_NET"):
                Path(os.environ["PIEBOT_SAVE_PAIRWISE_NET"]).write_bytes((d / "net.nnue").read_bytes())
        w = pairwise_floats(np.fromfile(ckpt / "quantised.bin", dtype="<f4"), hidden, l1, l2)
        model = pairwise_model(w, hidden, l1, l2).double()
        boards = [chess.Board(f) for f, _ in rows]
        arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
        for i, b in enumerate(boards):
            occ, pcs = conv.encode_board(b)
            arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
        ia, ib, bk = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        with torch.no_grad():
            ref = (model(ia, ib, bk) * tv3.EVAL_SCALE).numpy()
        got = np.array([cp for _, cp in rows])
        quant = np.array([tv3.quant_eval_fen(net, f) for f, _ in rows])
        print(f"\npairwise: bullet vs train_v3 float max {np.abs(got - ref).max():.4f} cp; "
              f"integer vs float mean {np.mean(quant - ref):+.2f}, max |{np.abs(quant - ref).max():.2f}| cp")
        self.assertLess(np.abs(got - ref).max(), 0.05)


if __name__ == "__main__":
    unittest.main()
