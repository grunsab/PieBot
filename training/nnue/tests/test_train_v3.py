import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import chess
    import torch
except ImportError:  # pragma: no cover
    torch = None


def random_boards(n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    b = chess.Board()
    while len(out) < n:
        if b.is_game_over() or b.ply() > 160:
            b = chess.Board()
        moves = list(b.legal_moves)
        b.push(moves[int(rng.integers(len(moves)))])
        if not b.is_check():
            out.append(b.copy())
    return out


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3Tests(unittest.TestCase):
    def setUp(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        self.conv, self.tv3 = conv, tv3

    def records(self, boards):
        arr = np.zeros(len(boards), dtype=self.conv.V3_DTYPE)
        for i, b in enumerate(boards):
            occ, pcs = self.conv.encode_board(b)
            arr[i]["occ"] = occ
            arr[i]["pcs"] = np.frombuffer(pcs, dtype=np.uint8)
            arr[i]["stm"] = 0 if b.turn else 1
        return arr

    def test_gpu_features_match_the_reference(self):
        boards = random_boards(300)
        arr = self.records(boards)
        ia, ib, bk = self.tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        for i, b in enumerate(boards):
            pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in b.piece_map().items()]
            fw, fbk = self.tv3.piece_features(pairs, b.king(chess.WHITE), b.king(chess.BLACK))
            stm, nstm = (fw, fbk) if b.turn else (fbk, fw)
            got_a = sorted(x for x in ia[i].tolist() if x != self.tv3.N_FEATURES)
            got_b = sorted(x for x in ib[i].tolist() if x != self.tv3.N_FEATURES)
            self.assertEqual(got_a, sorted(stm))
            self.assertEqual(got_b, sorted(nstm))
            self.assertEqual(int(bk[i]), self.tv3.output_bucket(len(pairs)))

    def test_mirroring_makes_mirrored_positions_identical(self):
        # A position and its left-right mirror activate the same features.
        b = chess.Board("r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w - - 0 1")
        m = b.transform(chess.flip_horizontal)
        for x in (b, m):
            pass
        pairs = lambda bb: [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in bb.piece_map().items()]
        f1 = self.tv3.piece_features(pairs(b), b.king(chess.WHITE), b.king(chess.BLACK))
        f2 = self.tv3.piece_features(pairs(m), m.king(chess.WHITE), m.king(chess.BLACK))
        self.assertEqual([sorted(x) for x in f1], [sorted(x) for x in f2])

    def test_export_round_trip_matches_the_float_model(self):
        torch.manual_seed(3)
        model = self.tv3.build_model(32)
        # Snap every weight onto its quantisation grid, so the integer net is
        # exactly the float net and only the final truncation can differ.
        tv3 = self.tv3
        with torch.no_grad():
            model.ft.copy_(torch.round(model.ft * tv3.QA) / tv3.QA)
            model.ft_bias.copy_(torch.round(model.ft_bias * tv3.QA) / tv3.QA)
            model.out_w.copy_(torch.round(model.out_w * tv3.QB) / tv3.QB)
            model.out_b.copy_(torch.round(model.out_b * tv3.QA * tv3.QA * tv3.QB) / (tv3.QA * tv3.QA * tv3.QB))
        model = model.double()
        boards = random_boards(64, seed=11)
        arr = self.records(boards)
        ia, ib, bk = self.tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        with torch.no_grad():
            want = (model(ia, ib, bk) * self.tv3.EVAL_SCALE).numpy()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.nnue"
            self.tv3.export(model, p, 32)
            net = self.tv3.load_quant(p)
        got = np.array([self.tv3.quant_eval_fen(net, b.fen()) for b in boards])
        self.assertLessEqual(np.max(np.abs(got - want)), 1.0, (got[:5], want[:5]))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3HiddenTests(unittest.TestCase):
    def test_hidden_stack_export_tracks_the_float_model(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        torch.manual_seed(77)
        model = tv3.build_model(32, 16, 32)
        with torch.no_grad():
            model.h1_w.mul_(8.0)
            model.h1_b.uniform_(0.0, 0.5)
            model.h2_b.uniform_(0.0, 0.5)
        boards = random_boards(80, seed=13)
        arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
        for i, b in enumerate(boards):
            occ, pcs = conv.encode_board(b)
            arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
        ia, ib, bk = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        with torch.no_grad():
            want = (model(ia, ib, bk) * tv3.EVAL_SCALE).numpy()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.nnue"
            tv3.export(model, p, 32)
            net = tv3.load_quant(p)
        self.assertEqual((net["l1"], net["l2"]), (16, 32))
        got = np.array([tv3.quant_eval_fen(net, b.fen()) for b in boards])
        self.assertLess(np.mean(np.abs(got - want)), 3.0)
        self.assertLess(np.max(np.abs(got - want)), 12.0)


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3BucketTests(unittest.TestCase):
    def test_32_buckets_keep_gpu_and_reference_features_identical(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        tv3.set_king_buckets(32)
        try:
            boards = random_boards(200, seed=21)
            arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
            for i, b in enumerate(boards):
                occ, pcs = conv.encode_board(b)
                arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
            ia, ib, _ = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
            for i, b in enumerate(boards):
                pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in b.piece_map().items()]
                fw, fbk = tv3.piece_features(pairs, b.king(chess.WHITE), b.king(chess.BLACK))
                stm = fw if b.turn else fbk
                self.assertEqual(sorted(x for x in ia[i].tolist() if x != tv3.N_FEATURES), sorted(stm))
                self.assertTrue(max(stm) < 32 * 768)
        finally:
            tv3.set_king_buckets(16)


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3ExpertTests(unittest.TestCase):
    def test_experts_route_by_piece_count_on_both_paths(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        tv3.set_experts(4)
        try:
            boards = random_boards(300, seed=31)
            arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
            for i, b in enumerate(boards):
                occ, pcs = conv.encode_board(b)
                arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
            ia, ib, _ = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
            seen = set()
            for i, b in enumerate(boards):
                pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in b.piece_map().items()]
                fw, fbk = tv3.piece_features(pairs, b.king(chess.WHITE), b.king(chess.BLACK))
                stm = fw if b.turn else fbk
                self.assertEqual(sorted(x for x in ia[i].tolist() if x != tv3.N_FEATURES), sorted(stm))
                e = tv3.experts_table(4)[len(pairs)]
                seen.add(e)
                self.assertTrue(all(e * 16 * 768 <= f < (e + 1) * 16 * 768 for f in stm))
            self.assertGreaterEqual(len(seen), 3)
        finally:
            tv3.set_experts(1)


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3FactorisedExpertTests(unittest.TestCase):
    """Experts as a shared base plus per-expert deltas (Consilium's trainer)."""

    def setUp(self):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        self.conv, self.tv3 = conv, tv3

    def tearDown(self):
        self.tv3.set_experts(1)

    def features(self, boards):
        arr = np.zeros(len(boards), dtype=self.conv.V3_DTYPE)
        for i, b in enumerate(boards):
            occ, pcs = self.conv.encode_board(b)
            arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
        return self.tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))

    def test_custom_cuts_route_piece_counts(self):
        self.tv3.set_experts(4, cuts=(24, 16, 10))
        t = self.tv3.EXPERT_OF_PIECES
        self.assertEqual([t[32], t[24], t[23], t[16], t[15], t[10], t[9], t[2]], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_zero_deltas_reproduce_the_base_network(self):
        tv3 = self.tv3
        torch.manual_seed(5)
        base = tv3.build_model(32)
        boards = random_boards(80, seed=13)
        with torch.no_grad():
            want = base(*self.features(boards))
        tv3.set_experts(4, cuts=(24, 16, 10))
        moe = tv3.build_model(32, factorised=True)
        tv3.init_from_single(moe, base.state_dict())
        with torch.no_grad():
            got = moe(*self.features(boards))
        self.assertTrue(torch.allclose(got, want, atol=1e-5), (got[:4], want[:4]))

    def test_export_folds_base_and_deltas(self):
        tv3 = self.tv3
        tv3.set_experts(4, cuts=(24, 16, 10))
        torch.manual_seed(9)
        model = tv3.build_model(32, factorised=True)
        with torch.no_grad():
            model.ft.normal_(0.0, 0.05)
            model.ft[tv3.N_FEATURES].zero_()
            for w in (model.ft, model.ft_base, model.ft_bias):
                w.copy_(torch.round(w * tv3.QA) / tv3.QA)
            model.out_w.copy_(torch.round(model.out_w * tv3.QB) / tv3.QB)
            model.out_b.zero_()
        model = model.double()
        boards = random_boards(64, seed=17)
        with torch.no_grad():
            want = (model(*self.features(boards)) * tv3.EVAL_SCALE).numpy()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.nnue"
            tv3.export(model, p, 32)
            net = tv3.load_quant(p)
        self.assertEqual(net["n_experts"], 4)
        got = np.array([tv3.quant_eval_fen(net, b.fen()) for b in boards])
        self.assertLessEqual(np.max(np.abs(got - want)), 1.0, (got[:5], want[:5]))


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3PreciseHeadTests(unittest.TestCase):
    """File version 4: i16 output weights, so w2 rounding stops costing centipawns."""

    def errors(self, version, experts=1):
        import lc0_to_v3bin as conv
        import train_v3 as tv3
        tv3.set_experts(experts, (26, 17, 11) if experts > 1 else None)
        try:
            torch.manual_seed(21)
            model = tv3.build_model(64)
            with torch.no_grad():
                # Input layer and biases on their grids; output weights NOT snapped.
                model.ft.copy_(torch.round(model.ft * tv3.QA) / tv3.QA)
                model.ft_bias.copy_(torch.round(model.ft_bias * tv3.QA) / tv3.QA)
                model.out_w.uniform_(-1.5, 1.5)
                model.out_b.uniform_(-0.5, 0.5)
            model = model.double()
            boards = random_boards(120, seed=23)
            arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
            for i, b in enumerate(boards):
                occ, pcs = conv.encode_board(b)
                arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
            ia, ib, bk = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
            with torch.no_grad():
                want = (model(ia, ib, bk) * tv3.EVAL_SCALE).numpy()
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "n.nnue"
                tv3.export(model, p, 64, version=version)
                net = tv3.load_quant(p)
            self.assertEqual(net["version"], version)
            got = np.array([tv3.quant_eval_fen(net, b.fen()) for b in boards])
            return np.abs(got - want)
        finally:
            tv3.set_experts(1)

    def test_precise_head_matches_the_float_model(self):
        err = self.errors(4)
        self.assertLessEqual(err.max(), 1.5, err.max())
        self.assertLess(err.mean(), 0.75, err.mean())

    def test_legacy_i8_head_is_measurably_worse(self):
        self.assertGreater(self.errors(1).max(), 3.0)

    def test_precise_head_with_experts(self):
        self.assertLessEqual(self.errors(4, experts=4).max(), 1.5)

    def test_linear_heads_export_version_4_by_default(self):
        import train_v3 as tv3
        model = tv3.build_model(16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.nnue"
            tv3.export(model, p, 16)
            self.assertEqual(tv3.load_quant(p)["version"], 4)


@unittest.skipIf(torch is None, "torch/chess not installed")
class V3PairwiseTests(unittest.TestCase):
    """File version 5: hidden stack fed by a pairwise-multiplied accumulator."""

    def records(self, boards):
        import lc0_to_v3bin as conv
        arr = np.zeros(len(boards), dtype=conv.V3_DTYPE)
        for i, b in enumerate(boards):
            occ, pcs = conv.encode_board(b)
            arr[i]["occ"], arr[i]["pcs"], arr[i]["stm"] = occ, np.frombuffer(pcs, dtype=np.uint8), 0 if b.turn else 1
        return arr

    def test_pairwise_halves_the_first_layer_input(self):
        import train_v3 as tv3
        model = tv3.build_model(32, 16, 32, pairwise=True)
        self.assertEqual(tuple(model.h1_w.shape), (8 * 16, 32))

    def test_pairwise_activation_is_the_product_of_the_two_halves(self):
        import train_v3 as tv3
        torch.manual_seed(3)
        model = tv3.build_model(8, 4, 4, pairwise=True)
        acc = torch.tensor([[-0.5, 0.2, 1.5, 0.9, 0.5, 0.5, 0.3, 2.0]])
        x = model.pairwise_act(acc)
        want = torch.tensor([[0.0, 0.1, 0.3, 0.9]])
        self.assertTrue(torch.allclose(x, want), x)

    def test_pairwise_forward_keeps_its_activations_for_a_sparsity_penalty(self):
        import train_v3 as tv3
        torch.manual_seed(4)
        model = tv3.build_model(32, 16, 32, pairwise=True)
        boards = random_boards(8, seed=3)
        arr = self.records(boards)
        ia, ib, bk = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        model(ia, ib, bk)
        self.assertEqual(tuple(model.last_act.shape), (8, 32))
        pen = tv3.activation_penalty(model)
        self.assertAlmostEqual(pen.item(), model.last_act.mean().item(), places=6)
        pen.backward()
        self.assertIsNotNone(model.ft.grad)

    def test_pairwise_export_tracks_the_float_model(self):
        import train_v3 as tv3
        torch.manual_seed(78)
        model = tv3.build_model(32, 16, 32, pairwise=True)
        with torch.no_grad():
            model.h1_w.mul_(8.0)
            model.h1_b.uniform_(0.0, 0.5)
            model.h2_b.uniform_(0.0, 0.5)
        boards = random_boards(80, seed=19)
        arr = self.records(boards)
        ia, ib, bk = tv3.gpu_features(arr["occ"], arr["pcs"], arr["stm"], torch.device("cpu"))
        with torch.no_grad():
            want = (model(ia, ib, bk) * tv3.EVAL_SCALE).numpy()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pw.nnue"
            tv3.export(model, p, 32)
            net = tv3.load_quant(p)
        self.assertEqual(net["version"], tv3.FILE_VERSION_PAIRWISE)
        self.assertTrue(net["pairwise"])
        self.assertEqual(net["h1_w"].shape, (8, 16, 32))
        got = np.array([tv3.quant_eval_fen(net, b.fen()) for b in boards])
        self.assertLess(np.mean(np.abs(got - want)), 3.0)
        self.assertLess(np.max(np.abs(got - want)), 12.0)
