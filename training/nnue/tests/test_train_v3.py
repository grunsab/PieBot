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
