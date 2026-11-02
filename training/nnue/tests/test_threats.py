"""Threat input features (threats.py), the reference the engine and bullet must match.

A threat feature is (attacker piece, its square, attacked piece, its square)
from one side's point of view, using the piece features' orientation (ranks
flipped for Black, files mirrored when that side's king is on files e-h).
The pair rules follow Stockfish's FullThreats (59,808 features): kings never
attack or get attacked, pawns attack only knights and rooks, bishops and rooks
attack P/N/B/R, knights and queens attack P/N/B/R/Q, and a same-type pair
counts in one direction only.
"""

import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import chess  # noqa: E402

import threats  # noqa: E402

PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def random_positions(n, seed=1, max_plies=120):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randrange(0, max_plies)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        out.append(b.copy(stack=False))
    return out


def ray_attacks(board, sq):
    """Attacked squares by walking rays; independent of python-chess's tables."""
    p = board.piece_at(sq)
    f, r = chess.square_file(sq), chess.square_rank(sq)
    out = []
    def on(ff, rr):
        return 0 <= ff < 8 and 0 <= rr < 8
    if p.piece_type == chess.PAWN:
        dr = 1 if p.color == chess.WHITE else -1
        out = [chess.square(f + df, r + dr) for df in (-1, 1) if on(f + df, r + dr)]
    elif p.piece_type in (chess.KNIGHT, chess.KING):
        steps = ([(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
                 if p.piece_type == chess.KNIGHT else
                 [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)])
        out = [chess.square(f + a, r + b) for a, b in steps if on(f + a, r + b)]
    else:
        dirs = []
        if p.piece_type in (chess.BISHOP, chess.QUEEN):
            dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if p.piece_type in (chess.ROOK, chess.QUEEN):
            dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for a, b in dirs:
            ff, rr = f + a, r + b
            while on(ff, rr):
                s = chess.square(ff, rr)
                out.append(s)
                if board.piece_at(s):
                    break
                ff, rr = ff + a, rr + b
    return out


def slow_threat_features(board, persp):
    """Rule-by-rule restatement with ray-walked attacks and a dict index."""
    king = board.king(chess.WHITE if persp == 0 else chess.BLACK)
    k = king if persp == 0 else king ^ 56
    flip = (56 if persp else 0) ^ (7 if (k & 7) >= 4 else 0)
    allowed = {chess.PAWN: (chess.KNIGHT, chess.ROOK),
               chess.KNIGHT: (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN),
               chess.BISHOP: (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK),
               chess.ROOK: (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK),
               chess.QUEEN: (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN),
               chess.KING: ()}
    out = []
    for frm, a in board.piece_map().items():
        for to in ray_attacks(board, frm):
            b = board.piece_at(to)
            if b is None or b.piece_type not in allowed[a.piece_type]:
                continue
            fo, tt = frm ^ flip, to ^ flip
            ra = 0 if (a.color == chess.WHITE) == (persp == 0) else 1
            rb = 0 if (b.color == chess.WHITE) == (persp == 0) else 1
            if a.piece_type == b.piece_type and fo < tt:
                continue
            out.append(threats.TABLE[(ra, a.piece_type - 1, fo, tt, rb, b.piece_type - 1)])
    return sorted(out)


class ThreatIndexTests(unittest.TestCase):
    def test_feature_count_matches_stockfish_full_threats(self):
        self.assertEqual(threats.N_THREAT_FEATURES, 59808)

    def test_every_empty_board_pair_has_a_unique_index_and_they_fill_the_range(self):
        # TABLE holds every (attacker, from, to, attacked) the empty-board
        # attack masks allow, before the one-direction rule for same types.
        idx = list(threats.TABLE.values())
        self.assertEqual(len(idx), len(set(idx)))
        self.assertEqual(sorted(idx), list(range(threats.N_THREAT_FEATURES)))

    def test_index_function_agrees_with_the_table(self):
        rng = random.Random(3)
        keys = list(threats.TABLE)
        for key in rng.sample(keys, 5000):
            self.assertEqual(threats.index(*key), threats.TABLE[key])


class ThreatFeatureTests(unittest.TestCase):
    def test_start_position_has_28_threats_in_each_view(self):
        # Each colour has 14 (Nb1-d2, Ng1-e2, 2 per bishop, 2 per rook, Qd1 onto
        # c1/c2/d2/e2), and each perspective sees both colours' threats.
        b = chess.Board()
        self.assertEqual(len(threats.threat_features(b, 0)), 28)
        self.assertEqual(len(threats.threat_features(b, 1)), 28)

    def test_matches_the_slow_ray_walking_restatement(self):
        for b in random_positions(400, seed=11):
            for persp in (0, 1):
                self.assertEqual(threats.threat_features(b, persp), slow_threat_features(b, persp),
                                 b.fen())

    def test_black_view_equals_white_view_of_the_colour_flipped_board(self):
        for b in random_positions(200, seed=12):
            self.assertEqual(threats.threat_features(b, 1), threats.threat_features(b.mirror(), 0),
                             b.fen())

    def test_mirroring_files_does_not_change_the_features(self):
        for b in random_positions(200, seed=13):
            m = b.transform(chess.flip_horizontal)
            for persp in (0, 1):
                self.assertEqual(threats.threat_features(b, persp), threats.threat_features(m, persp),
                                 b.fen())

    def test_features_are_sorted_unique_and_bounded(self):
        for b in random_positions(300, seed=14):
            for persp in (0, 1):
                f = threats.threat_features(b, persp)
                self.assertEqual(f, sorted(set(f)))
                self.assertLessEqual(len(f), threats.MAX_ACTIVE)
                self.assertTrue(all(0 <= i < threats.N_THREAT_FEATURES for i in f))

    def test_no_king_and_no_pawn_onto_pawn_features(self):
        b = chess.Board("4k3/8/8/3p4/4P3/8/8/4K2R w - - 0 1")
        # e4xd5 is pawn onto pawn and Rh1-e1 defends the king: neither counts.
        self.assertEqual(threats.threat_features(b, 0), [])

    def test_same_type_pair_counts_once(self):
        b = chess.Board("4k3/8/8/8/8/2N5/8/1N2K3 w - - 0 1")
        # b1 and c3 are a knight's move apart: two attacks, one feature.
        self.assertEqual(len(threats.threat_features(b, 0)), 1)

    def test_max_active_covers_any_legal_position(self):
        # At most 30 non-king pieces, each attacking at most 8 pieces (one per
        # ray or knight jump; a pawn at most 2).
        self.assertGreaterEqual(threats.MAX_ACTIVE, 30 * 8)

FIXTURE = HERE.parents[3] / "PieBot/tests/data/v3pw_fixture.nnue"


class ThreatFileTests(unittest.TestCase):
    """File version 6: version 5 plus u32 n_threat_features and i8 weights."""

    def setUp(self):
        import tempfile
        import numpy as np
        import train_v3 as tv3
        self.np, self.tv3 = np, tv3
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.v5 = tv3.load_quant(FIXTURE)
        self.hidden = self.v5["hidden"]

    def write(self, wt):
        out = Path(self.tmp.name) / "v6.nnue"
        threats.write_threat_net(FIXTURE, out, wt)
        return out

    def test_zero_threat_weights_evaluate_like_the_version_5_net(self):
        net = self.tv3.load_quant(self.write(self.np.zeros((threats.N_THREAT_FEATURES, self.hidden), "i1")))
        self.assertEqual(net["version"], self.tv3.FILE_VERSION_THREATS)
        for b in random_positions(30, seed=21):
            self.assertEqual(self.tv3.quant_eval_fen(net, b.fen()), self.tv3.quant_eval_fen(self.v5, b.fen()))

    def test_threat_rows_are_added_to_each_perspectives_accumulator(self):
        rng = self.np.random.default_rng(5)
        wt = rng.integers(-20, 21, size=(threats.N_THREAT_FEATURES, self.hidden)).astype("i1")
        net = self.tv3.load_quant(self.write(wt))
        self.assertTrue((net["threat_w"] == wt).all())
        b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        accs = self.tv3.quant_accumulators(net, b.fen())
        base = self.tv3.quant_accumulators(self.v5, b.fen())
        for persp in (0, 1):
            want = base[persp] + wt[threats.threat_features(b, persp)].astype(self.np.int64).sum(0)
            self.assertTrue((accs[persp] == want).all())

    def test_a_wrong_threat_count_is_rejected(self):
        out = self.write(self.np.zeros((threats.N_THREAT_FEATURES, self.hidden), "i1"))
        raw = bytearray(out.read_bytes())
        n_off = len(raw) - threats.N_THREAT_FEATURES * self.hidden - 4
        raw[n_off:n_off + 4] = (123).to_bytes(4, "little")
        out.write_bytes(bytes(raw))
        with self.assertRaises(AssertionError):
            self.tv3.load_quant(out)


if __name__ == "__main__":
    unittest.main()
