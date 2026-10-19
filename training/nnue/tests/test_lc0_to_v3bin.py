import sys
import unittest
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc0_to_v3bin as v3  # noqa: E402


class EncodeTests(unittest.TestCase):
    def test_board_round_trips_through_occupancy_and_nibbles(self):
        for fen in [chess.STARTING_FEN, "8/8/4k3/8/8/3K4/8/7R b - - 7 40",
                    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"]:
            b = chess.Board(fen)
            occ, pcs = v3.encode_board(b)
            decoded = v3.decode_board(occ, pcs)
            want = {sq: (0 if p.color else 1) * 6 + p.piece_type - 1 for sq, p in b.piece_map().items()}
            self.assertEqual(decoded, want, fen)

    def test_record_is_32_bytes(self):
        self.assertEqual(v3.V3_DTYPE.itemsize, 32)
        self.assertEqual(np.zeros(3, dtype=v3.V3_DTYPE).tobytes().__len__(), 96)

    def test_holdout_is_deterministic_and_close_to_the_fraction(self):
        names = [f"training.{i}.gz" for i in range(20000)]
        picks = [v3.is_holdout(n, 0.01) for n in names]
        self.assertEqual(picks, [v3.is_holdout(n, 0.01) for n in names])
        self.assertTrue(120 < sum(picks) < 280)


if __name__ == "__main__":
    unittest.main()
