import sys
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import uci_match  # noqa: E402


class GameOverTests(unittest.TestCase):
    def test_claimable_repetition_is_not_a_draw(self):
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
            board.push_uci(uci)
        # Black can claim (the move f6g8 would make a third occurrence) but
        # nothing has happened three times yet.
        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertIsNone(uci_match.game_over(board))
        board.push_uci("f6g8")
        outcome = uci_match.game_over(board)
        self.assertEqual(outcome.termination, chess.Termination.THREEFOLD_REPETITION)

    def test_checkmate_has_a_winner(self):
        board = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            board.push_uci(uci)
        self.assertEqual(uci_match.game_over(board).winner, chess.BLACK)


class StatsTests(unittest.TestCase):
    def test_even_match_is_zero_elo(self):
        s = uci_match.pentanomial_stats([1.0, 1.0, 2.0, 0.0])
        self.assertAlmostEqual(s["elo"], 0.0, places=6)
        self.assertEqual(s["pentanomial"], [1, 0, 2, 0, 1])
        self.assertLess(s["elo_95_ci"][0], 0.0)
        self.assertGreater(s["elo_95_ci"][1], 0.0)

    def test_options_are_typed(self):
        self.assertEqual(uci_match.parse_opts(["Hash=64", "UseNNUE=true", "F=a.nnue"]),
                         {"Hash": 64, "UseNNUE": True, "F": "a.nnue"})


if __name__ == "__main__":
    unittest.main()
