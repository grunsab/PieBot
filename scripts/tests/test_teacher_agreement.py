import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_agreement as ta


class ProbeLoadingTests(unittest.TestCase):
    def test_loads_fens_skipping_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.fen"
            probe.write_text(
                "# frozen probe v1\n"
                "\n"
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n",
                encoding="utf-8",
            )
            fens = ta.load_probe_fens(probe)
        self.assertEqual(2, len(fens))
        self.assertTrue(fens[0].startswith("rnbqkbnr"))

    def test_empty_probe_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.fen"
            probe.write_text("# nothing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ta.load_probe_fens(probe)

    def test_garbage_line_is_an_error_citing_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.fen"
            probe.write_text("not-a-fen\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                ta.load_probe_fens(probe)


class AgreementReportTests(unittest.TestCase):
    def test_full_agreement_reports_one(self) -> None:
        labels = [
            {"best_move": "e2e4", "score_cp": 30},
            {"best_move": "g1f3", "score_cp": -10},
        ]
        report = ta.agreement_report(labels, list(labels))
        self.assertEqual(1.0, report["best_move_agreement"])
        self.assertEqual(0.0, report["cp_delta_mean"])
        self.assertEqual(2, report["positions"])

    def test_partial_agreement_and_cp_deltas(self) -> None:
        a = [
            {"best_move": "e2e4", "score_cp": 30},
            {"best_move": "d2d4", "score_cp": 0},
            {"best_move": "c2c4", "score_cp": 100},
            {"best_move": "g1f3", "score_cp": -50},
        ]
        b = [
            {"best_move": "e2e4", "score_cp": 40},
            {"best_move": "g1f3", "score_cp": 20},
            {"best_move": "c2c4", "score_cp": 90},
            {"best_move": "b1c3", "score_cp": -80},
        ]
        report = ta.agreement_report(a, b)
        self.assertEqual(0.5, report["best_move_agreement"])
        self.assertAlmostEqual((10 + 20 + 10 + 30) / 4.0, report["cp_delta_mean"])
        self.assertEqual(30, report["cp_delta_max"])

    def test_mismatched_lengths_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            ta.agreement_report([{"best_move": "e2e4"}], [])


if __name__ == "__main__":
    unittest.main()
