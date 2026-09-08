"""Exact ordered-feature compatibility and bounded encoder work."""
from pathlib import Path
import unittest
from unittest.mock import patch

from training.nnue import features_v2


def reference_parse(fen):
    """Frozen pre-optimization parser, including its permissive edge behavior."""
    parts = fen.split()
    board_part = parts[0]
    stm_white = len(parts) < 2 or parts[1] == "w"
    ranks = board_part.split("/")
    if len(ranks) != 8:
        raise ValueError(f"invalid FEN board part: {board_part}")
    pieces = []
    white_king = black_king = None
    for fen_rank, rank_str in enumerate(ranks):
        rank_idx = 7 - fen_rank
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            sq = rank_idx * 8 + file_idx
            pieces.append((ch, sq))
            if ch == "K":
                white_king = sq
            elif ch == "k":
                black_king = sq
            file_idx += 1
    if white_king is None or black_king is None:
        raise ValueError(f"FEN missing a king: {fen}")
    return pieces, white_king, black_king, stm_white


def reference_indices(fen):
    """Independent old traversal; intentionally retains its repeated scans."""
    pieces, wk, bk, stm = reference_parse(fen)
    outputs = []
    for perspective, king in ((True, wk), (False, bk)):
        indices = []
        flip = 0 if perspective else 56
        for white in (True, False):
            for plane, symbol in enumerate("PNBRQ"):
                selected = [(ch, square) for ch, square in pieces
                            if ch.upper() == symbol and ch.isupper() == white]
                for _, square in sorted(selected, key=lambda item: item[1]):
                    colored = plane if white == perspective else plane + 5
                    indices.append((((king ^ flip) * 10 + colored) * 64) + (square ^ flip))
        outputs.append(indices)
    return outputs[0], outputs[1], stm


def result_or_error(function, fen):
    try:
        return ("result", function(fen))
    except Exception as exc:
        return ("error", type(exc), str(exc))


class FeaturesV2ThroughputTests(unittest.TestCase):
    def test_piece_collection_is_scanned_once_for_both_perspectives(self):
        class CountedPieces(list):
            scans = 0

            def __iter__(self):
                self.scans += 1
                return super().__iter__()

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        raw, wk, bk, stm = reference_parse(fen)
        pieces = CountedPieces(raw)
        with patch.object(features_v2, "_parse_board", return_value=(pieces, wk, bk, stm)) as parser:
            actual = features_v2.active_indices(fen)
        self.assertEqual(actual, reference_indices(fen))
        parser.assert_called_once_with(fen)
        self.assertEqual(pieces.scans, 1, "classify pieces once before deriving both perspectives")

    def test_exact_color_plane_and_original_square_order_before_black_flip(self):
        board = "4k3/PP3p2/8/2N1R3/1P3n2/8/3r2q1/1Q2K3"
        ordered = [(True, 0, 25), (True, 0, 48), (True, 0, 49),
                   (True, 1, 34), (True, 3, 36), (True, 4, 1),
                   (False, 0, 53), (False, 1, 29), (False, 3, 11), (False, 4, 14)]
        expected = []
        for perspective, king, flip in ((True, 4, 0), (False, 60, 56)):
            expected.append([(((king ^ flip) * 10 + plane + (0 if white == perspective else 5)) * 64)
                             + (square ^ flip) for white, plane, square in ordered])
        # Original squares 25,48,49 become 33,8,9: sorting the black-view
        # squares would silently change EmbeddingBag's summation order.
        self.assertEqual([value % 64 for value in expected[1][:3]], [33, 8, 9])
        for stm in ("w", "b"):
            with self.subTest(stm=stm):
                fen = board + " " + stm + " - - 0 1"
                self.assertEqual(features_v2.active_indices(fen), (*expected, stm == "w"))
                self.assertEqual(features_v2.stm_ordered(fen),
                                 tuple(expected if stm == "w" else reversed(expected)))

    def test_committed_book_has_exact_ordered_bags(self):
        root = Path(__file__).resolve().parents[3]
        fens = [line for line in (root / "books/openings_v1.fen").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        self.assertEqual(len(fens), 1279)
        for index, fen in enumerate(fens):
            with self.subTest(index=index):
                self.assertEqual(features_v2.active_indices(fen), reference_indices(fen))

    def test_special_material_and_empty_bags_match_ordered_reference(self):
        boards = [
            "4k3/8/8/8/8/8/8/4K3",  # Bare kings, empty feature bags.
            "4k3/Q6Q/2N5/8/8/5n2/q6q/4K3",  # Promoted same-plane pieces.
            "r3k2r/8/8/8/8/8/8/R3K2R",  # Castling material.
            "4k3/8/8/3pP3/8/8/8/4K3",  # En-passant material.
        ]
        for board in boards:
            for stm in ("w", "b"):
                with self.subTest(board=board, stm=stm):
                    fen = board + " " + stm + " - - 0 1"
                    self.assertEqual(features_v2.active_indices(fen), reference_indices(fen))

    def test_parser_acceptance_and_errors_are_unchanged(self):
        # This encoder's parser is deliberately not the corpus validator.
        # Preserve its handling of malformed ranks, extra/unknown pieces,
        # missing fields, duplicate kings and unusual digit characters.
        fens = [
            "", "8/8/8/8/8/8/8", "8/8/8/8/8/8/8/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/8", "8/8/8/8/8/8/8/4K3",
            "4k3/8/8/8/8/8/8/4K3", "4k3/8/8/8/8/8/8/4K3 ? - - 0 1",
            "4k3/8/8/8/8/8/8/4K2 w", "4k3/9/8/8/8/8/8/4K3 w",
            "4k3/8/8//8/8/8/4K3 w", "4k3/8/8/8/8/8/8/04K3 w",
            "4k3/8/8/8/8/8/8/3XK3 w", "4k3/8/8/8/8/8/8/3ſK3 w",
            "4k2K/8/8/8/8/8/8/4K3 b", "4k3/8/8/8/8/8/8/²K5 w",
            "4k3/8/8/8/8/8/8/４K3 w", "4k3/8/8/8/8/8/8/3ßK3 w",
        ]
        for fen in fens:
            with self.subTest(fen=fen):
                self.assertEqual(result_or_error(features_v2._parse_board, fen),
                                 result_or_error(reference_parse, fen))
                self.assertEqual(result_or_error(features_v2.active_indices, fen),
                                 result_or_error(reference_indices, fen))


if __name__ == "__main__":
    unittest.main()
