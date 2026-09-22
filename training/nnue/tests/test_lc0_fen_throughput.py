"""Exact FEN and filtering contracts for sparse piece-mask enumeration."""

import itertools
import random
import unittest
from unittest import mock

import chess

from training.nnue.tests.test_lc0_bin import _make_chunk, binmod


ORDER = tuple(zip('PpNnBbRrQqKk', (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)))


def record(**kwargs):
    return binmod.parse_v6_record(_make_chunk(**kwargs))


def reference_board_part(rec, planes):
    """Legacy first-match semantics by square, serialized independently by chess.

    Coordinates restore the black view directly; no production bit-mirroring
    helper, sparse enumeration or FEN run-length builder is used.
    """
    black = (bool(rec.invariance_info & 128) if rec.input_format in (3, 4, 5, 132, 133)
             else bool(planes[108].mask))
    board = chess.BaseBoard(None)
    for square in range(64):
        source_square = square ^ 56 if black else square
        for symbol, index in ORDER:
            if black:
                index = index + 6 if index < 6 else index - 6
            if (planes[index].mask & ((1 << 64) - 1)) & (1 << source_square):
                board.set_piece_at(square, chess.Piece.from_symbol(symbol))
                break
    return board.board_fen()


def valid_record(**kwargs):
    masks = [0] * 104
    masks[0] = binmod.reverse_bits_in_bytes(1 << chess.E2)
    masks[5] = binmod.reverse_bits_in_bytes(1 << chess.E1)
    masks[11] = binmod.reverse_bits_in_bytes(1 << chess.E8)
    return record(planes=masks, best_q=.5, root_q=-.25, result_q=1., **kwargs)


class FenConstructionTests(unittest.TestCase):
    def assert_reference_board(self, rec, planes):
        before = [(plane.mask, plane.value) for plane in planes]
        result = binmod.build_fen_from_planes(rec, planes)
        self.assertEqual(result['fen'].split()[0], reference_board_part(rec, planes))
        self.assertEqual([(plane.mask, plane.value) for plane in planes], before)
        return result

    def test_board_construction_avoids_per_square_mask_scan(self):
        rec = valid_record()
        planes = binmod.decode_planes(rec)
        with mock.patch.object(binmod, '_piece_char', wraps=binmod._piece_char) as piece_scan, \
             mock.patch.object(binmod, '_mask_has', wraps=binmod._mask_has) as mask_scan:
            result = binmod.build_fen_from_planes(rec, planes)
        self.assertEqual(result['fen'], '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1')
        self.assertEqual(piece_scan.call_count, 0, 'FEN construction must enumerate occupied bits')
        self.assertEqual(mask_scan.call_count, 0, 'empty squares must not scan all twelve masks')

    def test_all_pairwise_overlap_precedence_for_both_colors(self):
        for black in (False, True):
            rec = record(side_to_move_or_enpassant=int(black))
            for first, second in itertools.combinations(range(12), 2):
                with self.subTest(black=black, first=first, second=second):
                    planes = [binmod.Plane() for _ in range(112)]
                    planes[108].mask = binmod.FULL_MASK if black else 0
                    planes[ORDER[first][1]].mask = 1 << chess.D4
                    planes[ORDER[second][1]].mask = 1 << chess.D4
                    self.assert_reference_board(rec, planes)

    def test_zero_full_negative_and_wide_masks_preserve_64_bit_contract(self):
        for black in (False, True):
            rec = record(side_to_move_or_enpassant=int(black))
            for mask in (0, -1, -(1 << 80), 1 << 100, (1 << 100) | (1 << 63),
                         binmod.FULL_MASK, (1 << 64) | 1):
                with self.subTest(black=black, mask=mask):
                    planes = [binmod.Plane() for _ in range(112)]
                    planes[108].mask = binmod.FULL_MASK if black else 0
                    planes[0].mask = mask
                    planes[6].mask = binmod.FULL_MASK
                    self.assert_reference_board(rec, planes)

    def test_all_formats_transforms_and_black_views_match_square_reference(self):
        rng = random.Random(2026090801)
        masks = [rng.getrandbits(64) for _ in range(104)]
        for fmt in (1, 2, 3, 4, 5, 132, 133):
            for transform in range(8):
                for black in (False, True):
                    with self.subTest(fmt=fmt, transform=transform, black=black):
                        rec = record(input_format=fmt, planes=masks,
                                     invariance_info=transform | (128 if black else 0),
                                     side_to_move_or_enpassant=int(black) if fmt < 3 else 0,
                                     rule50_count=99)
                        planes = binmod.decode_planes(rec)
                        result = self.assert_reference_board(rec, planes)
                        self.assertEqual(result['fen'].split()[1:],
                                         ['b' if black else 'w', '-', '-', '99', '1'])
                        self.assertEqual(result['black_to_move'], black)
                        self.assertEqual(result['castling'].to_fen(), '-')
                        self.assertEqual(result['en_passant'], '-')

    def test_castling_en_passant_and_rule50_fields_are_unchanged(self):
        for black in (False, True):
            for canonical in (False, True):
                # Opponent's double pawn move in the side-to-move frame.
                masks = [0] * 104
                masks[5] = binmod.reverse_bits_in_bytes(1 << chess.E1)
                masks[11] = binmod.reverse_bits_in_bytes(1 << chess.E8)
                masks[6] = binmod.reverse_bits_in_bytes(1 << chess.E5)
                masks[19] = binmod.reverse_bits_in_bytes(1 << chess.E7)
                rec = record(input_format=3 if canonical else 1, planes=masks,
                             side_to_move_or_enpassant=(1 << 4) if canonical else int(black),
                             invariance_info=128 if black else 0,
                             castling_us_ooo=1,
                             castling_us_oo=(1 << 7) if canonical else 1,
                             castling_them_ooo=1, castling_them_oo=(1 << 7) if canonical else 1,
                             rule50_count=255)
                result = self.assert_reference_board(rec, binmod.decode_planes(rec))
                self.assertEqual(result['castling'].to_fen(), 'KQkq')
                self.assertEqual(result['en_passant'], 'e3' if black else 'e6')
                self.assertEqual(result['fen'].split()[1:],
                                 ['b' if black else 'w', 'KQkq', 'e3' if black else 'e6', '255', '1'])

    def test_corpus_labels_and_malformed_filtering_match_reference_construction(self):
        from training.nnue import lc0_corpus
        cases = [valid_record(), valid_record(side_to_move_or_enpassant=1),
                 valid_record(invariance_info=16), valid_record(invariance_info=64),
                 valid_record(castling_us_oo=1), record()]
        overlap_valid = valid_record()
        overlap_valid.planes[1] = overlap_valid.planes[0]
        cases.append(overlap_valid)  # Pawn wins over overlapping knight; still accepted.
        overlap_invalid = valid_record()
        overlap_invalid.planes[0] |= overlap_invalid.planes[5]
        cases.append(overlap_invalid)  # Pawn hides king; remains invalid.
        nan = valid_record(); nan.best_q = float('nan'); cases.append(nan)
        original = binmod.build_fen_from_planes

        def reference(rec, planes):
            info = original(rec, planes)
            info['fen'] = reference_board_part(rec, planes) + ' ' + info['fen'].split(' ', 1)[1]
            return info

        actual = [lc0_corpus._sample(rec) for rec in cases]
        with mock.patch.object(lc0_corpus.lc0_bin, 'build_fen_from_planes', side_effect=reference):
            expected = [lc0_corpus._sample(rec) for rec in cases]
        self.assertEqual(actual, expected)
        self.assertEqual([item is not None for item in actual],
                         [True, True, True, False, False, False, True, False, False])
        self.assertEqual(actual[0]['best_q'], .5)
        self.assertEqual(actual[1]['best_q'], -.5)
        self.assertEqual(actual[1]['result_q'], -1.)
        self.assertFalse(actual[2]['outcome_valid'])
        frc = valid_record(input_format=2, castling_us_oo=1 << 6)
        with self.assertRaisesRegex(ValueError, 'Chess960'):
            lc0_corpus._sample(frc)


if __name__ == '__main__':
    unittest.main()
