import math
from pathlib import Path
import unittest

import importlib
import sys as _sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "lc0_bin.py"
_spec = importlib.util.spec_from_file_location("training.nnue.lc0_bin", MODULE_PATH)
binmod = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = binmod
_spec.loader.exec_module(binmod)  # type: ignore[attr-defined]


def _make_chunk(**kwargs):
    version = kwargs.get("version", 6)
    input_format = kwargs.get("input_format", binmod.INPUT_CLASSICAL_112_PLANE)
    probabilities = kwargs.get("probabilities", [-1.0] * 1858)
    planes = kwargs.get("planes", [0] * 104)
    castling_fields = [
        kwargs.get("castling_us_ooo", 0),
        kwargs.get("castling_us_oo", 0),
        kwargs.get("castling_them_ooo", 0),
        kwargs.get("castling_them_oo", 0),
        kwargs.get("side_to_move_or_enpassant", 0),
        kwargs.get("rule50_count", 0),
        kwargs.get("invariance_info", 0),
        kwargs.get("dummy", 0),
    ]
    floats = [
        kwargs.get("root_q", 0.0),
        kwargs.get("best_q", 0.0),
        kwargs.get("root_d", 0.0),
        kwargs.get("best_d", 0.0),
        kwargs.get("root_m", 0.0),
        kwargs.get("best_m", 0.0),
        kwargs.get("plies_left", 0.0),
        kwargs.get("result_q", 0.0),
        kwargs.get("result_d", 0.0),
        kwargs.get("played_q", 0.0),
        kwargs.get("played_d", 0.0),
        kwargs.get("played_m", 0.0),
        kwargs.get("orig_q", 0.0),
        kwargs.get("orig_d", 0.0),
        kwargs.get("orig_m", 0.0),
    ]
    visits = kwargs.get("visits", 0)
    played_idx = kwargs.get("played_idx", 0)
    best_idx = kwargs.get("best_idx", 0)
    policy_kld = kwargs.get("policy_kld", 0.0)
    reserved = kwargs.get("reserved", 0)

    packed = (
        [version, input_format]
        + list(probabilities)
        + [p & binmod.FULL_MASK for p in planes]
        + castling_fields
        + floats
        + [visits, played_idx, best_idx, policy_kld, reserved]
    )
    return binmod.STRUCT_V6.pack(*packed)


class ParseRecordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        idx = binmod._load_policy_moves().index("a2a3")
        probs = [-1.0] * 1858
        probs[idx] = 0.5
        planes = [0] * 104
        planes[0] = binmod.reverse_bits_in_bytes(1 << 8)
        chunk = _make_chunk(probabilities=probs, planes=planes, best_idx=idx, visits=42)
        record = binmod.parse_v6_record(chunk)
        self.assertEqual(record.version, 6)
        self.assertEqual(record.input_format, binmod.INPUT_CLASSICAL_112_PLANE)
        self.assertTrue(math.isclose(record.probabilities[idx], 0.5))
        self.assertEqual(record.planes[0], binmod.reverse_bits_in_bytes(1 << 8))
        self.assertEqual(record.best_idx, idx)
        self.assertEqual(record.visits, 42)


class MoveDecodingTests(unittest.TestCase):
    def test_move_from_nn_index_identity(self) -> None:
        idx = binmod._load_policy_moves().index("a2a3")
        move = binmod.move_from_nn_index(idx, 0)
        self.assertEqual(move, "a2a3")

    def test_move_from_nn_index_transform(self) -> None:
        idx = binmod._load_policy_moves().index("a2a3")
        move = binmod.move_from_nn_index(idx, binmod.FLIP_TRANSFORM)
        self.assertEqual(move, "h2h3")


class PlaneDecodingTests(unittest.TestCase):
    def test_decode_planes_restores_mask(self) -> None:
        original_mask = 1 << 8
        stored = binmod.reverse_bits_in_bytes(original_mask)
        planes = [stored] + [0] * 103
        chunk = _make_chunk(planes=planes)
        record = binmod.parse_v6_record(chunk)
        decoded = binmod.decode_planes(record)
        self.assertEqual(decoded[0].mask, original_mask)


class PolicyExtractionTests(unittest.TestCase):
    def test_extract_top_moves(self) -> None:
        idx = binmod._load_policy_moves().index("a2a3")
        probs = [-1.0] * 1858
        probs[idx] = 0.25
        probs[0] = 0.5
        chunk = _make_chunk(probabilities=probs)
        record = binmod.parse_v6_record(chunk)
        top = binmod.extract_policy_top(record, top_n=2)
        self.assertEqual(top[0][0], binmod._load_policy_moves()[0])
        self.assertAlmostEqual(top[0][1], 0.5)
        self.assertEqual(top[1][0], "a2a3")
        self.assertAlmostEqual(top[1][1], 0.25)


class SampleTests(unittest.TestCase):
    def test_record_to_sample_fen(self) -> None:
        idx = binmod._load_policy_moves().index("a2a3")
        probs = [-1.0] * 1858
        probs[idx] = 0.5
        planes = [0] * 104
        planes[0] = binmod.reverse_bits_in_bytes(1 << 8)
        chunk = _make_chunk(probabilities=probs, planes=planes, best_idx=idx)
        record = binmod.parse_v6_record(chunk)
        sample = binmod.record_to_sample(record, top_policy=1)
        self.assertEqual(sample['fen'], '8/8/8/8/8/8/P7/8 w - - 0 1')
        self.assertEqual(sample['best_move'], 'a2a3')
        self.assertEqual(sample['policy_top'][0][0], 'a2a3')

    def test_canonical_white_sample_uses_absolute_white_perspective(self) -> None:
        idx = binmod._load_policy_moves().index("e2e4")
        probs = [-1.0] * 1858
        probs[idx] = 1.0
        planes = [0] * 104
        planes[0] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 1))
        planes[5] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 0))
        planes[11] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 7))
        chunk = _make_chunk(
            input_format=binmod.INPUT_112_WITH_CANONICALIZATION,
            probabilities=probs,
            planes=planes,
            result_q=1.0,
            best_q=0.75,
            root_q=0.5,
            best_idx=idx,
            played_idx=idx,
        )

        sample = binmod.record_to_sample(binmod.parse_v6_record(chunk), top_policy=1)

        self.assertEqual(sample['fen'], '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1')
        self.assertEqual(sample['fen_side_to_move'], 'w')
        self.assertEqual(sample['best_move'], 'e2e4')
        self.assertEqual(sample['played_move'], 'e2e4')
        self.assertEqual(sample['policy_top'], [('e2e4', 1.0)])
        self.assertEqual(sample['result'], 1)
        self.assertEqual(sample['result_q'], 1.0)
        self.assertEqual(sample['best_q'], 0.75)
        self.assertEqual(sample['root_q'], 0.5)

    def test_canonical_black_sample_restores_absolute_board_move_and_value(self) -> None:
        # LC0 stores both colors from the side-to-move perspective.  This is a
        # black e7-e5 position encoded canonically as an "our" pawn on e2 and
        # an e2-e4 policy move.  result_q is likewise side-to-move relative.
        idx = binmod._load_policy_moves().index("e2e4")
        probs = [-1.0] * 1858
        probs[idx] = 1.0
        planes = [0] * 104
        planes[0] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 1))
        planes[3] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(7, 0))
        planes[5] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 0))
        planes[6] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 4))
        planes[11] = binmod.reverse_bits_in_bytes(1 << binmod._bit_index(4, 7))
        chunk = _make_chunk(
            input_format=binmod.INPUT_112_WITH_CANONICALIZATION,
            probabilities=probs,
            planes=planes,
            castling_us_oo=1 << 7,
            side_to_move_or_enpassant=1 << 4,
            invariance_info=1 << 7,
            result_q=1.0,
            best_q=0.75,
            root_q=0.5,
            best_idx=idx,
            played_idx=idx,
        )

        sample = binmod.record_to_sample(binmod.parse_v6_record(chunk), top_policy=1)

        self.assertEqual(sample['fen'], '4k2r/4p3/8/8/4P3/8/8/4K3 b k e3 0 1')
        self.assertEqual(sample['fen_side_to_move'], 'b')
        self.assertEqual(sample['original_side_to_move'], 'b')
        self.assertEqual(sample['best_move'], 'e7e5')
        self.assertEqual(sample['played_move'], 'e7e5')
        self.assertEqual(sample['policy_top'], [('e7e5', 1.0)])
        self.assertEqual(sample['result'], -1)
        self.assertEqual(sample['result_q'], -1.0)
        self.assertEqual(sample['best_q'], -0.75)
        self.assertEqual(sample['root_q'], -0.5)
        self.assertEqual(sample['result_q_side_to_move'], 1.0)

    def test_unresolved_or_deleted_lc0_records_are_not_valid_outcomes(self) -> None:
        for flag in (1 << 4, 1 << 6):
            with self.subTest(flag=flag):
                chunk = _make_chunk(
                    input_format=binmod.INPUT_112_WITH_CANONICALIZATION,
                    invariance_info=flag,
                )
                sample = binmod.record_to_sample(binmod.parse_v6_record(chunk))
                self.assertFalse(sample['outcome_valid'])

        adjudicated = _make_chunk(
            input_format=binmod.INPUT_112_WITH_CANONICALIZATION,
            invariance_info=1 << 5,
        )
        sample = binmod.record_to_sample(binmod.parse_v6_record(adjudicated))
        self.assertTrue(sample['outcome_valid'])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
