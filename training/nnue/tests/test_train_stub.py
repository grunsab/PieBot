import json
import hashlib
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import train_stub
from training.nnue.dataloader import TrainingRecord


def _write_dataset(root: Path, n: int = 90) -> None:
    file_path = root / "shard_000000.jsonl"
    white_win = {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}
    draw = {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "result": 0}
    black_win = {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1}
    samples = [white_win, draw, black_win] * (n // 3)
    with file_path.open("w", encoding="utf-8") as handle:
        for rec in samples:
            handle.write(json.dumps(rec) + "\n")


def _objective(**overrides: object) -> dict:
    values = {
        "loss_kind": "mse",
        "target_cp": 100.0,
        "teacher_mix": 0.7,
        "max_teacher_cp": 1500.0,
        "outcome_decay": 1.0,
        "min_teacher_depth": 0,
        "huber_delta_cp": 100.0,
        "wdl_scale_cp": 400.0,
    }
    values.update(overrides)
    return train_stub.objective_metadata(**values)


class TrainStubTests(unittest.TestCase):
    def test_full_halfkp_schema_keys_both_piece_colors_to_both_kings(self) -> None:
        start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        active = train_stub._active_halfkp_indices(start)
        self.assertEqual(81_920, train_stub.HALFKP_DIM)
        self.assertEqual(60, len(active))
        self.assertEqual(60, len(set(active)))

        # Removing black's a7 pawn must remove one feature under each king.
        no_black_a7 = "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        removed = set(active) - set(train_stub._active_halfkp_indices(no_black_a7))
        self.assertEqual(2, len(removed))

    def test_target_blends_teacher_and_result_q(self) -> None:
        rec = TrainingRecord(
            fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            result=1,
            result_q=0.2,
            value_cp=300.0,
        )
        t = train_stub._target_cp_for_record(
            rec,
            target_cp=100.0,
            teacher_mix=0.75,
            max_teacher_cp=400.0,
        )
        self.assertAlmostEqual(t, 230.0, places=5)

    def test_target_uses_outcome_when_teacher_missing(self) -> None:
        rec = TrainingRecord(
            fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            result=-1,
            result_q=-0.4,
            value_cp=None,
        )
        t = train_stub._target_cp_for_record(
            rec,
            target_cp=100.0,
            teacher_mix=0.8,
            max_teacher_cp=400.0,
        )
        self.assertAlmostEqual(t, -40.0, places=5)

    def test_invalid_outcome_uses_unmixed_teacher_target(self) -> None:
        rec = TrainingRecord(
            fen="8/8/8/8/8/8/4K3/7k w - - 0 1",
            result=0,
            result_q=0.0,
            value_cp=75.0,
            outcome_valid=False,
        )
        target = train_stub._target_cp_for_record(
            rec,
            target_cp=100.0,
            teacher_mix=0.2,
            max_teacher_cp=1500.0,
        )
        self.assertAlmostEqual(75.0, target)

    def test_shallow_teacher_is_ignored_when_outcome_is_valid(self) -> None:
        rec = TrainingRecord(
            fen="8/8/8/8/8/8/4K3/7k w - - 0 1",
            result=-1,
            result_q=-0.4,
            value_cp=900.0,
            teacher_depth=2,
        )
        target = train_stub._target_cp_for_record(
            rec,
            target_cp=100.0,
            teacher_mix=0.8,
            max_teacher_cp=1200.0,
            min_teacher_depth=6,
        )
        self.assertAlmostEqual(-40.0, target)

    def test_wdl_target_blends_probabilities_and_respects_teacher_depth(self) -> None:
        rec = TrainingRecord(
            fen="8/8/8/8/8/8/4K3/7k w - - 0 1",
            result=1,
            result_q=1.0,
            value_cp=0.0,
            teacher_depth=6,
        )
        target = train_stub._target_wdl_probability_for_record(
            rec,
            teacher_mix=0.75,
            max_teacher_cp=1200.0,
            min_teacher_depth=6,
            wdl_scale_cp=400.0,
        )
        outcome_probability = train_stub._sigmoid(100.0 / 400.0)
        self.assertAlmostEqual(
            0.75 * 0.5 + 0.25 * outcome_probability,
            target,
        )

        rec.teacher_depth = 2
        shallow_target = train_stub._target_wdl_probability_for_record(
            rec,
            teacher_mix=0.75,
            max_teacher_cp=1200.0,
            min_teacher_depth=6,
            wdl_scale_cp=400.0,
        )
        self.assertAlmostEqual(outcome_probability, shallow_target)

    def test_wdl_decisive_outcomes_are_finite_cp_anchors(self) -> None:
        fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
        for result, expected_cp in ((1, 125.0), (0, 0.0), (-1, -125.0)):
            record = TrainingRecord(
                fen=fen,
                result=result,
                result_q=float(result),
            )
            probability = train_stub._target_wdl_probability_for_record(
                record,
                target_cp=125.0,
                teacher_mix=0.8,
                max_teacher_cp=1200.0,
                wdl_scale_cp=400.0,
                min_teacher_depth=6,
            )
            self.assertGreater(probability, 0.0)
            self.assertLess(probability, 1.0)
            self.assertAlmostEqual(
                expected_cp,
                train_stub._wdl_probability_to_cp(probability, 400.0),
                places=6,
            )

    def test_wdl_outcome_decay_is_applied_in_cp_space(self) -> None:
        record = TrainingRecord(
            fen="8/8/8/8/8/8/4K3/7k w - - 0 1",
            result=1,
            result_q=1.0,
            ply=2,
        )
        probability = train_stub._target_wdl_probability_for_record(
            record,
            target_cp=100.0,
            teacher_mix=0.8,
            max_teacher_cp=1200.0,
            wdl_scale_cp=400.0,
            outcome_decay=0.5,
        )
        self.assertAlmostEqual(
            25.0,
            train_stub._wdl_probability_to_cp(probability, 400.0),
            places=6,
        )

    def test_wdl_cp_gradient_matches_reported_bce_objective(self) -> None:
        pred_cp = 137.0
        target_probability = 0.73
        scale_cp = 400.0
        epsilon = 1e-3

        def loss(value: float) -> float:
            logit = value / scale_cp
            return (
                max(logit, 0.0)
                - logit * target_probability
                + math.log1p(math.exp(-abs(logit)))
            )

        finite_difference = (
            loss(pred_cp + epsilon) - loss(pred_cp - epsilon)
        ) / (2.0 * epsilon)
        analytical = train_stub._wdl_loss_gradient_cp(
            pred_cp,
            target_probability,
            scale_cp,
        )
        self.assertAlmostEqual(finite_difference, analytical, places=9)

    def test_iterate_samples_skips_invalid_outcome_without_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            records = [
                {
                    "id": "invalid-no-teacher",
                    "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                    "result": 0,
                    "outcome_valid": False,
                },
                {
                    "id": "invalid-with-teacher",
                    "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                    "result": 0,
                    "outcome_valid": False,
                    "value_cp": 25.0,
                },
                {
                    "id": "valid-outcome",
                    "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                    "result": 0,
                },
            ]
            with (data_dir / "shard_000000.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            loaded = list(train_stub.iterate_samples(data_dir, max_samples=0))

        self.assertEqual(
            ["invalid-with-teacher", "valid-outcome"],
            [record.raw["id"] for _features, record in loaded],
        )

    def test_iterate_samples_skips_invalid_outcome_with_shallow_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            records = [
                {
                    "id": "shallow-only",
                    "fen": fen,
                    "result": 0,
                    "outcome_valid": False,
                    "value_cp": 25.0,
                    "teacher_depth": 2,
                },
                {
                    "id": "deep-only",
                    "fen": fen,
                    "result": 0,
                    "outcome_valid": False,
                    "value_cp": 25.0,
                    "teacher_depth": 6,
                },
                {
                    "id": "outcome-fallback",
                    "fen": fen,
                    "result": 1,
                    "value_cp": -500.0,
                    "teacher_depth": 2,
                },
            ]
            with (data_dir / "shard.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            loaded = list(
                train_stub.iterate_samples(
                    data_dir,
                    max_samples=0,
                    min_teacher_depth=6,
                )
            )

        self.assertEqual(
            ["deep-only", "outcome-fallback"],
            [record.raw["id"] for _features, record in loaded],
        )

    def test_replay_sources_are_balanced_before_max_samples_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            for source_idx, count in ((0, 40), (1, 10)):
                path = data_dir / f"src{source_idx:02d}_shard000000.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record_idx in range(count):
                        handle.write(
                            json.dumps(
                                {
                                    "id": f"s{source_idx}-{record_idx}",
                                    "source": source_idx,
                                    "fen": fen,
                                    "result": 0,
                                }
                            )
                            + "\n"
                        )

            first = list(train_stub.iterate_samples(data_dir, max_samples=10, seed=9))
            second = list(train_stub.iterate_samples(data_dir, max_samples=10, seed=9))

        first_ids = [record.raw["id"] for _features, record in first]
        self.assertEqual(first_ids, [record.raw["id"] for _features, record in second])
        source_counts = {0: 0, 1: 0}
        for _features, record in first:
            source_counts[int(record.raw["source"])] += 1
        self.assertEqual({0: 5, 1: 5}, source_counts)

    def test_primary_source_gets_half_of_capped_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            for source_idx in range(7):
                path = data_dir / f"src{source_idx:02d}_shard000000.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record_idx in range(100):
                        handle.write(json.dumps({
                            "source": source_idx,
                            "id": f"{source_idx}-{record_idx}",
                            "fen": fen,
                            "result": 0,
                        }) + "\n")

            loaded = list(train_stub.iterate_samples(
                data_dir,
                max_samples=70,
                seed=13,
                primary_sample_fraction=0.5,
            ))

        counts = {source_idx: 0 for source_idx in range(7)}
        for _features, record in loaded:
            counts[int(record.raw["source"])] += 1
        self.assertEqual(35, counts[0])
        self.assertEqual(35, sum(counts[idx] for idx in range(1, 7)))
        self.assertLessEqual(max(counts.values()) if not counts else max(counts[idx] for idx in range(1, 7)), 6)
        self.assertGreaterEqual(min(counts[idx] for idx in range(1, 7)), 5)

    def test_teacher_fraction_is_independent_of_primary_source_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            # The primary source can supply only three deep-teacher samples,
            # while replay can supply the rest. Source and teacher quotas must
            # both remain satisfied.
            for source_idx, teacher_count in ((0, 3), (1, 10)):
                path = data_dir / f"src{source_idx:02d}_shard000000.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record_idx in range(20):
                        record = {
                            "id": f"{source_idx}-{record_idx}",
                            "source": source_idx,
                            "fen": fen,
                            "result": 1,
                        }
                        if record_idx < teacher_count:
                            record["value_cp"] = 25.0
                            record["teacher_depth"] = 6
                        handle.write(json.dumps(record) + "\n")

            first = list(train_stub.iterate_samples(
                data_dir,
                max_samples=20,
                seed=17,
                primary_sample_fraction=0.5,
                teacher_sample_fraction=0.5,
                min_teacher_depth=6,
            ))
            second = list(train_stub.iterate_samples(
                data_dir,
                max_samples=20,
                seed=17,
                primary_sample_fraction=0.5,
                teacher_sample_fraction=0.5,
                min_teacher_depth=6,
            ))

        self.assertEqual(
            [record.raw["id"] for _features, record in first],
            [record.raw["id"] for _features, record in second],
        )
        self.assertEqual(
            {0: 10, 1: 10},
            {
                source: sum(
                    int(record.raw["source"]) == source
                    for _features, record in first
                )
                for source in (0, 1)
            },
        )
        self.assertEqual(
            10,
            sum(
                train_stub._teacher_available(record, 6)
                for _features, record in first
            ),
        )

    def test_teacher_fraction_deterministically_oversamples_sparse_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            for source_idx, teacher_count in ((0, 2), (1, 3)):
                path = data_dir / f"src{source_idx:02d}_shard000000.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record_idx in range(20):
                        record = {
                            "id": f"{source_idx}-{record_idx}",
                            "source": source_idx,
                            "fen": fen,
                            "result": 1,
                        }
                        if record_idx < teacher_count:
                            record["value_cp"] = 25.0
                            record["teacher_depth"] = 6
                        handle.write(json.dumps(record) + "\n")

            kwargs = {
                "max_samples": 20,
                "seed": 23,
                "primary_sample_fraction": 0.5,
                "teacher_sample_fraction": 0.5,
                "min_teacher_depth": 6,
            }
            first = list(train_stub.iterate_samples(data_dir, **kwargs))
            second = list(train_stub.iterate_samples(data_dir, **kwargs))

        first_ids = [record.raw["id"] for _features, record in first]
        teacher_ids = [
            record.raw["id"]
            for _features, record in first
            if train_stub._teacher_available(record, 6)
        ]
        self.assertEqual(first_ids, [record.raw["id"] for _features, record in second])
        self.assertEqual(20, len(first))
        self.assertEqual(10, len(teacher_ids))
        self.assertLess(len(set(teacher_ids)), len(teacher_ids))
        self.assertEqual(
            {0: 10, 1: 10},
            {
                source: sum(
                    int(record.raw["source"]) == source
                    for _features, record in first
                )
                for source in (0, 1)
            },
        )

    def test_internal_split_keeps_oversampled_records_in_one_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            black_king_placements = [
                f"{rank}/8" for rank in (
                    "k7", "1k6", "2k5", "3k4", "4k3",
                    "5k2", "6k1", "7k",
                )
            ] + [
                "8/k7",
                "8/1k6",
            ]
            records = []
            for record_idx in range(10):
                record = {
                    "id": f"record-{record_idx}",
                    "fen": (
                        f"{black_king_placements[record_idx]}/8/8/8/8/4K3/8 "
                        "w - - 0 1"
                    ),
                    "result": 1,
                }
                if record_idx == 0:
                    record["value_cp"] = 25.0
                    record["teacher_depth"] = 6
                records.append(record)
            with (data_dir / "shard.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            metrics = train_stub.train_model(
                jsonl_dir=data_dir,
                batch_size=2,
                max_samples=10,
                epochs=1,
                val_split=0.2,
                learning_rate=0.0,
                hidden_dim=1,
                teacher_sample_fraction=0.5,
                min_teacher_depth=6,
                seed=1,
                out_dir=root / "out",
            )
            sampled_records = [
                record
                for _features, record in train_stub.iterate_samples(
                    data_dir,
                    10,
                    seed=1,
                    teacher_sample_fraction=0.5,
                    min_teacher_depth=6,
                )
            ]
            order = list(range(len(sampled_records)))
            random.Random(1).shuffle(order)
            sampled_records = [sampled_records[idx] for idx in order]
            train_indices, validation_indices = train_stub._internal_validation_partition(
                sampled_records,
                0.2,
            )

            self.assertEqual(0, metrics["internal_validation_record_overlap"])
            self.assertEqual(10, metrics["train_samples"] + metrics["val_samples"])
            self.assertGreater(metrics["val_samples"], 0)
            self.assertEqual(
                5,
                sum(
                    sampled_records[idx].raw["id"] == "record-0"
                    for idx in train_indices
                ),
            )
            self.assertNotIn(
                "record-0",
                [sampled_records[idx].raw["id"] for idx in validation_indices],
            )

    def test_internal_split_keeps_complete_games_in_one_partition(self) -> None:
        records = []
        fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
        for game_idx in range(6):
            for ply in range(4):
                records.append(
                    train_stub.TrainingRecord(
                        fen=fen,
                        result=0,
                        run_id="run-internal-split",
                        game_id=f"game-{game_idx}",
                        ply=ply,
                        raw={
                            "fen": fen,
                            "run_id": "run-internal-split",
                            "game_id": f"game-{game_idx}",
                            "ply": ply,
                        },
                    )
                )

        train_indices, validation_indices = train_stub._internal_validation_partition(
            records,
            0.25,
        )
        train_games = {records[idx].game_id for idx in train_indices}
        validation_games = {records[idx].game_id for idx in validation_indices}

        self.assertTrue(train_games)
        self.assertTrue(validation_games)
        self.assertFalse(train_games.intersection(validation_games))

    def test_internal_split_hash_is_order_invariant_and_stable_for_replay(self) -> None:
        identities = [
            f"game\0run-stable\0game-{game_idx:03}"
            for game_idx in range(100)
            for _ply in range(2)
        ]
        teacher_flags = [idx % 2 == 0 for idx in range(len(identities))]

        def roles(
            ordered_identities: list[str],
            ordered_teacher_flags: list[bool],
            *,
            validation_seed: int,
        ) -> dict[str, str]:
            train_indices, validation_indices = (
                train_stub._internal_validation_partition_from_metadata(
                    ordered_identities,
                    ordered_teacher_flags,
                    0.2,
                    validation_seed=validation_seed,
                )
            )
            result = {
                ordered_identities[idx]: "train" for idx in train_indices
            }
            result.update(
                {
                    ordered_identities[idx]: "validation"
                    for idx in validation_indices
                }
            )
            return result

        baseline = roles(identities, teacher_flags, validation_seed=71)
        order = list(reversed(range(len(identities))))
        reordered = roles(
            [identities[idx] for idx in order],
            [teacher_flags[idx] for idx in order],
            validation_seed=71,
        )
        expanded_identities = identities + [
            f"game\0run-new\0game-{game_idx:03}"
            for game_idx in range(100, 150)
        ]
        expanded = roles(
            expanded_identities,
            teacher_flags + [False] * 50,
            validation_seed=71,
        )
        other_seed = roles(identities, teacher_flags, validation_seed=72)

        self.assertEqual(baseline, reordered)
        self.assertEqual(
            baseline,
            {identity: expanded[identity] for identity in set(identities)},
        )
        self.assertNotEqual(baseline, other_seed)
        self.assertIn("train", baseline.values())
        self.assertIn("validation", baseline.values())

    def test_internal_split_tiny_fallback_keeps_both_sides_nonempty(self) -> None:
        identities = ["game\0run-tiny\0game-a", "game\0run-tiny\0game-b"]
        teacher_flags = [False, True]

        train_indices, validation_indices = (
            train_stub._internal_validation_partition_from_metadata(
                identities,
                teacher_flags,
                1e-12,
                validation_seed=91,
            )
        )

        self.assertEqual(1, len(train_indices))
        self.assertEqual(1, len(validation_indices))

    def test_compact_partition_matches_training_record_wrapper(self) -> None:
        records = [
            train_stub.TrainingRecord(
                fen="8/8/8/8/8/8/4K3/7k w - - 0 1",
                result=0,
                value_cp=25.0 if game_idx % 2 == 0 else None,
                teacher_depth=6 if game_idx % 2 == 0 else None,
                run_id="run-parity",
                game_id=f"game-{game_idx}",
                ply=ply,
            )
            for game_idx in range(20)
            for ply in range(2)
        ]
        wrapped = train_stub._internal_validation_partition(
            records,
            0.2,
            min_teacher_depth=6,
            validation_seed=123,
        )
        compact = train_stub._internal_validation_partition_from_metadata(
            [train_stub._validation_group_identity(record) for record in records],
            [train_stub._teacher_available(record, 6) for record in records],
            0.2,
            validation_seed=123,
        )

        self.assertEqual(wrapped, compact)

    def test_internal_split_preserves_teacher_mix_when_groups_allow_it(self) -> None:
        records = []
        fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
        for game_idx in range(40):
            for ply in range(2):
                teacher = game_idx % 2 == 0
                records.append(
                    train_stub.TrainingRecord(
                        fen=fen,
                        result=0,
                        value_cp=25.0 if teacher else None,
                        teacher_depth=6 if teacher else None,
                        run_id="run-stratified-split",
                        game_id=f"game-{game_idx}",
                        ply=ply,
                        raw={
                            "fen": fen,
                            "run_id": "run-stratified-split",
                            "game_id": f"game-{game_idx}",
                            "ply": ply,
                        },
                    )
                )

        train_indices, validation_indices = train_stub._internal_validation_partition(
            records,
            0.25,
            min_teacher_depth=6,
        )

        self.assertGreater(len(validation_indices), 0)
        self.assertAlmostEqual(
            0.5,
            sum(records[idx].teacher_depth == 6 for idx in validation_indices)
            / len(validation_indices),
            delta=0.1,
        )
        self.assertAlmostEqual(
            0.5,
            sum(records[idx].teacher_depth == 6 for idx in train_indices)
            / len(train_indices),
            delta=0.1,
        )

    def test_train_model_writes_metrics_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(data_dir, n=90)

            metrics = train_stub.train_model(
                jsonl_dir=data_dir,
                batch_size=9,
                max_samples=90,
                epochs=4,
                val_split=0.2,
                learning_rate=0.2,
                hidden_dim=4,
                target_cp=50.0,
                seed=7,
                out_dir=out_dir,
            )

            self.assertEqual(4, len(metrics["train_loss_history"]))
            self.assertEqual(4, len(metrics["val_loss_history"]))
            self.assertGreater(metrics["train_samples"], 0)
            self.assertGreater(metrics["val_samples"], 0)
            self.assertEqual(90, metrics["train_samples"] + metrics["val_samples"])
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "checkpoint.json").exists())

    def test_train_loss_improves_on_simple_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(data_dir, n=120)

            metrics = train_stub.train_model(
                jsonl_dir=data_dir,
                batch_size=12,
                max_samples=120,
                epochs=6,
                val_split=0.25,
                learning_rate=0.25,
                hidden_dim=4,
                target_cp=50.0,
                seed=11,
                out_dir=out_dir,
            )

            first = metrics["train_loss_history"][0]
            best = min(metrics["train_loss_history"])
            self.assertLessEqual(best, first, "training loss never improved")

    def test_wdl_loss_reports_cp_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=12)
            metrics = train_stub.train_model(
                jsonl_dir=data_dir,
                batch_size=4,
                max_samples=12,
                epochs=1,
                val_split=0.25,
                learning_rate=0.01,
                hidden_dim=1,
                loss_kind="wdl",
                wdl_scale_cp=400.0,
                out_dir=root / "out",
            )
            self.assertEqual("wdl", metrics["loss_kind"])
            self.assertEqual(1, len(metrics["train_cp_mse_history"]))
            self.assertEqual(1, len(metrics["train_prediction_mean_abs_history"]))
            self.assertEqual(1, len(metrics["train_prediction_max_abs_history"]))
            self.assertTrue(metrics["train_loss_history"][0] >= 0.0)

    def test_external_validation_is_reference_only_for_epoch_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train-data"
            validation_dir = root / "fixed-validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            _write_dataset(train_dir, n=12)
            validation_records = [
                {"fen": "1k6/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "1k6/8/8/8/8/8/8/K7 w - - 0 1", "result": 0},
                {"fen": "1kq5/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ] * 2
            with (validation_dir / "shard_000000.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for record in validation_records:
                    handle.write(json.dumps(record) + "\n")

            metrics = train_stub.train_model(
                jsonl_dir=train_dir,
                batch_size=4,
                max_samples=12,
                epochs=1,
                val_split=0.9,
                learning_rate=0.0,
                hidden_dim=1,
                validation_jsonl_dir=validation_dir,
                max_validation_samples=4,
                validation_seed=99,
                out_dir=root / "out",
            )

            self.assertEqual(12, metrics["train_samples"] + metrics["val_samples"])
            self.assertGreater(metrics["val_samples"], 0)
            self.assertEqual(4, metrics["reference_val_samples"])
            self.assertEqual(validation_dir.resolve().as_posix(), metrics["validation_jsonl_dir"])
            self.assertEqual(
                train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
                metrics["validation_sampling_schema"],
            )
            self.assertEqual(
                train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA,
                metrics["reference_validation_sampling_schema"],
            )
            self.assertEqual(
                train_stub.CHECKPOINT_SELECTION_SCHEMA,
                metrics["checkpoint_selection_schema"],
            )
            self.assertEqual(1, len(metrics["reference_val_loss_history"]))

    def test_reference_guard_allows_small_mismatch_but_blocks_large_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            reference_dir = root / "reference"
            train_dir.mkdir()
            reference_dir.mkdir()
            with (train_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
                for game_idx in range(4):
                    handle.write(json.dumps({
                        "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                        "result": 1,
                        "run_id": "run-guard",
                        "game_id": f"game-{game_idx}",
                        "ply": 0,
                    }) + "\n")
            (reference_dir / "reference.jsonl").write_text(
                json.dumps({
                    "fen": "1k6/8/8/8/8/8/8/KQ6 w - - 0 1",
                    "result": 1,
                    "value_cp": 500.0,
                    "teacher_depth": 6,
                }) + "\n",
                encoding="utf-8",
            )
            parent = {
                "format": "piebot-halfkp-mse-v2",
                "feature_set": train_stub.FEATURE_SET,
                "target_schema": train_stub.TARGET_SCHEMA,
                "objective": _objective(),
                "input_dim": train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": [0.0] * train_stub.HALFKP_DIM,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            parent_path = root / "parent.json"
            parent_path.write_text(json.dumps(parent), encoding="utf-8")

            def run_case(name: str, epoch_reference_loss: float) -> dict:
                evaluations = [
                    (0.9, 1.0, 0.5, 0.0, 0.0),
                    (1.0, 1.0, 0.5, 0.0, 0.0),
                    (1.0, 1.0, 0.5, 0.0, 0.0),
                    (0.8, 1.0, 0.5, 0.0, 0.0),
                    (0.8, 1.0, 0.5, 0.0, 0.0),
                    (epoch_reference_loss, 1.1, 0.4, 2.0, 3.0),
                ]
                with mock.patch(
                    "training.nnue.train_stub._eval_split",
                    side_effect=evaluations,
                ):
                    return train_stub.train_model(
                        jsonl_dir=train_dir,
                        batch_size=2,
                        max_samples=4,
                        epochs=1,
                        val_split=0.25,
                        learning_rate=0.0,
                        hidden_dim=1,
                        seed=17,
                        validation_seed=31,
                        out_dir=root / name,
                        initial_checkpoint=parent_path,
                        validation_jsonl_dir=reference_dir,
                        max_validation_samples=1,
                    )

            allowed = run_case("allowed", 1.002)
            blocked = run_case("blocked", 1.02)

            self.assertEqual(1, allowed["best_epoch"])
            self.assertEqual(1.002, allowed["best_reference_val_loss"])
            self.assertEqual(1.1, allowed["best_reference_val_cp_mse"])
            self.assertEqual(0.4, allowed["best_reference_val_acc"])
            self.assertEqual(2.0, allowed["best_reference_val_prediction_mean_abs"])
            self.assertEqual(3.0, allowed["best_reference_val_prediction_max_abs"])
            self.assertEqual(
                [True],
                allowed["reference_val_checkpoint_eligible_history"],
            )
            self.assertEqual(0, blocked["best_epoch"])
            self.assertEqual(1.0, blocked["best_reference_val_loss"])
            self.assertEqual(
                [False],
                blocked["reference_val_checkpoint_eligible_history"],
            )

    def test_external_validation_rejects_the_training_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=6)

            with self.assertRaisesRegex(ValueError, "validation.*training"):
                train_stub.train_model(
                    jsonl_dir=data_dir,
                    batch_size=2,
                    max_samples=6,
                    epochs=1,
                    hidden_dim=1,
                    validation_jsonl_dir=data_dir,
                    max_validation_samples=3,
                    out_dir=root / "out",
                )

    def test_external_validation_rejects_a_copied_training_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            _write_dataset(train_dir, n=6)
            (validation_dir / "copy.jsonl").write_bytes(
                (train_dir / "shard_000000.jsonl").read_bytes()
            )

            with self.assertRaisesRegex(ValueError, "copied shard"):
                train_stub.train_model(
                    jsonl_dir=train_dir,
                    max_samples=6,
                    epochs=1,
                    hidden_dim=1,
                    validation_jsonl_dir=validation_dir,
                    max_validation_samples=3,
                    out_dir=root / "out",
                )

    def test_external_validation_rejects_overlapping_game_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            train_record = {
                "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                "result": 1,
                "run_id": "run-a",
                "game_id": "game-7",
                "ply": 0,
            }
            validation_record = {
                "fen": "k7/8/8/8/8/8/8/KR6 b - - 1 1",
                "result": 1,
                "run_id": "run-a",
                "game_id": "game-7",
                "ply": 1,
            }
            (train_dir / "train.jsonl").write_text(
                json.dumps(train_record) + "\n", encoding="utf-8"
            )
            (validation_dir / "validation.jsonl").write_text(
                json.dumps(validation_record) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "game provenance"):
                train_stub.train_model(
                    jsonl_dir=train_dir,
                    max_samples=1,
                    epochs=1,
                    hidden_dim=1,
                    validation_jsonl_dir=validation_dir,
                    max_validation_samples=1,
                    out_dir=root / "out",
                )

    def test_external_validation_rejects_a_copied_legacy_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            shared = {
                "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                "result": 1,
                "value_cp": 42.0,
                "teacher_depth": 6,
            }
            (train_dir / "train.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(shared),
                        json.dumps(
                            {
                                "fen": "k7/8/8/8/8/8/8/KR6 b - - 1 1",
                                "result": 0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (validation_dir / "validation.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({**shared, "split": "validation-only-metadata"}),
                        json.dumps(
                            {
                                "fen": "8/k7/8/8/8/8/8/KQ6 w - - 2 2",
                                "result": -1,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "record identity"):
                train_stub.train_model(
                    jsonl_dir=train_dir,
                    max_samples=2,
                    epochs=1,
                    hidden_dim=1,
                    validation_jsonl_dir=validation_dir,
                    max_validation_samples=2,
                    out_dir=root / "out",
                )

    def test_external_validation_sampling_ignores_training_mix_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            _write_dataset(train_dir, n=3)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            for source_idx, has_teacher in ((0, True), (1, False)):
                path = validation_dir / f"src{source_idx:02d}_shard.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record_idx in range(8):
                        record = {
                            "id": f"validation-{source_idx}-{record_idx}",
                            "fen": fen,
                            "result": 1,
                        }
                        if has_teacher:
                            record["value_cp"] = 50.0
                            record["teacher_depth"] = 6
                        handle.write(json.dumps(record) + "\n")

            common = {
                "jsonl_dir": train_dir,
                "max_samples": 0,
                "epochs": 1,
                "learning_rate": 0.0,
                "hidden_dim": 1,
                "min_teacher_depth": 6,
                "validation_jsonl_dir": validation_dir,
                "max_validation_samples": 6,
                "validation_seed": 71,
            }
            primary_teacher = train_stub.train_model(
                **common,
                primary_sample_fraction=1.0,
                teacher_sample_fraction=1.0,
                out_dir=root / "out-primary-teacher",
            )
            replay_outcome = train_stub.train_model(
                **common,
                primary_sample_fraction=0.0,
                teacher_sample_fraction=0.0,
                out_dir=root / "out-replay-outcome",
            )

            self.assertEqual(
                primary_teacher["validation_records_with_teacher_value"],
                replay_outcome["validation_records_with_teacher_value"],
            )
            self.assertEqual(
                primary_teacher["validation_sample_sha256"],
                replay_outcome["validation_sample_sha256"],
            )

    def test_external_validation_can_require_depth_eligible_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train-data"
            validation_dir = root / "fixed-validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            _write_dataset(train_dir, n=3)
            fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
            records = [
                {"fen": fen, "result": 1},
                {
                    "fen": fen,
                    "result": 1,
                    "value_cp": 20.0,
                    "teacher_depth": 2,
                },
                {
                    "fen": fen,
                    "result": 1,
                    "value_cp": 30.0,
                    "teacher_depth": 6,
                },
            ]
            with (validation_dir / "shard.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            metrics = train_stub.train_model(
                jsonl_dir=train_dir,
                batch_size=3,
                max_samples=3,
                epochs=1,
                learning_rate=0.0,
                hidden_dim=1,
                min_teacher_depth=6,
                validation_jsonl_dir=validation_dir,
                max_validation_samples=10,
                validation_require_teacher=True,
                out_dir=root / "out",
            )

            self.assertEqual(3, metrics["train_samples"] + metrics["val_samples"])
            self.assertEqual(1, metrics["reference_val_samples"])
            self.assertTrue(metrics["validation_require_teacher"])
            self.assertEqual(1, metrics["validation_records_with_teacher_value"])
            self.assertEqual(1, metrics["validation_records_with_raw_teacher_value"])

    def test_checkpoint_records_target_and_objective_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=3)
            train_stub.train_model(
                jsonl_dir=data_dir,
                max_samples=3,
                epochs=1,
                learning_rate=0.0,
                hidden_dim=1,
                loss_kind="wdl",
                out_dir=root / "out",
            )
            checkpoint = json.loads((root / "out" / "checkpoint.json").read_text())
            metrics = json.loads((root / "out" / "metrics.json").read_text())

            self.assertEqual(train_stub.TARGET_SCHEMA, checkpoint["target_schema"])
            self.assertEqual(train_stub.OBJECTIVE_SCHEMA, checkpoint["objective"]["schema"])
            self.assertEqual(checkpoint["objective"], metrics["objective"])
            self.assertEqual(
                train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
                checkpoint["validation_sampling_schema"],
            )
            self.assertEqual(
                train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA,
                checkpoint["reference_validation_sampling_schema"],
            )
            self.assertEqual(
                train_stub.CHECKPOINT_SELECTION_SCHEMA,
                checkpoint["checkpoint_selection_schema"],
            )
            self.assertEqual(
                train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION,
                checkpoint["reference_validation_max_relative_loss_regression"],
            )

    def test_stub_warm_start_preserves_parent_when_zero_lr_cannot_improve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=3)
            w1 = [0.0] * train_stub.HALFKP_DIM
            w1[0] = 1.0
            w1[-1] = 2.0
            parent = {
                "format": "piebot-halfkp-mse-v2",
                "feature_set": train_stub.FEATURE_SET,
                "target_schema": train_stub.TARGET_SCHEMA,
                "objective": _objective(),
                "input_dim": train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": w1,
                "b1": [0.5],
                "w2": [1.5],
                "b2": 2.0,
            }
            parent_path = root / "parent.json"
            parent_path.write_text(json.dumps(parent), encoding="utf-8")
            parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()

            metrics = train_stub.train_model(
                jsonl_dir=data_dir,
                batch_size=2,
                max_samples=3,
                epochs=1,
                val_split=0.34,
                learning_rate=0.0,
                hidden_dim=1,
                seed=5,
                out_dir=root / "out",
                initial_checkpoint=parent_path,
            )

            checkpoint = json.loads((root / "out" / "checkpoint.json").read_text())
            self.assertEqual(parent["w1"], checkpoint["w1"])
            self.assertEqual(parent["b1"], checkpoint["b1"])
            self.assertEqual(parent["w2"], checkpoint["w2"])
            self.assertEqual(parent["b2"], checkpoint["b2"])
            self.assertEqual(0, checkpoint["best_epoch"])
            self.assertEqual(parent_sha, metrics["initialized_from"]["sha256"])
            self.assertFalse(metrics["optimizer_state_restored"])

    def test_stub_warm_start_rejects_missing_or_mismatched_target_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=3)
            base = {
                "format": "piebot-halfkp-mse-v2",
                "feature_set": train_stub.FEATURE_SET,
                "input_dim": train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": [0.0] * train_stub.HALFKP_DIM,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            cases = [
                ("missing", base),
                (
                    "schema",
                    {**base, "target_schema": "hard-outcome-v1", "objective": _objective()},
                ),
                (
                    "objective",
                    {
                        **base,
                        "target_schema": train_stub.TARGET_SCHEMA,
                        "objective": _objective(target_cp=200.0),
                    },
                ),
            ]
            for name, parent in cases:
                parent_path = root / f"{name}.json"
                parent_path.write_text(json.dumps(parent), encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "target_schema|objective",
                ):
                    train_stub.train_model(
                        jsonl_dir=data_dir,
                        max_samples=3,
                        epochs=1,
                        hidden_dim=1,
                        out_dir=root / f"out-{name}",
                        initial_checkpoint=parent_path,
                    )

    def test_stub_warm_start_requires_exact_feature_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=3)
            base = {
                "format": "piebot-halfkp-mse-v2",
                "target_schema": train_stub.TARGET_SCHEMA,
                "objective": _objective(),
                "input_dim": train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": [0.0] * train_stub.HALFKP_DIM,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            for name, feature_set in (("missing", None), ("wrong", "halfkp-v1")):
                parent = dict(base)
                if feature_set is not None:
                    parent["feature_set"] = feature_set
                parent_path = root / f"feature-{name}.json"
                parent_path.write_text(json.dumps(parent), encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "feature_set",
                ):
                    train_stub.train_model(
                        jsonl_dir=data_dir,
                        max_samples=3,
                        epochs=1,
                        hidden_dim=1,
                        out_dir=root / f"feature-out-{name}",
                        initial_checkpoint=parent_path,
                    )


class CheckpointSelectionTests(unittest.TestCase):
    """Selection must key off the low-variance teacher-labeled reference split.

    Regression coverage for the campaign_v6 stall: cycles 43/45/46 improved the
    reference loss but were discarded because the noisy primary validation loss
    rose by as little as 4e-5 (relative 6e-5), an order of magnitude below the
    between-cycle noise of that metric. Four consecutive no-op cycles resulted.
    """

    def test_reference_gain_with_primary_noise_is_selected(self) -> None:
        # Exact cycle_000046 numbers from the stalled run.
        self.assertTrue(
            train_stub.is_better_checkpoint(
                val_loss=0.6795620153623119,
                best_val_loss=0.6795337581211998,
                reference_val_loss=0.6332717661834855,
                best_reference_val_loss=0.6338223168296169,
                initial_reference_val_loss=0.6338223168296169,
            )
        )

    def test_reference_regression_is_rejected(self) -> None:
        # Exact cycle_000044 numbers: reference got worse, so this is a true reject.
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=0.6799593300151229,
                best_val_loss=0.6794232297487318,
                reference_val_loss=0.6352489354328845,
                best_reference_val_loss=0.6338223168296169,
                initial_reference_val_loss=0.6338223168296169,
            )
        )

    def test_primary_regression_beyond_tolerance_is_rejected(self) -> None:
        # Reference improves, but the primary split diverges far past the noise band.
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=0.90,
                best_val_loss=0.6795337581211998,
                reference_val_loss=0.6000,
                best_reference_val_loss=0.6338223168296169,
                initial_reference_val_loss=0.6338223168296169,
            )
        )

    def test_reference_absolute_regression_guard_still_applies(self) -> None:
        # Improving on best-so-far is not enough if it regressed past the
        # initial-loss envelope that REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION sets.
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=0.6790,
                best_val_loss=0.6795337581211998,
                reference_val_loss=0.7000,
                best_reference_val_loss=0.7200,
                initial_reference_val_loss=0.6338223168296169,
            )
        )

    def test_without_reference_split_falls_back_to_strict_primary(self) -> None:
        self.assertTrue(
            train_stub.is_better_checkpoint(
                val_loss=0.60,
                best_val_loss=0.61,
                reference_val_loss=None,
                best_reference_val_loss=None,
                initial_reference_val_loss=None,
            )
        )
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=0.61,
                best_val_loss=0.60,
                reference_val_loss=None,
                best_reference_val_loss=None,
                initial_reference_val_loss=None,
            )
        )

    def test_non_finite_losses_are_rejected(self) -> None:
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=float("nan"),
                best_val_loss=0.68,
                reference_val_loss=0.60,
                best_reference_val_loss=0.63,
                initial_reference_val_loss=0.63,
            )
        )
        self.assertFalse(
            train_stub.is_better_checkpoint(
                val_loss=0.68,
                best_val_loss=0.68,
                reference_val_loss=float("inf"),
                best_reference_val_loss=0.63,
                initial_reference_val_loss=0.63,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
