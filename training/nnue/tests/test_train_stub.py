import json
import hashlib
import tempfile
import unittest
from pathlib import Path

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
        self.assertAlmostEqual(0.625, target)

        rec.teacher_depth = 2
        shallow_target = train_stub._target_wdl_probability_for_record(
            rec,
            teacher_mix=0.75,
            max_teacher_cp=1200.0,
            min_teacher_depth=6,
            wdl_scale_cp=400.0,
        )
        self.assertAlmostEqual(1.0, shallow_target)

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
            self.assertTrue(metrics["train_loss_history"][0] >= 0.0)

    def test_external_validation_is_fixed_and_not_part_of_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train-data"
            validation_dir = root / "fixed-validation"
            train_dir.mkdir()
            validation_dir.mkdir()
            _write_dataset(train_dir, n=12)
            _write_dataset(validation_dir, n=6)

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

            self.assertEqual(12, metrics["train_samples"])
            self.assertEqual(4, metrics["val_samples"])
            self.assertEqual(validation_dir.resolve().as_posix(), metrics["validation_jsonl_dir"])

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
