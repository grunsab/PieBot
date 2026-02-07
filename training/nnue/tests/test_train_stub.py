import json
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
