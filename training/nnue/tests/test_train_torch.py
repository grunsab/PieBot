import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from training.nnue import train_torch
except Exception:  # pragma: no cover - exercised on environments without torch
    train_torch = None  # type: ignore[assignment]


@unittest.skipUnless(
    train_torch is not None and train_torch.torch_available(),
    "torch is not installed",
)
class TorchBatchPackingTests(unittest.TestCase):
    def test_all_empty_feature_bags_preserve_batch_shape(self) -> None:
        device = train_torch.torch.device("cpu")
        flat, offsets, targets = train_torch._pack_batch(
            [[], [], []],
            [1.0, 0.0, -1.0],
            device,
        )

        self.assertEqual(0, flat.numel())
        self.assertEqual([0, 0, 0], offsets.tolist())
        self.assertEqual((3,), tuple(targets.shape))

        model = train_torch.TorchNnue(input_dim=4, hidden_dim=2)
        predictions = model(flat, offsets)
        self.assertEqual((3,), tuple(predictions.shape))

    def test_warm_start_round_trips_parent_weights_with_zero_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "result": 0},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            with (data_dir / "shard_000000.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            input_dim = train_torch.train_stub.HALFKP_DIM
            hidden_dim = 2
            parent_w1 = [0.0] * (input_dim * hidden_dim)
            parent_w1[0] = 1.0
            parent_w1[input_dim - 1] = 2.0
            parent_w1[input_dim] = 3.0
            parent_w1[-1] = 4.0
            parent = {
                "format": "piebot-halfkp-mse-v2-torch",
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "w1": parent_w1,
                "b1": [0.5, 1.0],
                "w2": [1.5, -0.5],
                "b2": 2.0,
            }
            parent_path = root / "parent.json"
            parent_path.write_text(json.dumps(parent), encoding="utf-8")
            parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()

            result = train_torch.train_model(
                jsonl_dir=data_dir,
                batch_size=2,
                max_samples=3,
                epochs=1,
                val_split=0.34,
                learning_rate=0.0,
                hidden_dim=hidden_dim,
                seed=9,
                out_dir=root / "out",
                device="cpu",
                initial_checkpoint=parent_path,
            )

            checkpoint = json.loads((root / "out" / "checkpoint.json").read_text())
            metrics = json.loads((root / "out" / "metrics.json").read_text())
            self.assertEqual(parent["w1"], checkpoint["w1"])
            self.assertEqual(parent["b1"], checkpoint["b1"])
            self.assertEqual(parent["w2"], checkpoint["w2"])
            self.assertEqual(parent["b2"], checkpoint["b2"])
            self.assertEqual(0, checkpoint["best_epoch"])
            self.assertEqual(0, metrics["best_epoch"])
            self.assertEqual(parent_sha, checkpoint["initialized_from"]["sha256"])
            self.assertEqual(parent_path.resolve().as_posix(), result["initialized_from"]["path"])
            self.assertFalse(metrics["optimizer_state_restored"])

    def test_warm_start_rejects_incompatible_hidden_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "shard.jsonl").write_text(
                json.dumps({"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "result": 0})
                + "\n",
                encoding="utf-8",
            )
            input_dim = train_torch.train_stub.HALFKP_DIM
            parent_path = root / "parent.json"
            parent_path.write_text(
                json.dumps(
                    {
                        "format": "piebot-halfkp-mse-v2-torch",
                        "input_dim": input_dim,
                        "hidden_dim": 3,
                        "w1": [0.0] * (input_dim * 3),
                        "b1": [0.0] * 3,
                        "w2": [0.0] * 3,
                        "b2": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "hidden_dim"):
                train_torch.train_model(
                    jsonl_dir=data_dir,
                    max_samples=1,
                    epochs=1,
                    val_split=0.0,
                    hidden_dim=2,
                    out_dir=root / "out",
                    device="cpu",
                    initial_checkpoint=parent_path,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
