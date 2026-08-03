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
    @staticmethod
    def _write_records(path: Path, records: list[dict]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "shard_000000.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

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
            self.assertEqual(train_torch.train_stub.FEATURE_SET, checkpoint["feature_set"])
            self.assertEqual(400.0, metrics["wdl_scale_cp"])
            self.assertEqual(20_260_802, metrics["validation_seed"])
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

    def test_objective_loss_supports_mse_huber_and_soft_wdl(self) -> None:
        pred_cp = train_torch.torch.tensor([0.0, 100.0])
        target_cp = train_torch.torch.tensor([0.0, 0.0])
        target_wdl = train_torch.torch.tensor([0.5, 0.75])

        mse = train_torch._objective_loss(
            pred_cp,
            target_cp,
            target_wdl,
            loss_kind="mse",
            huber_delta_cp=20.0,
            wdl_scale_cp=100.0,
        )
        huber = train_torch._objective_loss(
            pred_cp,
            target_cp,
            target_wdl,
            loss_kind="huber",
            huber_delta_cp=20.0,
            wdl_scale_cp=100.0,
        )
        wdl = train_torch._objective_loss(
            pred_cp,
            target_cp,
            target_wdl,
            loss_kind="wdl",
            huber_delta_cp=20.0,
            wdl_scale_cp=100.0,
        )
        expected_wdl = train_torch.torch.nn.functional.binary_cross_entropy_with_logits(
            pred_cp / 100.0,
            target_wdl,
        )

        self.assertAlmostEqual(5000.0, float(mse.item()), places=5)
        self.assertAlmostEqual(900.0, float(huber.item()), places=5)
        self.assertAlmostEqual(float(expected_wdl.item()), float(wdl.item()), places=7)

    def test_fixed_validation_is_separate_from_training_and_reports_cp_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "k7/8/8/8/8/8/8/KR6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
                {"fen": "kr6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            validation = [
                {"fen": "k7/8/8/8/8/8/8/KN6 w - - 0 1", "result": 1},
                {"fen": "kn6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "train", training)
            self._write_records(root / "validation", validation)

            metrics = train_torch.train_model(
                jsonl_dir=root / "train",
                batch_size=2,
                max_samples=4,
                epochs=1,
                val_split=0.5,
                learning_rate=0.0,
                hidden_dim=2,
                seed=17,
                out_dir=root / "out",
                device="cpu",
                loss_kind="wdl",
                wdl_scale_cp=200.0,
                min_teacher_depth=6,
                primary_sample_fraction=0.75,
                validation_jsonl_dir=root / "validation",
                max_validation_samples=2,
                validation_seed=91,
            )

            self.assertEqual(4, metrics["train_samples"])
            self.assertEqual(2, metrics["val_samples"])
            self.assertEqual(4, metrics["records_total"])
            self.assertEqual("wdl", metrics["loss_kind"])
            self.assertEqual(0.75, metrics["primary_sample_fraction"])
            self.assertEqual(6, metrics["min_teacher_depth"])
            self.assertTrue(metrics["fixed_validation"])
            self.assertEqual(
                (root / "validation").resolve().as_posix(),
                metrics["validation_jsonl_dir"],
            )
            self.assertEqual(1, len(metrics["val_loss_history"]))
            self.assertEqual(1, len(metrics["val_cp_mse_history"]))
            self.assertEqual(1, len(metrics["val_acc_history"]))

    def test_fixed_validation_rejects_the_training_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            self._write_records(
                data,
                [{"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}],
            )

            with self.assertRaisesRegex(ValueError, "validation.*training"):
                train_torch.train_model(
                    jsonl_dir=data,
                    max_samples=1,
                    epochs=1,
                    val_split=0.0,
                    hidden_dim=1,
                    out_dir=root / "out",
                    device="cpu",
                    validation_jsonl_dir=data,
                    max_validation_samples=1,
                )

    def test_teacher_metrics_distinguish_raw_from_depth_eligible_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(root / "data", [
                {
                    "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                    "result": 1,
                    "value_cp": 50.0,
                    "teacher_depth": 2,
                },
                {
                    "fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1",
                    "result": -1,
                    "value_cp": -50.0,
                    "teacher_depth": 6,
                },
            ])
            metrics = train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                val_split=0.0,
                learning_rate=0.0,
                hidden_dim=1,
                min_teacher_depth=6,
                out_dir=root / "out",
                device="cpu",
            )

            self.assertEqual(2, metrics["records_with_raw_teacher_value"])
            self.assertEqual(1, metrics["records_with_teacher_value"])

    def test_best_optimizer_state_round_trips_with_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "k7/8/8/8/8/8/8/KR6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
                {"fen": "kr6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)

            first = train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=4,
                epochs=1,
                val_split=0.25,
                learning_rate=0.001,
                hidden_dim=2,
                seed=31,
                out_dir=root / "first",
                device="cpu",
            )
            first_optimizer = root / "first" / "optimizer.pt"
            self.assertTrue(first_optimizer.is_file())
            self.assertEqual(
                hashlib.sha256(first_optimizer.read_bytes()).hexdigest(),
                first["optimizer_state"]["sha256"],
            )
            self.assertFalse(first["optimizer_state_restored"])

            second = train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=4,
                epochs=1,
                val_split=0.25,
                learning_rate=0.004,
                adam_beta1=0.8,
                adam_beta2=0.95,
                adam_eps=1e-6,
                hidden_dim=2,
                seed=31,
                out_dir=root / "second",
                device="cpu",
                initial_checkpoint=root / "first" / "checkpoint.json",
                initial_optimizer_state=first_optimizer,
            )

            self.assertTrue(second["optimizer_state_restored"])
            self.assertEqual(
                first_optimizer.resolve().as_posix(),
                second["optimizer_initialized_from"]["path"],
            )
            self.assertEqual(
                second["optimizer_initialized_from"],
                second["initialized_optimizer_state"],
            )
            self.assertEqual(
                hashlib.sha256(first_optimizer.read_bytes()).hexdigest(),
                second["optimizer_initialized_from"]["sha256"],
            )
            self.assertTrue((root / "second" / "optimizer.pt").is_file())
            resumed_payload = train_torch._torch_load(root / "second" / "optimizer.pt")
            resumed_group = resumed_payload["state_dict"]["param_groups"][0]
            self.assertEqual(0.004, resumed_group["lr"])
            self.assertEqual((0.8, 0.95), tuple(resumed_group["betas"]))
            self.assertEqual(1e-6, resumed_group["eps"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
