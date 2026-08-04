import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from training.nnue import train_torch
except Exception:  # pragma: no cover - exercised on environments without torch
    train_torch = None  # type: ignore[assignment]


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
    return train_torch.train_stub.objective_metadata(**values)


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
                "feature_set": train_torch.train_stub.FEATURE_SET,
                "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                "objective": _objective(),
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
            self.assertFalse(metrics["initial_checkpoint_weights_only"])
            self.assertFalse(checkpoint["initial_checkpoint_weights_only"])
            self.assertEqual("strict", checkpoint["initialized_from"]["mode"])
            self.assertFalse(
                checkpoint["initialized_from"]["objective_transition"]
            )

    def test_strict_warm_start_still_rejects_objective_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(
                root / "data",
                [{
                    "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                    "result": 1,
                    "value_cp": 75.0,
                    "teacher_depth": 6,
                }],
            )
            input_dim = train_torch.train_stub.HALFKP_DIM
            parent_path = root / "parent.json"
            parent_path.write_text(
                json.dumps({
                    "format": "piebot-halfkp-mse-v2-torch",
                    "feature_set": train_torch.train_stub.FEATURE_SET,
                    "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                    "objective": _objective(min_teacher_depth=0),
                    "input_dim": input_dim,
                    "hidden_dim": 1,
                    "w1": [0.0] * input_dim,
                    "b1": [0.5],
                    "w2": [1.5],
                    "b2": 2.0,
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "objective"):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    max_samples=1,
                    epochs=1,
                    val_split=0.0,
                    learning_rate=0.0,
                    hidden_dim=1,
                    loss_kind="huber",
                    min_teacher_depth=6,
                    teacher_sample_fraction=1.0,
                    initial_checkpoint=parent_path,
                    out_dir=root / "out",
                    device="cpu",
                )

    def test_weights_only_warm_start_loads_exact_weights_across_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(
                root / "data",
                [{
                    "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                    "result": 1,
                    "value_cp": 75.0,
                    "teacher_depth": 6,
                }],
            )
            input_dim = train_torch.train_stub.HALFKP_DIM
            parent_w1 = [0.0] * input_dim
            parent_w1[0] = 1.0
            parent_w1[-1] = -2.0
            source_objective = _objective(min_teacher_depth=0)
            requested_objective = _objective(
                loss_kind="huber",
                min_teacher_depth=6,
            )
            parent = {
                "format": "piebot-halfkp-mse-v2-torch",
                "feature_set": train_torch.train_stub.FEATURE_SET,
                "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                "objective": source_objective,
                "input_dim": input_dim,
                "hidden_dim": 1,
                "w1": parent_w1,
                "b1": [0.5],
                "w2": [1.5],
                "b2": 2.0,
            }
            parent_path = root / "parent.json"
            parent_path.write_text(json.dumps(parent), encoding="utf-8")

            metrics = train_torch.train_model(
                jsonl_dir=root / "data",
                max_samples=1,
                epochs=1,
                val_split=0.0,
                learning_rate=0.0,
                hidden_dim=1,
                loss_kind="huber",
                min_teacher_depth=6,
                teacher_sample_fraction=1.0,
                initial_checkpoint=parent_path,
                initial_checkpoint_weights_only=True,
                out_dir=root / "out",
                device="cpu",
            )

            checkpoint = json.loads((root / "out" / "checkpoint.json").read_text())
            self.assertEqual(parent["w1"], checkpoint["w1"])
            self.assertEqual(parent["b1"], checkpoint["b1"])
            self.assertEqual(parent["w2"], checkpoint["w2"])
            self.assertEqual(parent["b2"], checkpoint["b2"])
            self.assertEqual(requested_objective, checkpoint["objective"])
            self.assertTrue(checkpoint["initial_checkpoint_weights_only"])
            self.assertTrue(metrics["initial_checkpoint_weights_only"])
            for provenance in (
                checkpoint["initialized_from"],
                metrics["initialized_from"],
            ):
                self.assertEqual("weights-only", provenance["mode"])
                self.assertTrue(provenance["weights_only"])
                self.assertTrue(provenance["objective_transition"])
                self.assertTrue(provenance["weights_only_objective_transition"])
                self.assertEqual(source_objective, provenance["source_objective"])
                self.assertEqual(
                    requested_objective,
                    provenance["requested_objective"],
                )
            self.assertFalse(metrics["optimizer_state_restored"])

    def test_weights_only_warm_start_categorically_rejects_optimizer_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(
                root / "data",
                [{"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}],
            )
            parent_path = root / "parent.json"
            parent_path.write_text("{}", encoding="utf-8")
            optimizer_path = root / "optimizer.pt"
            optimizer_path.write_bytes(b"stale Adam moments")

            with self.assertRaisesRegex(
                ValueError,
                "weights.only.*optimizer|optimizer.*weights.only",
            ):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    max_samples=1,
                    epochs=1,
                    hidden_dim=1,
                    initial_checkpoint=parent_path,
                    initial_checkpoint_weights_only=True,
                    initial_optimizer_state=optimizer_path,
                    out_dir=root / "out",
                    device="cpu",
                )

    def test_weights_only_warm_start_rejects_non_finite_model_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dim = train_torch.train_stub.HALFKP_DIM
            parent_w1 = [0.0] * input_dim
            # Finite as a JSON/Python float, but not representable by the
            # float32 model tensor. Validation must happen after conversion.
            parent_w1[0] = 1e100
            parent_path = root / "parent.json"
            parent_path.write_text(
                json.dumps({
                    "format": "piebot-halfkp-mse-v2-torch",
                    "feature_set": train_torch.train_stub.FEATURE_SET,
                    "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                    "objective": _objective(),
                    "input_dim": input_dim,
                    "hidden_dim": 1,
                    "w1": parent_w1,
                    "b1": [0.0],
                    "w2": [0.0],
                    "b2": 0.0,
                }),
                encoding="utf-8",
            )
            model = train_torch.TorchNnue(input_dim=input_dim, hidden_dim=1)

            with self.assertRaisesRegex(ValueError, "finite model tensor"):
                train_torch._load_initial_checkpoint(
                    model,
                    parent_path,
                    input_dim=input_dim,
                    hidden_dim=1,
                    device=train_torch.torch.device("cpu"),
                    objective=_objective(min_teacher_depth=6),
                    weights_only=True,
                )

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
                        "feature_set": train_torch.train_stub.FEATURE_SET,
                        "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                        "objective": _objective(),
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

    def test_wdl_cp_diagnostics_use_probability_implied_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                "result": -1,
                "value_cp": 800.0,
                "teacher_depth": 6,
            }
            self._write_records(root / "train", [{**record, "split": "train"}])
            self._write_records(
                root / "validation",
                [{
                    **record,
                    "fen": "1k6/8/8/8/8/8/8/KQ6 w - - 0 1",
                    "split": "validation",
                }],
            )
            input_dim = train_torch.train_stub.HALFKP_DIM
            parent = {
                "format": "piebot-halfkp-mse-v2-torch",
                "feature_set": train_torch.train_stub.FEATURE_SET,
                "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                "objective": _objective(
                    loss_kind="wdl",
                    teacher_mix=0.5,
                    max_teacher_cp=1200.0,
                    min_teacher_depth=6,
                    wdl_scale_cp=200.0,
                ),
                "input_dim": input_dim,
                "hidden_dim": 1,
                "w1": [0.0] * input_dim,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            parent_path = root / "parent.json"
            parent_path.write_text(json.dumps(parent), encoding="utf-8")

            metrics = train_torch.train_model(
                jsonl_dir=root / "train",
                batch_size=1,
                max_samples=1,
                epochs=1,
                learning_rate=0.0,
                hidden_dim=1,
                target_cp=100.0,
                teacher_mix=0.5,
                max_teacher_cp=1200.0,
                loss_kind="wdl",
                wdl_scale_cp=200.0,
                min_teacher_depth=6,
                validation_jsonl_dir=root / "validation",
                max_validation_samples=1,
                initial_checkpoint=parent_path,
                out_dir=root / "out",
                device="cpu",
            )
            parsed = next(train_torch.train_stub.jsonl_to_training_samples([record]))
            probability = train_torch.train_stub._target_wdl_probability_for_record(
                parsed,
                target_cp=100.0,
                teacher_mix=0.5,
                max_teacher_cp=1200.0,
                wdl_scale_cp=200.0,
                min_teacher_depth=6,
            )
            implied_cp = train_torch.train_stub._wdl_probability_to_cp(
                probability,
                200.0,
            )
            linear_cp = train_torch.train_stub._target_cp_for_record(
                parsed,
                target_cp=100.0,
                teacher_mix=0.5,
                max_teacher_cp=1200.0,
                min_teacher_depth=6,
            )

            self.assertNotAlmostEqual(linear_cp, implied_cp, places=2)
            self.assertAlmostEqual(
                implied_cp * implied_cp,
                metrics["initial_val_cp_mse"],
                places=2,
            )
            self.assertEqual(0.0, metrics["initial_val_prediction_mean_abs"])
            self.assertEqual(0.0, metrics["initial_val_prediction_max_abs"])

    def test_warm_start_rejects_missing_or_mismatched_target_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(
                root / "data",
                [{"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}],
            )
            base = {
                "format": "piebot-halfkp-mse-v2-torch",
                "feature_set": train_torch.train_stub.FEATURE_SET,
                "input_dim": train_torch.train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": [0.0] * train_torch.train_stub.HALFKP_DIM,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            cases = [
                ("missing", base),
                (
                    "schema",
                    {
                        **base,
                        "target_schema": "hard-outcome-v1",
                        "objective": _objective(),
                    },
                ),
                (
                    "objective",
                    {
                        **base,
                        "target_schema": train_torch.train_stub.TARGET_SCHEMA,
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
                    train_torch.train_model(
                        jsonl_dir=root / "data",
                        max_samples=1,
                        epochs=1,
                        hidden_dim=1,
                        out_dir=root / f"out-{name}",
                        device="cpu",
                        initial_checkpoint=parent_path,
                    )

    def test_warm_start_rejects_missing_or_mismatched_feature_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(
                root / "data",
                [{"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}],
            )
            base = {
                "format": "piebot-halfkp-mse-v2-torch",
                "target_schema": train_torch.train_stub.TARGET_SCHEMA,
                "objective": _objective(),
                "input_dim": train_torch.train_stub.HALFKP_DIM,
                "hidden_dim": 1,
                "w1": [0.0] * train_torch.train_stub.HALFKP_DIM,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            cases = {
                "missing": base,
                "mismatched": {**base, "feature_set": "halfkp-legacy-v0"},
            }
            for name, parent in cases.items():
                parent_path = root / f"{name}.json"
                parent_path.write_text(json.dumps(parent), encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "feature_set",
                ):
                    train_torch.train_model(
                        jsonl_dir=root / "data",
                        max_samples=1,
                        epochs=1,
                        hidden_dim=1,
                        out_dir=root / f"out-{name}",
                        device="cpu",
                        initial_checkpoint=parent_path,
                    )

    def test_fixed_validation_can_require_depth_eligible_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fen = "k7/8/8/8/8/8/8/KQ6 w - - 0 1"
            self._write_records(
                root / "train",
                [{"fen": "1k6/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}],
            )
            self._write_records(root / "validation", [
                {"fen": fen, "result": 1},
                {
                    "fen": fen,
                    "result": 1,
                    "value_cp": 25.0,
                    "teacher_depth": 2,
                },
                {
                    "fen": fen,
                    "result": 1,
                    "value_cp": 50.0,
                    "teacher_depth": 6,
                },
            ])
            metrics = train_torch.train_model(
                jsonl_dir=root / "train",
                max_samples=1,
                epochs=1,
                learning_rate=0.0,
                hidden_dim=1,
                min_teacher_depth=6,
                validation_jsonl_dir=root / "validation",
                max_validation_samples=10,
                validation_require_teacher=True,
                out_dir=root / "out",
                device="cpu",
            )

            self.assertEqual(1, metrics["val_samples"])
            self.assertTrue(metrics["validation_require_teacher"])
            self.assertEqual(1, metrics["validation_records_with_teacher_value"])
            self.assertEqual(1, metrics["validation_records_with_raw_teacher_value"])

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

    def test_fixed_validation_rejects_overlapping_game_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_records(root / "train", [{
                "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                "result": 1,
                "run_id": "run-a",
                "game_id": "game-a",
                "ply": 0,
            }])
            self._write_records(root / "validation", [{
                "fen": "k7/8/8/8/8/8/8/KR6 b - - 1 1",
                "result": 1,
                "run_id": "run-a",
                "game_id": "game-a",
                "ply": 1,
            }])

            with self.assertRaisesRegex(ValueError, "game provenance"):
                train_torch.train_model(
                    jsonl_dir=root / "train",
                    max_samples=1,
                    epochs=1,
                    hidden_dim=1,
                    out_dir=root / "out",
                    device="cpu",
                    validation_jsonl_dir=root / "validation",
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
            first_optimizer_payload = train_torch._torch_load(first_optimizer)
            self.assertEqual("piebot-torch-adam-v3", first_optimizer_payload["format"])
            self.assertEqual(
                first["optimizer_state"]["model_parameters_sha256"],
                first_optimizer_payload["model_parameters_sha256"],
            )
            self.assertRegex(
                first_optimizer_payload["model_parameters_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertFalse(first["optimizer_state_restored"])

            second = train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=4,
                epochs=1,
                val_split=0.25,
                learning_rate=0.004,
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
            self.assertEqual(
                first_optimizer_payload["model_parameters_sha256"],
                second["optimizer_initialized_from"]["model_parameters_sha256"],
            )
            self.assertTrue((root / "second" / "optimizer.pt").is_file())
            resumed_payload = train_torch._torch_load(root / "second" / "optimizer.pt")
            resumed_group = resumed_payload["state_dict"]["param_groups"][0]
            self.assertEqual(0.004, resumed_group["lr"])
            self.assertEqual((0.9, 0.999), tuple(resumed_group["betas"]))
            self.assertEqual(1e-8, resumed_group["eps"])
            self.assertEqual(
                train_torch.train_stub.OBJECTIVE_SCHEMA,
                resumed_payload["objective"]["schema"],
            )

    def test_optimizer_rejects_adam_beta_or_epsilon_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)
            train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                learning_rate=0.001,
                hidden_dim=1,
                out_dir=root / "first",
                device="cpu",
            )

            cases = {
                "betas": {"adam_beta1": 0.8},
                "epsilon": {"adam_eps": 1e-6},
            }
            for name, overrides in cases.items():
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    f"Adam {name}",
                ):
                    train_torch.train_model(
                        jsonl_dir=root / "data",
                        batch_size=2,
                        max_samples=2,
                        epochs=1,
                        learning_rate=0.004,
                        hidden_dim=1,
                        initial_checkpoint=root / "first" / "checkpoint.json",
                        initial_optimizer_state=root / "first" / "optimizer.pt",
                        out_dir=root / f"second-{name}",
                        device="cpu",
                        **overrides,
                    )

    def test_optimizer_rejects_an_incompatible_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)
            train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                learning_rate=0.001,
                hidden_dim=1,
                loss_kind="wdl",
                out_dir=root / "first",
                device="cpu",
            )

            with self.assertRaisesRegex(ValueError, "objective"):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    batch_size=2,
                    max_samples=2,
                    epochs=1,
                    learning_rate=0.001,
                    hidden_dim=1,
                    loss_kind="huber",
                    initial_checkpoint=root / "first" / "checkpoint.json",
                    initial_optimizer_state=root / "first" / "optimizer.pt",
                    out_dir=root / "second",
                    device="cpu",
                )

    def test_optimizer_state_requires_an_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)
            train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                learning_rate=0.001,
                hidden_dim=1,
                out_dir=root / "first",
                device="cpu",
            )

            with self.assertRaisesRegex(ValueError, "optimizer.*requires.*checkpoint"):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    batch_size=2,
                    max_samples=2,
                    epochs=1,
                    learning_rate=0.001,
                    hidden_dim=1,
                    initial_optimizer_state=root / "first" / "optimizer.pt",
                    out_dir=root / "second",
                    device="cpu",
                )

    def test_optimizer_rejects_same_schema_checkpoint_with_modified_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)
            train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                learning_rate=0.001,
                hidden_dim=1,
                out_dir=root / "first",
                device="cpu",
            )
            checkpoint_path = root / "first" / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["b2"] = float(checkpoint["b2"]) + 1.0
            modified_checkpoint = root / "modified-checkpoint.json"
            modified_checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "optimizer.*model parameters"):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    batch_size=2,
                    max_samples=2,
                    epochs=1,
                    learning_rate=0.001,
                    hidden_dim=1,
                    initial_checkpoint=modified_checkpoint,
                    initial_optimizer_state=root / "first" / "optimizer.pt",
                    out_dir=root / "second",
                    device="cpu",
                )

    def test_checkpoint_rejects_different_optimizer_moments_for_same_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1},
                {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1},
            ]
            self._write_records(root / "data", records)
            train_torch.train_model(
                jsonl_dir=root / "data",
                batch_size=2,
                max_samples=2,
                epochs=1,
                learning_rate=0.001,
                hidden_dim=1,
                out_dir=root / "first",
                device="cpu",
            )
            original_optimizer = root / "first" / "optimizer.pt"
            payload = train_torch._torch_load(original_optimizer)
            changed = False
            for parameter_state in payload["state_dict"]["state"].values():
                for key, value in parameter_state.items():
                    if train_torch.torch.is_tensor(value) and value.numel() > 0:
                        replacement = value.clone()
                        replacement.view(-1)[0] += 1.0
                        parameter_state[key] = replacement
                        changed = True
                        break
                if changed:
                    break
            self.assertTrue(changed)
            different_optimizer = root / "different-optimizer.pt"
            train_torch._atomic_torch_save(payload, different_optimizer)
            self.assertNotEqual(
                hashlib.sha256(original_optimizer.read_bytes()).hexdigest(),
                hashlib.sha256(different_optimizer.read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "optimizer.*SHA"):
                train_torch.train_model(
                    jsonl_dir=root / "data",
                    batch_size=2,
                    max_samples=2,
                    epochs=1,
                    learning_rate=0.001,
                    hidden_dim=1,
                    initial_checkpoint=root / "first" / "checkpoint.json",
                    initial_optimizer_state=different_optimizer,
                    out_dir=root / "second",
                    device="cpu",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
