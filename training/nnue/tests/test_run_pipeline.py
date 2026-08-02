import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import run_pipeline


def _write_dataset(root: Path, n: int = 90) -> None:
    file_path = root / "shard_000000.jsonl"
    white_win = {"fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1", "result": 1}
    draw = {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "result": 0}
    black_win = {"fen": "kq6/8/8/8/8/8/8/K7 w - - 0 1", "result": -1}
    samples = [white_win, draw, black_win] * (n // 3)
    with file_path.open("w", encoding="utf-8") as handle:
        for rec in samples:
            handle.write(json.dumps(rec) + "\n")


class RunPipelineTests(unittest.TestCase):
    def test_resolve_trainer_backend_stub(self) -> None:
        self.assertEqual("stub", run_pipeline._resolve_trainer_backend("stub"))

    def test_resolve_trainer_backend_auto_without_torch_falls_back(self) -> None:
        original = run_pipeline.train_torch
        try:
            run_pipeline.train_torch = None
            self.assertEqual("stub", run_pipeline._resolve_trainer_backend("auto"))
        finally:
            run_pipeline.train_torch = original

    def test_build_selfplay_command_includes_jsonl_and_skip_bin(self) -> None:
        cmd = run_pipeline.build_selfplay_command(
            piebot_dir=Path("/tmp/repo/PieBot"),
            jsonl_out=Path("/tmp/out/jsonl"),
            games=12,
            max_plies=80,
            threads=2,
            parallel_games=8,
            depth=5,
            movetime_ms=50,
            seed=42,
            max_records_per_shard=1000,
            use_engine=True,
            openings=None,
            temperature_tau=1.0,
            temp_cp_scale=200.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            dirichlet_plies=8,
            temperature_moves=20,
            temperature_tau_final=0.1,
            nnue_quant_file=None,
            nnue_blend_percent=100,
        )
        self.assertIn("--jsonl-out", cmd)
        self.assertIn("--locked", cmd)
        self.assertIn("/tmp/out/jsonl", cmd)
        self.assertIn("--skip-bin", cmd)
        self.assertIn("--movetime-ms", cmd)
        self.assertIn("--parallel-games", cmd)
        self.assertIn("8", cmd)

    def test_build_relabel_command_uses_depth_and_period(self) -> None:
        cmd = run_pipeline.build_relabel_command(
            piebot_dir=Path("/tmp/repo/PieBot"),
            jsonl_in=Path("/tmp/in_jsonl"),
            jsonl_out=Path("/tmp/out_jsonl"),
            depth=8,
            every=4,
            threads=2,
            hash_mb=256,
            max_records=1000,
            nnue_quant_file=None,
            nnue_blend_percent=100,
        )
        self.assertIn("--bin", cmd)
        self.assertIn("--locked", cmd)
        self.assertIn("relabel_jsonl", cmd)
        self.assertIn("--input", cmd)
        self.assertIn("/tmp/in_jsonl", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("/tmp/out_jsonl", cmd)
        self.assertIn("--depth", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--every", cmd)
        self.assertIn("4", cmd)

    def test_build_selfplay_command_with_bootstrap_nnue(self) -> None:
        cmd = run_pipeline.build_selfplay_command(
            piebot_dir=Path("/tmp/repo/PieBot"),
            jsonl_out=Path("/tmp/out/jsonl"),
            games=12,
            max_plies=80,
            threads=2,
            parallel_games=6,
            depth=5,
            movetime_ms=50,
            seed=42,
            max_records_per_shard=1000,
            use_engine=True,
            openings=None,
            temperature_tau=1.0,
            temp_cp_scale=200.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            dirichlet_plies=8,
            temperature_moves=20,
            temperature_tau_final=0.1,
            nnue_quant_file=Path("/tmp/prev_cycle/nnue_quant.nnue"),
            nnue_blend_percent=90,
        )
        self.assertIn("--nnue-quant-file", cmd)
        self.assertIn("/tmp/prev_cycle/nnue_quant.nnue", cmd)
        self.assertIn("--nnue-blend-percent", cmd)
        self.assertIn("90", cmd)

    def test_build_relabel_command_with_bootstrap_nnue(self) -> None:
        cmd = run_pipeline.build_relabel_command(
            piebot_dir=Path("/tmp/repo/PieBot"),
            jsonl_in=Path("/tmp/in_jsonl"),
            jsonl_out=Path("/tmp/out_jsonl"),
            depth=8,
            every=4,
            threads=2,
            hash_mb=256,
            max_records=1000,
            nnue_quant_file=Path("/tmp/prev_cycle/nnue_quant.nnue"),
            nnue_blend_percent=95,
        )
        self.assertIn("--nnue-quant-file", cmd)
        self.assertIn("/tmp/prev_cycle/nnue_quant.nnue", cmd)
        self.assertIn("--nnue-blend-percent", cmd)
        self.assertIn("95", cmd)

    def test_bin_ingest_resets_stale_shards_before_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_dir = root / "jsonl"
            jsonl_dir.mkdir()
            stale = jsonl_dir / "shard_999999.jsonl"
            stale.write_text('{"stale":true}\n', encoding="utf-8")

            def _fake_process(_inputs, writer, _glob, _top_policy, _max_records):
                writer.write({"fen": "8/8/8/8/8/8/4K3/7k w - - 0 1", "result": 0})
                return 1

            with mock.patch(
                "training.nnue.run_pipeline.process_bins.process_inputs",
                side_effect=_fake_process,
            ):
                total = run_pipeline._ingest_bins_to_jsonl(
                    bin_inputs=[root / "input.bin"],
                    jsonl_dir=jsonl_dir,
                    bin_glob="*.bin*",
                    shard_size=10,
                    top_policy=1,
                    max_bin_records=0,
                )

            self.assertEqual(1, total)
            self.assertFalse(stale.exists())
            self.assertTrue(run_pipeline._jsonl_stage_is_complete(jsonl_dir, "bin_ingest"))

    def test_resume_skips_existing_selfplay_relabel_train_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            selfplay_dir = out_dir / "selfplay_jsonl"
            relabel_dir = out_dir / "jsonl_relabel"
            train_dir = out_dir / "train"
            selfplay_dir.mkdir(parents=True, exist_ok=True)
            relabel_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)

            sample = {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "result": 0, "best_move": "a1a2"}
            (selfplay_dir / "shard_000000.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            (relabel_dir / "shard_000000.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            run_pipeline.run_pipeline(
                out_dir=out_dir,
                jsonl_dir=relabel_dir,
                max_samples=1,
                epochs=1,
                hidden_dim=1,
                trainer_backend="stub",
            )
            piebot_dir = Path(run_pipeline.__file__).resolve().parents[2] / "PieBot"
            selfplay_provenance = run_pipeline._selfplay_stage_provenance(
                piebot_dir=piebot_dir,
                games=100,
                max_plies=100,
                threads=1,
                parallel_games=0,
                depth=4,
                movetime_ms=None,
                seed=42,
                max_records_per_shard=200_000,
                use_engine=True,
                openings=None,
                temperature_tau=1.0,
                temp_cp_scale=200.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                dirichlet_plies=8,
                temperature_moves=20,
                temperature_tau_final=0.1,
                nnue_quant_file=None,
                nnue_blend_percent=100,
            )
            run_pipeline._write_jsonl_stage_manifest(
                selfplay_dir,
                "selfplay",
                provenance=selfplay_provenance,
            )
            relabel_provenance = run_pipeline._relabel_stage_provenance(
                piebot_dir=piebot_dir,
                jsonl_in=selfplay_dir,
                depth=8,
                every=4,
                threads=1,
                hash_mb=64,
                max_records=0,
                nnue_quant_file=None,
                nnue_blend_percent=100,
            )
            run_pipeline._write_jsonl_stage_manifest(
                relabel_dir,
                "relabel",
                provenance=relabel_provenance,
            )

            with mock.patch("training.nnue.run_pipeline._generate_selfplay_jsonl", side_effect=AssertionError("selfplay should be skipped")):
                with mock.patch("training.nnue.run_pipeline._relabel_jsonl", side_effect=AssertionError("relabel should be skipped")):
                    with mock.patch("training.nnue.train_stub.train_model", side_effect=AssertionError("train should be skipped")):
                        with mock.patch("training.nnue.run_pipeline.export_checkpoint_as_nnue", side_effect=AssertionError("export should be skipped")):
                            summary = run_pipeline.run_pipeline(
                                out_dir=out_dir,
                                selfplay_games=100,
                                teacher_relabel_depth=8,
                                max_samples=1,
                                epochs=1,
                                hidden_dim=1,
                                trainer_backend="stub",
                                resume=True,
                            )
            self.assertEqual(str(relabel_dir), summary["jsonl_dir"])
            self.assertEqual(1, summary["ingested_records"])

    def test_jsonl_stage_manifest_detects_missing_or_changed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp) / "stage"
            stage_dir.mkdir(parents=True, exist_ok=True)
            shard = stage_dir / "shard_000000.jsonl"
            shard.write_text('{"fen":"8/8/8/8/8/8/4K3/7k w - - 0 1","result":0}\n', encoding="utf-8")

            self.assertFalse(run_pipeline._jsonl_stage_is_complete(stage_dir, "selfplay"))
            run_pipeline._write_jsonl_stage_manifest(stage_dir, "selfplay")
            self.assertTrue(run_pipeline._jsonl_stage_is_complete(stage_dir, "selfplay"))
            self.assertFalse(run_pipeline._jsonl_stage_is_complete(stage_dir, "relabel"))

            shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            self.assertFalse(run_pipeline._jsonl_stage_is_complete(stage_dir, "selfplay"))

    def test_resume_selfplay_binds_seed_model_content_and_manifest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            model = root / "teacher.nnue"
            model.write_bytes(b"model-a")
            calls = []

            def fake_selfplay(**kwargs):
                calls.append(kwargs)
                marker = Path(kwargs["nnue_quant_file"]).read_bytes().decode("ascii")
                sample = {
                    "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                    "result": 0,
                    "seed_marker": kwargs["seed"],
                    "model_marker": marker,
                }
                jsonl_out = Path(kwargs["jsonl_out"])
                (jsonl_out / "shard_000000.jsonl").write_text(
                    json.dumps(sample) + "\n",
                    encoding="utf-8",
                )
                return ["fake-selfplay"]

            kwargs = {
                "out_dir": out_dir,
                "selfplay_games": 1,
                "selfplay_max_plies": 1,
                "selfplay_depth": 1,
                "selfplay_seed": 7,
                "selfplay_nnue_quant_file": model,
                "max_samples": 1,
                "epochs": 1,
                "val_split": 0.0,
                "hidden_dim": 1,
                "trainer_backend": "stub",
            }
            with mock.patch(
                "training.nnue.run_pipeline._generate_selfplay_jsonl",
                side_effect=fake_selfplay,
            ):
                run_pipeline.run_pipeline(**kwargs)
                run_pipeline.run_pipeline(**kwargs, resume=True)
                self.assertEqual(1, len(calls))

                manifest_path = out_dir / "selfplay_jsonl" / ".piebot_stage_complete.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(2, manifest["version"])
                self.assertIn("provenance", manifest)

                # A v1 output-only marker from an older run must rebuild once.
                manifest["version"] = 1
                manifest.pop("provenance")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                run_pipeline.run_pipeline(**kwargs, resume=True)
                self.assertEqual(2, len(calls))

                changed_seed = dict(kwargs, selfplay_seed=8)
                run_pipeline.run_pipeline(**changed_seed, resume=True)
                self.assertEqual(3, len(calls))

                model.write_bytes(b"model-b")
                run_pipeline.run_pipeline(**changed_seed, resume=True)
                self.assertEqual(4, len(calls))

    def test_resume_relabel_binds_input_config_and_teacher_model_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            input_dir = root / "input"
            input_dir.mkdir()
            input_shard = input_dir / "shard_000000.jsonl"
            teacher = root / "teacher.nnue"
            teacher.write_bytes(b"teacher-a")

            def write_input(marker: str) -> None:
                sample = {
                    "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                    "result": 0,
                    "input_marker": marker,
                }
                replacement = input_dir / "replacement.tmp"
                replacement.write_text(json.dumps(sample) + "\n", encoding="utf-8")
                replacement.replace(input_shard)

            write_input("input-a")
            calls = []

            def fake_relabel(**kwargs):
                calls.append(kwargs)
                sample = json.loads(input_shard.read_text(encoding="utf-8"))
                sample["depth_marker"] = kwargs["depth"]
                sample["teacher_marker"] = Path(kwargs["nnue_quant_file"]).read_bytes().decode(
                    "ascii"
                )
                jsonl_out = Path(kwargs["jsonl_out"])
                (jsonl_out / "shard_000000.jsonl").write_text(
                    json.dumps(sample) + "\n",
                    encoding="utf-8",
                )
                return ["fake-relabel"]

            kwargs = {
                "out_dir": out_dir,
                "jsonl_dir": input_dir,
                "teacher_relabel_depth": 1,
                "teacher_relabel_every": 1,
                "teacher_relabel_threads": 1,
                "teacher_relabel_hash_mb": 16,
                "teacher_relabel_nnue_quant_file": teacher,
                "max_samples": 1,
                "epochs": 1,
                "val_split": 0.0,
                "hidden_dim": 1,
                "trainer_backend": "stub",
            }
            with mock.patch(
                "training.nnue.run_pipeline._relabel_jsonl",
                side_effect=fake_relabel,
            ):
                run_pipeline.run_pipeline(**kwargs)
                run_pipeline.run_pipeline(**kwargs, resume=True)
                self.assertEqual(1, len(calls))

                write_input("input-b")
                run_pipeline.run_pipeline(**kwargs, resume=True)
                self.assertEqual(2, len(calls))

                changed_depth = dict(kwargs, teacher_relabel_depth=2)
                run_pipeline.run_pipeline(**changed_depth, resume=True)
                self.assertEqual(3, len(calls))

                teacher.write_bytes(b"teacher-b")
                run_pipeline.run_pipeline(**changed_depth, resume=True)
                self.assertEqual(4, len(calls))

    def test_resume_rebuilds_unmarked_partial_selfplay_and_relabel_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            selfplay_dir = out_dir / "selfplay_jsonl"
            relabel_dir = out_dir / "jsonl_relabel"
            train_dir = out_dir / "train"
            selfplay_dir.mkdir(parents=True, exist_ok=True)
            relabel_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)
            (selfplay_dir / "shard_999999.jsonl").write_text("partial\n", encoding="utf-8")
            (relabel_dir / "shard_999999.jsonl").write_text("stale\n", encoding="utf-8")

            checkpoint = {
                "format": "piebot-halfkp-mse-v2",
                "input_dim": 12,
                "hidden_dim": 1,
                "w1": [0.0] * 12,
                "b1": [0.0],
                "w2": [0.0],
                "b2": 0.0,
            }
            (train_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            (train_dir / "metrics.json").write_text(
                json.dumps({"train_samples": 1, "val_samples": 0}),
                encoding="utf-8",
            )
            (out_dir / "nnue_dense.nnue").write_bytes(b"PIENNUE1dummy")
            (out_dir / "nnue_quant.nnue").write_bytes(b"PIENNQ01dummy")

            sample = {
                "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                "result": 0,
                "ply": 0,
            }
            calls = []

            def fake_selfplay(**kwargs):
                calls.append("selfplay")
                jsonl_out = Path(kwargs["jsonl_out"])
                self.assertFalse((jsonl_out / "shard_999999.jsonl").exists())
                (jsonl_out / "shard_000000.jsonl").write_text(
                    json.dumps(sample) + "\n",
                    encoding="utf-8",
                )
                return ["fake-selfplay"]

            def fake_relabel(**kwargs):
                calls.append("relabel")
                jsonl_out = Path(kwargs["jsonl_out"])
                self.assertFalse((jsonl_out / "shard_999999.jsonl").exists())
                relabeled = dict(sample)
                relabeled["value_cp"] = 0.0
                (jsonl_out / "shard_000000.jsonl").write_text(
                    json.dumps(relabeled) + "\n",
                    encoding="utf-8",
                )
                return ["fake-relabel"]

            with mock.patch(
                "training.nnue.run_pipeline._generate_selfplay_jsonl",
                side_effect=fake_selfplay,
            ):
                with mock.patch(
                    "training.nnue.run_pipeline._relabel_jsonl",
                    side_effect=fake_relabel,
                ):
                    summary = run_pipeline.run_pipeline(
                        out_dir=out_dir,
                        selfplay_games=1,
                        selfplay_max_plies=1,
                        teacher_relabel_depth=1,
                        resume=True,
                    )

            self.assertEqual(["selfplay", "relabel"], calls)
            self.assertTrue(run_pipeline._jsonl_stage_is_complete(selfplay_dir, "selfplay"))
            self.assertTrue(run_pipeline._jsonl_stage_is_complete(relabel_dir, "relabel"))
            self.assertEqual(1, summary["ingested_records"])

    def test_classifier_projection_uses_win_minus_loss(self) -> None:
        checkpoint = {
            "input_dim": 3,
            "num_classes": 3,
            "weights": [
                [1.0, 2.0, 3.0],   # loss
                [9.0, 9.0, 9.0],   # draw (ignored by projection)
                [4.0, 8.0, 10.0],  # win
            ],
            "bias": [-2.0, 0.0, 5.0],
        }
        w, b = run_pipeline.classifier_head_to_scalar(checkpoint, cp_scale=10.0)
        self.assertEqual([30.0, 60.0, 70.0], w)
        self.assertEqual(70.0, b)

    def test_pipeline_from_jsonl_writes_export_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(data_dir, n=120)

            summary = run_pipeline.run_pipeline(
                jsonl_dir=data_dir,
                out_dir=out_dir,
                batch_size=12,
                max_samples=120,
                epochs=4,
                val_split=0.2,
                learning_rate=0.2,
                hidden_dim=4,
                target_cp=50.0,
                seed=13,
                cp_scale=50.0,
            )

            dense_path = Path(summary["dense_path"])
            quant_path = Path(summary["quant_path"])
            metrics_path = Path(summary["metrics_path"])
            ckpt_path = Path(summary["checkpoint_path"])
            pipeline_summary = out_dir / "pipeline_summary.json"

            self.assertTrue(dense_path.exists())
            self.assertTrue(quant_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue(ckpt_path.exists())
            self.assertTrue(pipeline_summary.exists())
            self.assertTrue((out_dir / "train" / ".piebot_train_complete.json").exists())
            self.assertTrue((out_dir / ".piebot_export_complete.json").exists())

            self.assertEqual(b"PIENNUE1", dense_path.read_bytes()[:8])
            self.assertEqual(b"PIENNQ01", quant_path.read_bytes()[:8])

            loaded = json.loads(pipeline_summary.read_text(encoding="utf-8"))
            self.assertEqual(str(dense_path), loaded["dense_path"])
            self.assertGreater(loaded["metrics"]["train_samples"], 0)

    def test_pipeline_replay_jsonl_dirs_are_merged_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_dir = root / "current"
            replay_dir = root / "replay"
            out_dir = root / "out"
            current_dir.mkdir(parents=True, exist_ok=True)
            replay_dir.mkdir(parents=True, exist_ok=True)

            _write_dataset(current_dir, n=30)
            _write_dataset(replay_dir, n=30)

            summary = run_pipeline.run_pipeline(
                jsonl_dir=current_dir,
                replay_jsonl_dirs=[replay_dir],
                out_dir=out_dir,
                batch_size=10,
                max_samples=60,
                epochs=2,
                val_split=0.2,
                learning_rate=0.1,
                hidden_dim=4,
                target_cp=50.0,
                seed=17,
            )

            train_jsonl_dir = Path(summary["train_jsonl_dir"])
            self.assertTrue(train_jsonl_dir.exists())
            shards = sorted(train_jsonl_dir.glob("*.jsonl"))
            self.assertGreaterEqual(len(shards), 2)
            self.assertEqual(str(replay_dir), summary["replay_jsonl_dirs"][0])

    @unittest.skipUnless(
        run_pipeline.train_torch is not None
        and run_pipeline.train_torch.torch_available(),
        "torch is not installed",
    )
    def test_initial_checkpoint_is_forwarded_and_bound_to_resume_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_dataset(data_dir, n=12)

            parent_summary = run_pipeline.run_pipeline(
                jsonl_dir=data_dir,
                out_dir=root / "parent",
                batch_size=4,
                max_samples=12,
                epochs=1,
                val_split=0.25,
                learning_rate=0.01,
                hidden_dim=2,
                seed=3,
                trainer_backend="torch",
                trainer_device="cpu",
            )
            parent_path = Path(parent_summary["checkpoint_path"])
            child_kwargs = {
                "jsonl_dir": data_dir,
                "out_dir": root / "child",
                "batch_size": 4,
                "max_samples": 12,
                "epochs": 1,
                "val_split": 0.25,
                "learning_rate": 0.0,
                "hidden_dim": 2,
                "seed": 5,
                "trainer_backend": "torch",
                "trainer_device": "cpu",
                "initial_checkpoint": parent_path,
                "resume": True,
            }
            first = run_pipeline.run_pipeline(**child_kwargs)
            self.assertEqual(parent_path.resolve().as_posix(), first["initial_checkpoint"]["path"])
            child_checkpoint = json.loads(Path(first["checkpoint_path"]).read_text())
            self.assertEqual(
                first["initial_checkpoint"]["sha256"],
                child_checkpoint["initialized_from"]["sha256"],
            )

            with mock.patch(
                "training.nnue.train_torch.train_model",
                side_effect=AssertionError("matching warm-start provenance should resume"),
            ):
                resumed = run_pipeline.run_pipeline(**child_kwargs)
            self.assertEqual(first["checkpoint_path"], resumed["checkpoint_path"])

            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            parent["b2"] = float(parent["b2"]) + 1.0
            replacement = parent_path.with_suffix(".replacement")
            replacement.write_text(json.dumps(parent), encoding="utf-8")
            replacement.replace(parent_path)
            with mock.patch(
                "training.nnue.train_torch.train_model",
                wraps=run_pipeline.train_torch.train_model,
            ) as train:
                changed = run_pipeline.run_pipeline(**child_kwargs)
            self.assertEqual(1, train.call_count)
            self.assertNotEqual(
                first["initial_checkpoint"]["sha256"],
                changed["initial_checkpoint"]["sha256"],
            )

    def test_resume_rebuilds_corrupt_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(data_dir, n=12)
            kwargs = {
                "jsonl_dir": data_dir,
                "out_dir": out_dir,
                "batch_size": 4,
                "max_samples": 12,
                "epochs": 1,
                "val_split": 0.2,
                "hidden_dim": 1,
                "seed": 7,
                "trainer_backend": "stub",
            }
            run_pipeline.run_pipeline(**kwargs)
            (out_dir / "train" / "checkpoint.json").write_text("{}", encoding="utf-8")

            with mock.patch(
                "training.nnue.train_stub.train_model",
                wraps=run_pipeline.train_stub.train_model,
            ) as train:
                summary = run_pipeline.run_pipeline(**kwargs, resume=True)

            self.assertEqual(1, train.call_count)
            checkpoint = json.loads(Path(summary["checkpoint_path"]).read_text(encoding="utf-8"))
            self.assertEqual(1, checkpoint["hidden_dim"])
            self.assertIsNotNone(
                run_pipeline._validated_training_stage_manifest(out_dir / "train")
            )

    def test_resume_rebuilds_corrupt_export_artifact_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(data_dir, n=12)
            kwargs = {
                "jsonl_dir": data_dir,
                "out_dir": out_dir,
                "batch_size": 4,
                "max_samples": 12,
                "epochs": 1,
                "val_split": 0.2,
                "hidden_dim": 1,
                "seed": 11,
                "trainer_backend": "stub",
            }
            run_pipeline.run_pipeline(**kwargs)
            quant_path = out_dir / "nnue_quant.nnue"
            quant_path.write_bytes(b"PIENNQ01truncated")

            with mock.patch(
                "training.nnue.run_pipeline.export_checkpoint_as_nnue",
                wraps=run_pipeline.export_checkpoint_as_nnue,
            ) as export:
                with mock.patch(
                    "training.nnue.train_stub.train_model",
                    side_effect=AssertionError("valid training artifacts should be reused"),
                ):
                    summary = run_pipeline.run_pipeline(**kwargs, resume=True)

            self.assertEqual(1, export.call_count)
            self.assertGreater(Path(summary["quant_path"]).stat().st_size, 16)
            self.assertIsNotNone(
                run_pipeline._validated_export_stage_manifest(
                    out_dir,
                    checkpoint_path=out_dir / "train" / "checkpoint.json",
                    dense_path=out_dir / "nnue_dense.nnue",
                    quant_path=quant_path,
                )
            )

    def test_resume_rebuilds_merged_training_data_when_source_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            replay = root / "replay"
            out_dir = root / "out"
            primary.mkdir()
            replay.mkdir()
            (primary / "a.jsonl").write_text('{"id":"primary-v1"}\n', encoding="utf-8")
            replay_shard = replay / "b.jsonl"
            replay_shard.write_text('{"id":"replay-v1"}\n', encoding="utf-8")

            merged = run_pipeline._build_training_jsonl_dir(
                out_dir=out_dir,
                primary_jsonl_dir=primary,
                replay_jsonl_dirs=[replay],
                resume=False,
            )
            self.assertTrue((merged / ".piebot_merge_complete.json").exists())
            replay_copy = merged / "src01_shard000000.jsonl"
            self.assertIn("replay-v1", replay_copy.read_text(encoding="utf-8"))

            replacement = replay / "replacement.tmp"
            replacement.write_text('{"id":"replay-v2"}\n', encoding="utf-8")
            replacement.replace(replay_shard)
            merged_again = run_pipeline._build_training_jsonl_dir(
                out_dir=out_dir,
                primary_jsonl_dir=primary,
                replay_jsonl_dirs=[replay],
                resume=True,
            )

            self.assertEqual(merged, merged_again)
            self.assertIn("replay-v2", replay_copy.read_text(encoding="utf-8"))
            self.assertNotIn("replay-v1", replay_copy.read_text(encoding="utf-8"))

    def test_changed_replay_source_invalidates_resumed_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            replay = root / "replay"
            out_dir = root / "out"
            primary.mkdir()
            replay.mkdir()
            _write_dataset(primary, n=12)
            _write_dataset(replay, n=12)
            kwargs = {
                "jsonl_dir": primary,
                "replay_jsonl_dirs": [replay],
                "out_dir": out_dir,
                "batch_size": 4,
                "max_samples": 24,
                "epochs": 1,
                "hidden_dim": 1,
                "seed": 19,
                "trainer_backend": "stub",
            }
            run_pipeline.run_pipeline(**kwargs)

            replay_shard = replay / "shard_000000.jsonl"
            replacement = replay / "replacement.tmp"
            changed = {
                "fen": "k7/8/8/8/8/8/8/KQ6 w - - 0 1",
                "result": 1,
                "value_cp": 321.0,
            }
            replacement.write_text((json.dumps(changed) + "\n") * 12, encoding="utf-8")
            replacement.replace(replay_shard)

            with mock.patch(
                "training.nnue.train_stub.train_model",
                wraps=run_pipeline.train_stub.train_model,
            ) as train:
                run_pipeline.run_pipeline(**kwargs, resume=True)

            self.assertEqual(1, train.call_count)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
