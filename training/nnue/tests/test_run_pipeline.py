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
            ckpt = {
                "format": "piebot-halfkp-mse-v2",
                "input_dim": 12,
                "hidden_dim": 4,
                "w1": [0.0] * (12 * 4),
                "b1": [0.0] * 4,
                "w2": [0.0] * 4,
                "b2": 0.0,
            }
            metrics = {"train_samples": 1, "val_samples": 0}
            (train_dir / "checkpoint.json").write_text(json.dumps(ckpt), encoding="utf-8")
            (train_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (out_dir / "nnue_dense.nnue").write_bytes(b"PIENNUE1dummy")
            (out_dir / "nnue_quant.nnue").write_bytes(b"PIENNQ01dummy")

            with mock.patch("training.nnue.run_pipeline._generate_selfplay_jsonl", side_effect=AssertionError("selfplay should be skipped")):
                with mock.patch("training.nnue.run_pipeline._relabel_jsonl", side_effect=AssertionError("relabel should be skipped")):
                    with mock.patch("training.nnue.train_stub.train_model", side_effect=AssertionError("train should be skipped")):
                        with mock.patch("training.nnue.run_pipeline.export_checkpoint_as_nnue", side_effect=AssertionError("export should be skipped")):
                            summary = run_pipeline.run_pipeline(
                                out_dir=out_dir,
                                selfplay_games=100,
                                teacher_relabel_depth=8,
                                resume=True,
                            )
            self.assertEqual(str(relabel_dir), summary["jsonl_dir"])
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
