#!/usr/bin/env python3
"""Tests for checkpoint JSON serialization, deserialization, and in-memory caching."""

import copy
import hashlib
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from training.nnue import train_torch, features_v2, train_stub
except ImportError:
    import train_torch
    import features_v2
    import train_stub


@unittest.skipUnless(train_torch.torch_available(), "PyTorch required")
class CheckpointOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        train_torch.clear_checkpoint_cache()

    def tearDown(self) -> None:
        train_torch.clear_checkpoint_cache()

    def _sample_checkpoint(self, input_dim: int = 16, hidden_dim: int = 4) -> dict:
        w1 = [float(i) * 0.01 for i in range(input_dim * hidden_dim)]
        b1 = [0.1 * (i + 1) for i in range(hidden_dim)]
        w2 = [0.05 * (i + 1) for i in range(2 * hidden_dim)]
        b2 = 0.5
        return {
            "format": "piebot-halfkp-dp-screlu-v1-torch",
            "feature_set": features_v2.FEATURE_SET_V2,
            "target_schema": train_stub.TARGET_SCHEMA,
            "objective": {
                "schema": "nnue-objective-v1",
                "loss_kind": "wdl",
                "target_cp": 250.0,
                "teacher_mix": 0.8,
                "max_teacher_cp": 1000.0,
                "outcome_decay": 1.0,
                "min_teacher_depth": 0,
                "wdl_scale_cp": 400.0,
                "target_schema": train_stub.TARGET_SCHEMA,
            },
            "arch": "v2",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "quant_qa": 255,
            "quant_qb": 64,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        }

    def test_load_initial_checkpoint_cold_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cp_data = self._sample_checkpoint()
            cp_file = tmp_path / "checkpoint.json"
            cp_file.write_text(json.dumps(cp_data), encoding="utf-8")

            dev = torch.device("cpu")
            model1 = train_torch.TorchNnueV2(
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                wdl_scale=400.0,
            ).to(dev)

            # Cold load
            self.assertEqual(0, len(train_torch._INITIAL_CHECKPOINT_CACHE))
            res1 = train_torch._load_initial_checkpoint(
                model1,
                cp_file,
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                device=dev,
                objective=cp_data["objective"],
                arch="v2",
            )
            self.assertEqual(1, len(train_torch._INITIAL_CHECKPOINT_CACHE))
            self.assertEqual(res1["sha256"], hashlib.sha256(cp_file.read_bytes()).hexdigest())

            # Warm load (from cache)
            model2 = train_torch.TorchNnueV2(
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                wdl_scale=400.0,
            ).to(dev)
            t0 = time.perf_counter()
            res2 = train_torch._load_initial_checkpoint(
                model2,
                cp_file,
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                device=dev,
                objective=cp_data["objective"],
                arch="v2",
            )
            t1 = time.perf_counter()
            self.assertLess(t1 - t0, 0.05, "Cached load must take under 50ms")
            self.assertEqual(res1["sha256"], res2["sha256"])

            # Verify identical weights
            self.assertTrue(torch.allclose(model1.embed.weight, model2.embed.weight))
            self.assertTrue(torch.allclose(model1.b1, model2.b1))
            self.assertTrue(torch.allclose(model1.out.weight, model2.out.weight))
            self.assertTrue(torch.allclose(model1.out.bias, model2.out.bias))

    def test_cache_invalidation_on_file_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cp_data = self._sample_checkpoint()
            cp_file = tmp_path / "checkpoint.json"
            cp_file.write_text(json.dumps(cp_data), encoding="utf-8")

            dev = torch.device("cpu")
            model = train_torch.TorchNnueV2(
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                wdl_scale=400.0,
            ).to(dev)

            res1 = train_torch._load_initial_checkpoint(
                model,
                cp_file,
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                device=dev,
                objective=cp_data["objective"],
                arch="v2",
            )

            # Modify file
            time.sleep(0.01)  # Ensure mtime updates
            cp_data["b2"] = 99.0
            cp_file.write_text(json.dumps(cp_data), encoding="utf-8")

            res2 = train_torch._load_initial_checkpoint(
                model,
                cp_file,
                input_dim=cp_data["input_dim"],
                hidden_dim=cp_data["hidden_dim"],
                device=dev,
                objective=cp_data["objective"],
                arch="v2",
            )

            self.assertNotEqual(res1["sha256"], res2["sha256"])
            self.assertAlmostEqual(99.0, model.out.bias.item(), places=4)

    def test_train_model_populates_cache_and_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_file = root / "train" / "train.jsonl"
            train_file.parent.mkdir(parents=True)
            rows = [
                json.dumps({
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "q": 0.0, "best_q": 0.0, "result": 0, "game_id": "g0", "run_id": "r0"
                }),
                json.dumps({
                    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                    "q": 0.05, "best_q": 0.05, "result": 1, "game_id": "g0", "run_id": "r0"
                }),
            ]
            train_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
            out_dir = root / "out"

            train_torch.clear_checkpoint_cache()
            self.assertEqual(0, len(train_torch._INITIAL_CHECKPOINT_CACHE))

            metrics = train_torch.train_model(
                jsonl_dir=train_file.parent,
                batch_size=2,
                max_samples=2,
                epochs=1,
                val_split=0.0,
                learning_rate=0.001,
                hidden_dim=16,
                loss_kind="wdl",
                wdl_scale_cp=400.0,
                cp_loss_weight=0.0,
                outcome_decay=1.0,
                min_teacher_depth=0,
                teacher_mix=0.8,
                arch="v2",
                target_mode="lc0-q-outcome",
                checkpoint_selection="latest",
                out_dir=out_dir,
                device="cpu",
            )

            # Checkpoint cache was populated
            self.assertEqual(1, len(train_torch._INITIAL_CHECKPOINT_CACHE))

            # Checkpoint file on disk is valid JSON
            cp_file = out_dir / "checkpoint.json"
            self.assertTrue(cp_file.is_file())
            loaded = json.loads(cp_file.read_text(encoding="utf-8"))
            self.assertEqual("piebot-halfkp-dp-screlu-v1-torch", loaded["format"])
            self.assertEqual(16, loaded["hidden_dim"])
            self.assertIsInstance(loaded["w1"], list)
            self.assertIsInstance(loaded["b1"], list)
            self.assertIsInstance(loaded["w2"], list)
            self.assertIsInstance(loaded["b2"], float)

            # Loading initial checkpoint immediately from this path hits the cache
            dev = torch.device("cpu")
            next_model = train_torch.TorchNnueV2(
                input_dim=loaded["input_dim"],
                hidden_dim=loaded["hidden_dim"],
                wdl_scale=400.0,
            ).to(dev)
            t0 = time.perf_counter()
            loaded_meta = train_torch._load_initial_checkpoint(
                next_model,
                cp_file,
                input_dim=loaded["input_dim"],
                hidden_dim=loaded["hidden_dim"],
                device=dev,
                objective=loaded["objective"],
                arch="v2",
            )
            t1 = time.perf_counter()
            self.assertLess(t1 - t0, 0.05, "Immediate reload must hit in-memory cache")
            self.assertEqual(loaded_meta["sha256"], hashlib.sha256(cp_file.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
