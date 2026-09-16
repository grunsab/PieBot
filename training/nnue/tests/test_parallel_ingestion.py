#!/usr/bin/env python3
"""Tests for parallel byte-chunk LC0 ingestion and in-memory validation/export optimizations."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.nnue import train_torch, train_stub, features_v2


class ParallelIngestionTests(unittest.TestCase):
    def test_find_byte_chunks_covers_file_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "lines.txt"
            lines = [f"line {i:04d} content with extra text\n" for i in range(100)]
            test_file.write_text("".join(lines), encoding="utf-8")
            size = test_file.stat().st_size

            chunks = train_torch._find_byte_chunks(test_file, 4)
            self.assertEqual(4, len(chunks))
            self.assertEqual(0, chunks[0][1])
            self.assertEqual(size, chunks[-1][2])

            # Check continuous ranges
            for i in range(len(chunks) - 1):
                self.assertEqual(chunks[i][2], chunks[i + 1][1])

            # Check that each chunk boundary ends with a newline
            with test_file.open("rb") as f:
                for _, start, end in chunks:
                    if end < size:
                        f.seek(end - 1)
                        self.assertEqual(b"\n", f.read(1))

    def test_worker_parse_lc0_chunk_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "train.jsonl"
            sample_fens = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 5",
                "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN5/PPP2PPP/R2Q1RK1 w - - 0 12",
                "4k3/8/8/8/8/7r/4P3/4K3 b - - 0 1",
            ]
            rows = []
            for i, fen in enumerate(sample_fens * 25):  # 100 lines
                rows.append(json.dumps({
                    "fen": fen,
                    "q": 0.1 * (i % 5),
                    "best_q": 0.15 * (i % 5),
                    "result": 1 if i % 2 == 0 else -1,
                    "game_id": f"g_{i // 10}",
                    "run_id": "r_1",
                    "move": "e2e4",
                }))
            test_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
            size = test_file.stat().st_size

            args_serial = (
                str(test_file), 0, size,
                "wdl", 250.0, 0.8, 1500.0, 1.0, 0, 400.0
            )
            serial_res = train_torch._worker_parse_lc0_chunk(args_serial)

            # Parallel with 3 chunks
            chunks = train_torch._find_byte_chunks(test_file, 3)
            par_res_list = [
                train_torch._worker_parse_lc0_chunk((
                    c[0], c[1], c[2], "wdl", 250.0, 0.8, 1500.0, 1.0, 0, 400.0
                ))
                for c in chunks
            ]

            par_xs = []
            par_cp = []
            par_wdl = []
            par_groups = []
            par_teachers = []
            for r in par_res_list:
                par_xs.extend(r[0])
                par_cp.extend(r[1])
                par_wdl.extend(r[2])
                par_groups.extend(r[3])
                par_teachers.extend(r[4])

            self.assertEqual(len(serial_res[0]), len(par_xs))
            self.assertEqual(serial_res[0], par_xs)
            self.assertEqual(serial_res[1], par_cp)
            self.assertEqual(serial_res[2], par_wdl)
            self.assertEqual(serial_res[3], par_groups)
            self.assertEqual(serial_res[4], par_teachers)

    def test_selected_checkpoint_sink_captures_checkpoint(self) -> None:
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

            captured = []
            def sink(cp):
                captured.append(cp)

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
                out_dir=root / "out",
                device="cpu",
                selected_checkpoint_sink=sink,
            )

            self.assertEqual(1, len(captured))
            checkpoint = captured[0]
            self.assertEqual("v2", checkpoint["arch"])
            self.assertEqual(16, checkpoint["hidden_dim"])
            # Saved file matches captured checkpoint
            saved_cp = json.loads((root / "out" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_cp["arch"], checkpoint["arch"])
            self.assertEqual(saved_cp["hidden_dim"], checkpoint["hidden_dim"])

    def test_fixed_validation_cache_hits_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_file = root / "validation" / "validation.jsonl"
            val_file.parent.mkdir(parents=True)
            rows = [
                json.dumps({
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "q": 0.0, "best_q": 0.0, "result": 0, "game_id": "g_val", "run_id": "r_val"
                }),
            ]
            val_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

            train_file = root / "train" / "train.jsonl"
            train_file.parent.mkdir(parents=True)
            train_file.write_text(json.dumps({
                "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "q": 0.0, "best_q": 0.0, "result": 0, "game_id": "g_tr", "run_id": "r_tr"
            }) + "\n", encoding="utf-8")

            # Run 1: populates cache
            metrics1 = train_torch.train_model(
                jsonl_dir=train_file.parent,
                batch_size=1, max_samples=1, epochs=1, val_split=0.0,
                learning_rate=0.0, hidden_dim=16, arch="v2",
                loss_kind="wdl", wdl_scale_cp=400.0, cp_loss_weight=0.0,
                outcome_decay=1.0, min_teacher_depth=0, teacher_mix=0.8,
                target_mode="lc0-q-outcome", checkpoint_selection="latest",
                out_dir=root / "out1", device="cpu",
                validation_jsonl_dir=val_file.parent,
            )

            # Run 2: hits cache
            metrics2 = train_torch.train_model(
                jsonl_dir=train_file.parent,
                batch_size=1, max_samples=1, epochs=1, val_split=0.0,
                learning_rate=0.0, hidden_dim=16, arch="v2",
                loss_kind="wdl", wdl_scale_cp=400.0, cp_loss_weight=0.0,
                outcome_decay=1.0, min_teacher_depth=0, teacher_mix=0.8,
                target_mode="lc0-q-outcome", checkpoint_selection="latest",
                out_dir=root / "out2", device="cpu",
                validation_jsonl_dir=val_file.parent,
            )

            self.assertEqual(
                metrics1["validation_sample_sha256"],
                metrics2["validation_sample_sha256"]
            )
            self.assertEqual(
                metrics1["initial_reference_val_loss"],
                metrics2["initial_reference_val_loss"]
            )


if __name__ == "__main__":
    unittest.main()
