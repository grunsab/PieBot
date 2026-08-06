#!/usr/bin/env python3
"""Static contract checks for the portable CPU benchmark."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts" / "cpu_benchmark.sh"


class CpuBenchmarkContractTests(unittest.TestCase):
    def test_script_is_executable_with_valid_syntax(self) -> None:
        self.assertTrue(BENCH.is_file())
        self.assertTrue(os.access(BENCH, os.X_OK))
        subprocess.run(["bash", "-n", str(BENCH)], check=True)

    def test_benchmark_is_deterministic_and_uses_committed_assets(self) -> None:
        script = BENCH.read_text(encoding="utf-8")
        self.assertIn("--seed 777", script)
        self.assertIn("models/cycle_000098_quant.nnue", script)
        self.assertIn("books/openings_v1.fen", script)
        self.assertIn("piebot-cpu-benchmark-v1", script)

    def test_committed_model_matches_the_campaign_incumbent_sha(self) -> None:
        import hashlib

        model = ROOT / "models" / "cycle_000098_quant.nnue"
        self.assertTrue(model.is_file())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(
            "3fa9bae3127319930ec16ebb1ee3117656abe7001984f6c8655108a08d278c3a",
            digest,
        )


if __name__ == "__main__":
    unittest.main()
