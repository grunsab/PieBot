#!/usr/bin/env python3
"""Regression checks for unattended production launchers and their docs."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class ProductionScriptContractTests(unittest.TestCase):
    def test_documented_launchers_are_executable(self) -> None:
        for relative in (
            "scripts/fetch_mate_suite.sh",
            "scripts/run_zen5_7day.sh",
            "scripts/run_zen5_day1_validation.sh",
        ):
            with self.subTest(script=relative):
                self.assertTrue(os.access(ROOT / relative, os.X_OK))

    def test_mate_suite_uses_case_sensitive_crate_path(self) -> None:
        script = read("scripts/fetch_mate_suite.sh")
        self.assertIn('$ROOT_DIR/PieBot/data', script)
        self.assertIn('$ROOT_DIR/PieBot/src/suites', script)
        self.assertIn('cd "$ROOT_DIR/PieBot"', script)
        self.assertNotIn('$ROOT_DIR/piebot', script)

    def test_gpu_launchers_fail_closed_when_cuda_is_requested(self) -> None:
        for relative in (
            "scripts/run_zen5_7day.sh",
            "scripts/run_zen5_day1_validation.sh",
        ):
            with self.subTest(script=relative):
                script = read(relative)
                self.assertIn('TRAINER_DEVICE" == "cuda', script)
                self.assertIn("torch.cuda.is_available()", script)
                self.assertRegex(script, r"CUDA requested.*unavailable")
                self.assertRegex(
                    script,
                    r'TRAINER_DEVICE" == "cuda" && "\$TRAINER_BACKEND" == "auto"[\s\S]*?TRAINER_BACKEND="torch"',
                )

    def test_seven_day_gpu_launcher_uses_torch_by_default(self) -> None:
        script = read("scripts/run_zen5_7day.sh")
        self.assertIn('TRAINER_BACKEND="${TRAINER_BACKEND:-torch}"', script)

    def test_production_launchers_forward_cycle_retention(self) -> None:
        for relative in (
            "scripts/run_zen5_7day.sh",
            "scripts/run_zen5_day1_validation.sh",
        ):
            with self.subTest(script=relative):
                script = read(relative)
                self.assertIn('RETAIN_FULL_CYCLES="${RETAIN_FULL_CYCLES:-8}"', script)
                self.assertIn('--retain-full-cycles "$RETAIN_FULL_CYCLES"', script)

    def test_day_one_validates_both_searches_and_candidate_model(self) -> None:
        script = read("scripts/run_zen5_day1_validation.sh")
        self.assertRegex(script, r"--bin accept\s+--bin accept_temp")
        self.assertRegex(script, r"--bin accept(?:\s|$)")
        self.assertRegex(script, r"--bin accept_temp(?:\s|$)")
        self.assertIn("--same-search", script)
        self.assertIn("--base-eval pst", script)
        self.assertIn("--base-use-nnue false", script)
        self.assertIn("--exp-eval nnue", script)
        self.assertIn("--exp-use-nnue true", script)
        self.assertIn('--exp-nnue-quant-file "$NNUE_QUANT"', script)

    def test_systemd_stops_after_a_clean_deadline_exit(self) -> None:
        setup = read("documents/ZEN5_3090_NNUE_SETUP.md")
        self.assertIn("Restart=on-failure", setup)
        self.assertNotIn("Restart=always", setup)

    def test_systemd_path_includes_service_users_cargo_bin(self) -> None:
        setup = read("documents/ZEN5_3090_NNUE_SETUP.md")
        self.assertIn("User=YOUR_USER", setup)
        self.assertRegex(
            setup,
            r"Environment=PATH=[^\n]*/home/YOUR_USER/\.cargo/bin(?:[:\n])",
        )

    def test_run_directory_ownership_is_documented(self) -> None:
        setup = read("documents/ZEN5_3090_NNUE_SETUP.md")
        self.assertRegex(setup, r"install -d .* /opt/piebot_runs")
        self.assertRegex(setup, r"/opt/piebot_runs.*own|own.*?/opt/piebot_runs")

    def test_documented_compare_flags_use_clap_kebab_case(self) -> None:
        top_readme = read("README.md")
        for flag in (
            "--base-eval",
            "--base-nnue-quant-file",
            "--exp-eval",
            "--exp-nnue-quant-file",
        ):
            self.assertIn(flag, top_readme)
        self.assertIsNone(re.search(r"--(?:base|exp)_[a-z0-9_-]+", top_readme))

    def test_training_docs_do_not_claim_bench_compile_drift(self) -> None:
        for relative in ("README.md", "training/nnue/README.md"):
            with self.subTest(document=relative):
                document = read(relative).lower()
                self.assertNotIn("bench.rs", document)
                self.assertNotIn("compile drift", document)

    def test_production_cargo_commands_are_dependency_locked(self) -> None:
        for relative in (
            "scripts/run_zen5_7day.sh",
            "scripts/run_zen5_day1_validation.sh",
            "scripts/fetch_mate_suite.sh",
        ):
            with self.subTest(script=relative):
                for command in re.findall(r"cargo (?:build|run)[^\n]*(?:\\\n[^\n]*)*", read(relative)):
                    self.assertIn("--locked", command)


if __name__ == "__main__":
    unittest.main()
