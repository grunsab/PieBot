#!/usr/bin/env python3
"""Static contract checks for the 72-hour Vast.ai self-teacher deployment."""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_vast_5090_self_teacher_72h.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "piebot_training_72h_self_teacher.conf"
VALIDATION_PROVENANCE = (
    ROOT / "deploy" / "vast" / "piebot_fixed_validation_provenance.json"
)


class VastSelfTeacherDeploymentTests(unittest.TestCase):
    def test_launcher_is_executable_and_has_valid_bash_syntax(self) -> None:
        self.assertTrue(LAUNCHER.is_file())
        self.assertTrue(os.access(LAUNCHER, os.X_OK))
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_launcher_is_explicitly_piebot_self_teacher_only(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("--teacher-relabel-engine", launcher)
        external_engine_name = "".join(("stock", "fish"))
        self.assertNotIn(external_engine_name, launcher.lower())
        self.assertIn('RELABEL_DEPTH="${RELABEL_DEPTH:-5}"', launcher)
        self.assertIn('RELABEL_EVERY="${RELABEL_EVERY:-2}"', launcher)
        self.assertIn('RELABEL_THREADS="${RELABEL_THREADS:-46}"', launcher)

    def test_launcher_bootstraps_from_exact_prior_active_piebot_model(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        expected_model = (
            "/workspace/piebot_runs/main_48h_20260802T081500Z/"
            "cycles/cycle_000083/nnue_quant.nnue"
        )
        expected_sha256 = (
            "6a9c02212cd4b08c30e1797bf94b4742e"
            "a3cc8370aa2410f2a58be8924737101"
        )
        self.assertIn(expected_model, launcher)
        self.assertIn(expected_sha256, launcher)
        self.assertIn(
            'INITIAL_ACTIVE_MODEL_BLEND_PERCENT="${INITIAL_ACTIVE_MODEL_BLEND_PERCENT:-50}"',
            launcher,
        )
        self.assertIn(
            'verify_sha256 "$INITIAL_ACTIVE_MODEL" "$INITIAL_ACTIVE_MODEL_SHA256"',
            launcher,
        )
        self.assertIn('require_autopilot_flag "--initial-active-model"', launcher)
        self.assertIn(
            'require_autopilot_flag "--initial-active-model-blend-percent"',
            launcher,
        )
        self.assertIn('"--initial-active-model" "$INITIAL_ACTIVE_MODEL"', launcher)
        self.assertIn(
            '"--initial-active-model-blend-percent" "$INITIAL_ACTIVE_MODEL_BLEND_PERCENT"',
            launcher,
        )
        self.assertIn('BOOTSTRAP_DIR="$OUT_ROOT/bootstrap"', launcher)
        self.assertIn(
            'stage_verified_file "$INITIAL_ACTIVE_MODEL_SOURCE" "$INITIAL_ACTIVE_MODEL"',
            launcher,
        )

    def test_launcher_preserves_72_hour_deadline_across_restarts(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('HOURS="${HOURS:-72}"', launcher)
        self.assertIn(
            'OUT_ROOT="${OUT_ROOT:-/workspace/piebot_runs/main_72h_self_teacher_repair_v1}"',
            launcher,
        )
        out_root_line = next(
            line for line in launcher.splitlines() if line.startswith("OUT_ROOT=")
        )
        self.assertNotIn("$(", out_root_line)
        self.assertIn('"--hours" "$HOURS"', launcher)

    def test_launcher_uses_prior_weights_without_old_optimizer_moments(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        expected_checkpoint = (
            "/workspace/piebot_runs/main_48h_20260802T081500Z/"
            "cycles/cycle_000086/train/checkpoint.json"
        )
        expected_checkpoint_sha256 = (
            "0ce48cc1299d5750bd43512793e843d83"
            "63e1e09a5c4a72c3b22e024951f367c"
        )
        self.assertIn(expected_checkpoint, launcher)
        self.assertIn(expected_checkpoint_sha256, launcher)
        self.assertIn(
            'verify_sha256 "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"',
            launcher,
        )
        self.assertIn(
            'stage_verified_file "$INITIAL_CHECKPOINT_SOURCE" "$INITIAL_CHECKPOINT"',
            launcher,
        )
        self.assertIn('"--initial-checkpoint-weights-only"', launcher)
        self.assertIn('require_autopilot_flag "--initial-checkpoint-weights-only"', launcher)
        self.assertIn('"--continue-optimizer-state"', launcher)

    def test_launcher_pins_clean_source_commit_across_restarts(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("git status --porcelain", launcher)
        self.assertIn("git rev-parse HEAD", launcher)
        self.assertIn("source_git_commit", launcher)
        self.assertIn("refusing source commit change", launcher)

    def test_launcher_copies_and_pins_self_labeled_validation_inside_new_run(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        expected_validation_sha256 = (
            "d6f4a72a356bb516f62f76488b89c4c"
            "70519acca93c625f55709e958485bc8d8"
        )
        self.assertIn(expected_validation_sha256, launcher)
        self.assertIn(
            'stage_verified_file "$VALIDATION_SHARD_SOURCE" "$VALIDATION_SHARD"',
            launcher,
        )
        self.assertIn('VALIDATION_JSONL_DIR="$BOOTSTRAP_DIR/validation"', launcher)
        self.assertIn('"--validation-jsonl-dir" "$VALIDATION_JSONL_DIR"', launcher)
        self.assertIn(
            '"--validation-provenance-json" "$VALIDATION_PROVENANCE"', launcher
        )

        metadata = json.loads(VALIDATION_PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual("piebot-validation-provenance-v1", metadata["schema"])
        self.assertFalse(metadata["independent_of_piebot"])
        self.assertIn("piebot", metadata["source"]["kind"].lower())
        self.assertEqual(
            "3b6b3668ef68d66b19dade18eedb7ec987762f7f46521664f5006f7da07de41a",
            metadata["dataset_sha256"],
        )

    def test_launcher_distinguishes_checkpoint_holdout_from_reference_envelope(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "--validation-jsonl-dir supplies the pinned depth-6 reference/safety envelope",
            launcher,
        )
        self.assertIn(
            "it never ranks epochs, but vetoes a >1% reference-loss regression",
            launcher,
        )
        self.assertIn(
            'log "reference/safety envelope: pinned depth-6 corpus at '
            '$VALIDATION_JSONL_DIR (--validation-jsonl-dir; vetoes >1% loss regression)"',
            launcher,
        )
        self.assertIn(
            'log "checkpoint selection: stable game-hash-aligned depth-5 holdout '
            'ranks eligible checkpoints"',
            launcher,
        )

    def test_launcher_uses_vm_and_gpu_capacity_fail_closed(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"', launcher)
        self.assertIn('SELFPLAY_PARALLEL_GAMES="${SELFPLAY_PARALLEL_GAMES:-46}"', launcher)
        self.assertIn('GATE_SEARCH_THREADS="${GATE_SEARCH_THREADS:-1}"', launcher)
        self.assertIn('GATE_PARALLEL_GAMES="${GATE_PARALLEL_GAMES:-12}"', launcher)
        self.assertIn(
            'GATE_CPU_SLOTS=$((GATE_SEARCH_THREADS * GATE_PARALLEL_GAMES))',
            launcher,
        )
        self.assertIn(
            'require_positive_int GATE_PARALLEL_GAMES "$GATE_PARALLEL_GAMES"',
            launcher,
        )
        self.assertIn(
            'parallel promotion matches require GATE_SEARCH_THREADS=1', launcher
        )
        self.assertIn(
            '"--gate-parallel-games" "$GATE_PARALLEL_GAMES"', launcher
        )
        self.assertIn('"--gate-threads" "$GATE_SEARCH_THREADS"', launcher)
        self.assertIn("torch.cuda.is_available()", launcher)
        self.assertIn("46", launcher)
        self.assertIn("-C target-cpu=native", launcher)

    def test_supervisor_template_has_safe_long_job_semantics(self) -> None:
        self.assertTrue(SUPERVISOR.is_file())
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(SUPERVISOR, encoding="utf-8")
        section = parser["program:piebot_training_72h_self_teacher"]
        self.assertEqual("/workspace/piebot_rust", section["directory"])
        self.assertEqual(
            "/workspace/piebot_rust/scripts/run_vast_5090_self_teacher_72h.sh",
            section["command"],
        )
        self.assertEqual("unexpected", section["autorestart"])
        self.assertEqual("true", section["stopasgroup"])
        self.assertEqual("true", section["killasgroup"])
        self.assertEqual("/dev/stdout", section["stdout_logfile"])
        self.assertEqual("0", section["stdout_logfile_maxbytes"])


if __name__ == "__main__":
    unittest.main()
