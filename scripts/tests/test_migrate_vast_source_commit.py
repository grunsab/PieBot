#!/usr/bin/env python3
"""Tests for the audited Vast.ai source-commit migration utility."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import migrate_vast_source_commit as migration


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceCommitMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.name", "PieBot Tests")
        _git(self.repo, "config", "user.email", "piebot-tests@example.invalid")

        tracked = self.repo / "tracked.txt"
        tracked.write_text("old\n", encoding="utf-8")
        _git(self.repo, "add", "tracked.txt")
        _git(self.repo, "commit", "-q", "-m", "old")
        self.old_commit = _git(self.repo, "rev-parse", "HEAD")

        tracked.write_text("new\n", encoding="utf-8")
        _git(self.repo, "commit", "-q", "-am", "new")
        self.new_commit = _git(self.repo, "rev-parse", "HEAD")

        self.out_root = self.root / "run"
        self.out_root.mkdir()
        self.lock_path = self.out_root / "autopilot.lock"
        self.lock_path.write_text("stopped-pid\n", encoding="utf-8")
        self.pin_path = self.out_root / "source_git_commit"
        self.pin_path.write_text(f"{self.old_commit}\n", encoding="utf-8")

        self.checkpoint = self.out_root / "cycles" / "cycle_000007" / "train" / "checkpoint.json"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_text('{"weights": "preserved"}\n', encoding="utf-8")
        self.active_model = self.out_root / "bootstrap" / "active.nnue"
        self.active_model.parent.mkdir()
        self.active_model.write_bytes(b"PIENNQ01-preserved-active")

        self.state_path = self.out_root / "autopilot_state.json"
        self.state = {
            "status": "stopped-at-cycle-boundary",
            "next_cycle": 8,
            "current_cycle": {"cycle": 7, "status": "completed"},
            "completed_cycles": [
                {"cycle": cycle, "status": "completed"} for cycle in range(1, 8)
            ],
            "training_checkpoint_path": str(self.checkpoint),
            "training_checkpoint_sha256": _sha256(self.checkpoint),
            "active_model_path": str(self.active_model),
            "active_model_sha256": _sha256(self.active_model),
        }
        self._write_state()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _migrate(self, **kwargs):
        return migration.migrate_source_commit(
            repo_root=self.repo,
            out_root=self.out_root,
            expected_old_commit=self.old_commit,
            expected_new_commit=self.new_commit,
            **kwargs,
        )

    def test_writes_complete_audit_before_atomically_replacing_pin(self) -> None:
        observed = {}

        def before_pin_replace(audit_path: Path, pin_path: Path) -> None:
            observed["audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
            observed["pin"] = pin_path.read_text(encoding="utf-8").strip()

        result = self._migrate(before_pin_replace=before_pin_replace)

        self.assertEqual("migrated", result.status)
        self.assertEqual(self.old_commit, observed["pin"])
        self.assertEqual(self.new_commit, self.pin_path.read_text(encoding="utf-8").strip())
        audit = observed["audit"]
        self.assertEqual("piebot-source-commit-migration-v1", audit["schema"])
        self.assertEqual("prepared", audit["phase"])
        self.assertEqual(self.old_commit, audit["source_commit"]["old"])
        self.assertEqual(self.new_commit, audit["source_commit"]["new"])
        self.assertTrue(audit["prepared_at_utc"].endswith("Z"))
        self.assertEqual(_sha256(self.state_path), audit["state"]["sha256"])
        self.assertEqual(7, audit["state"]["last_durable_completed_cycle"])
        self.assertEqual(8, audit["state"]["next_cycle"])
        self.assertEqual("stopped-at-cycle-boundary", audit["state"]["status"])
        self.assertEqual(
            {"cycle": 7, "status": "completed"},
            audit["state"]["current_cycle"],
        )
        self.assertEqual(_sha256(self.checkpoint), audit["training_checkpoint"]["sha256"])
        self.assertEqual(_sha256(self.active_model), audit["active_model"]["sha256"])

    def test_prepared_audit_recovers_idempotently_after_crash_before_pin_update(self) -> None:
        class SimulatedCrash(RuntimeError):
            pass

        def crash(_audit_path: Path, _pin_path: Path) -> None:
            raise SimulatedCrash("stop before pin replacement")

        with self.assertRaises(SimulatedCrash):
            self._migrate(before_pin_replace=crash)

        self.assertEqual(self.old_commit, self.pin_path.read_text(encoding="utf-8").strip())
        audit_path = migration.audit_path_for(
            self.out_root,
            self.old_commit,
            self.new_commit,
        )
        prepared_bytes = audit_path.read_bytes()

        recovered = self._migrate()
        self.assertEqual("migrated", recovered.status)
        self.assertEqual(prepared_bytes, audit_path.read_bytes())
        repeated = self._migrate()
        self.assertEqual("already-applied", repeated.status)
        self.assertEqual(prepared_bytes, audit_path.read_bytes())

    def test_refuses_migration_while_autopilot_lock_is_held(self) -> None:
        with self.lock_path.open("r+", encoding="utf-8") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaisesRegex(migration.MigrationError, "trainer.*running"):
                    self._migrate()
            finally:
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)

        self.assertEqual(self.old_commit, self.pin_path.read_text(encoding="utf-8").strip())
        self.assertFalse(
            migration.audit_path_for(
                self.out_root,
                self.old_commit,
                self.new_commit,
            ).exists()
        )

    def test_refuses_dirty_repository(self) -> None:
        (self.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

        with self.assertRaisesRegex(migration.MigrationError, "not clean"):
            self._migrate()

        self.assertEqual(self.old_commit, self.pin_path.read_text(encoding="utf-8").strip())

    def test_refuses_non_fast_forward_commit_transition(self) -> None:
        _git(self.repo, "checkout", "-q", self.old_commit)
        divergent = self.repo / "divergent.txt"
        divergent.write_text("divergent\n", encoding="utf-8")
        _git(self.repo, "add", "divergent.txt")
        _git(self.repo, "commit", "-q", "-m", "divergent")
        divergent_head = _git(self.repo, "rev-parse", "HEAD")

        with self.assertRaisesRegex(migration.MigrationError, "fast-forward"):
            migration.migrate_source_commit(
                repo_root=self.repo,
                out_root=self.out_root,
                expected_old_commit=self.new_commit,
                expected_new_commit=divergent_head,
            )

    def test_refuses_preserved_artifact_hash_mismatch(self) -> None:
        self.checkpoint.write_text('{"weights": "corrupt"}\n', encoding="utf-8")

        with self.assertRaisesRegex(migration.MigrationError, "checkpoint SHA-256 mismatch"):
            self._migrate()

        self.assertEqual(self.old_commit, self.pin_path.read_text(encoding="utf-8").strip())

    def test_refuses_preserved_artifact_outside_run_root(self) -> None:
        outside = self.root / "outside-checkpoint.json"
        outside.write_text('{"weights": "outside"}\n', encoding="utf-8")
        self.state["training_checkpoint_path"] = str(outside)
        self.state["training_checkpoint_sha256"] = _sha256(outside)
        self._write_state()

        with self.assertRaisesRegex(migration.MigrationError, "outside Vast output root"):
            self._migrate()

        self.assertEqual(self.old_commit, self.pin_path.read_text(encoding="utf-8").strip())

    def test_refuses_new_pin_without_prepared_audit(self) -> None:
        self.pin_path.write_text(f"{self.new_commit}\n", encoding="utf-8")

        with self.assertRaisesRegex(migration.MigrationError, "prepared audit"):
            self._migrate()


if __name__ == "__main__":
    unittest.main()
