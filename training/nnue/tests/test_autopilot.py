import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import autopilot


class _FakeLockBackend:
    name = "fake"

    def __init__(self) -> None:
        self._locked: set[str] = set()

    def lock(self, handle) -> None:
        key = str(Path(handle.name))
        if key in self._locked:
            raise BlockingIOError(f"lock already held for {key}")
        self._locked.add(key)

    def unlock(self, handle) -> None:
        self._locked.discard(str(Path(handle.name)))


class AutopilotTests(unittest.TestCase):
    def test_zen5_9755_profile_has_expected_relabel_defaults(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertEqual(1, profile["selfplay_threads"])
        self.assertEqual(0, profile["selfplay_parallel_games"])
        self.assertEqual(9, profile["teacher_relabel_depth"])
        self.assertEqual(8, profile["teacher_relabel_every"])
        self.assertGreaterEqual(profile["teacher_relabel_threads"], 32)
        self.assertGreaterEqual(profile["teacher_relabel_hash_mb"], 2048)

    def test_cycle_uses_previous_quant_for_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                return {"quant_path": str(quant_path)}

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                rc = autopilot.main(
                    [
                        "--out-root",
                        str(out_root),
                        "--hours",
                        "1",
                        "--max-cycles",
                        "2",
                        "--gate-games",
                        "0",
                    ]
                )

            self.assertEqual(0, rc)
            self.assertEqual(2, len(calls))
            first_quant = Path(calls[0]["out_dir"]) / "nnue_quant.nnue"
            self.assertEqual(first_quant, calls[1]["selfplay_nnue_quant_file"])
            self.assertEqual(first_quant, calls[1]["teacher_relabel_nnue_quant_file"])

    def test_collect_replay_jsonl_dirs_from_recent_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a" / "train_jsonl"
            p2 = Path(tmp) / "b" / "jsonl"
            p3 = Path(tmp) / "c" / "train_jsonl"
            p1.mkdir(parents=True, exist_ok=True)
            p2.mkdir(parents=True, exist_ok=True)
            p3.mkdir(parents=True, exist_ok=True)
            state = {
                "completed_cycles": [
                    {"cycle": 1, "train_jsonl_dir": str(p1)},
                    {"cycle": 2, "jsonl_dir": str(p2)},
                    {"cycle": 3, "train_jsonl_dir": str(p3)},
                ]
            }
            got = autopilot._collect_replay_jsonl_dirs(state, 2)
            self.assertEqual([p3, p2], got)

    def test_teacher_lag_selects_older_accepted_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m1 = Path(tmp) / "m1.nnue"
            m2 = Path(tmp) / "m2.nnue"
            m3 = Path(tmp) / "m3.nnue"
            m1.write_bytes(b"PIENNQ01dummy")
            m2.write_bytes(b"PIENNQ01dummy")
            m3.write_bytes(b"PIENNQ01dummy")
            state = {
                "accepted_models": [
                    {"cycle": 1, "quant_path": str(m1)},
                    {"cycle": 2, "quant_path": str(m2)},
                    {"cycle": 3, "quant_path": str(m3)},
                ]
            }
            self.assertEqual(m2, autopilot._resolve_teacher_quant_path(state, 1))
            self.assertEqual(m1, autopilot._resolve_teacher_quant_path(state, 2))

    def test_gate_reject_keeps_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            created = []

            def _fake_run_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                created.append((kwargs, quant_path))
                return {"quant_path": str(quant_path), "jsonl_dir": str(out_dir / "jsonl_relabel")}

            gate_calls = []

            def _fake_gate(*, base_quant, candidate_quant, **_kwargs):
                gate_calls.append((base_quant, candidate_quant))
                return {
                    "accepted": False,
                    "baseline_points": 7.0,
                    "experimental_points": 5.0,
                    "delta_points": -2.0,
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    side_effect=_fake_gate,
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--hours",
                            "1",
                            "--max-cycles",
                            "2",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(1, len(gate_calls))
            # cycle 2 should bootstrap from cycle 1's accepted model
            first_quant = created[0][1]
            second_kwargs = created[1][0]
            self.assertEqual(first_quant, second_kwargs["selfplay_nnue_quant_file"])

            state = json.loads((out_root / "autopilot_state.json").read_text(encoding="utf-8"))
            self.assertEqual(str(first_quant), state["active_model_path"])

    def test_select_lock_backend_prefers_msvcrt_when_fcntl_missing(self) -> None:
        fake_msvcrt = object()
        with mock.patch.object(autopilot, "fcntl", None):
            with mock.patch.object(autopilot, "msvcrt", fake_msvcrt):
                backend = autopilot._select_lock_backend()
        self.assertEqual("msvcrt", backend.name)

    def test_single_instance_lock_rejects_second_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "autopilot.lock"
            backend = _FakeLockBackend()
            with autopilot._single_instance_lock(lock_path, backend=backend):
                with self.assertRaises(BlockingIOError):
                    with autopilot._single_instance_lock(lock_path, backend=backend):
                        pass
            with autopilot._single_instance_lock(lock_path, backend=backend):
                self.assertTrue(lock_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
