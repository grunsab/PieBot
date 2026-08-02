import json
import inspect
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
    def test_cycle_seed_is_deterministic_but_varies_by_cycle_and_stream(self) -> None:
        first = autopilot._derive_cycle_seed(42, 1, stream=0)
        self.assertEqual(first, autopilot._derive_cycle_seed(42, 1, stream=0))
        self.assertNotEqual(first, autopilot._derive_cycle_seed(42, 2, stream=0))
        self.assertNotEqual(first, autopilot._derive_cycle_seed(42, 1, stream=1))
        self.assertGreater(first, 0)
        self.assertLessEqual(first, (1 << 64) - 1)

    def test_zen5_9755_profile_has_expected_relabel_defaults(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertEqual(1, profile["selfplay_threads"])
        self.assertEqual(0, profile["selfplay_parallel_games"])
        self.assertEqual(5, profile["teacher_relabel_depth"])
        self.assertEqual(8, profile["teacher_relabel_every"])
        self.assertGreaterEqual(profile["teacher_relabel_threads"], 32)
        self.assertGreaterEqual(profile["teacher_relabel_hash_mb"], 2048)
        self.assertEqual(0, profile["retain_full_cycles"])

    def test_cycle_retention_prunes_old_cycles_and_preserves_accepted_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            cycles_root = out_root / "cycles"
            completed = []
            quant_paths = {}
            for cycle in range(1, 5):
                cycle_dir = cycles_root / f"cycle_{cycle:06d}"
                (cycle_dir / "selfplay_jsonl").mkdir(parents=True)
                (cycle_dir / "jsonl_relabel").mkdir()
                (cycle_dir / "train").mkdir()
                (cycle_dir / "selfplay_jsonl" / "data.jsonl").write_text("bulk\n")
                (cycle_dir / "jsonl_relabel" / "data.jsonl").write_text("bulk\n")
                (cycle_dir / "train" / "checkpoint.json").write_text("bulk")
                (cycle_dir / "nnue_dense.nnue").write_bytes(b"dense")
                quant = cycle_dir / "nnue_quant.nnue"
                quant.write_bytes(f"quant-{cycle}".encode())
                quant_paths[cycle] = quant
                (cycle_dir / "pipeline_summary.json").write_text(
                    json.dumps({"cycle": cycle}), encoding="utf-8"
                )
                (cycle_dir / "gate_compare.json").write_text(
                    json.dumps({"cycle": cycle}), encoding="utf-8"
                )
                completed.append(
                    {
                        "cycle": cycle,
                        "out_dir": str(cycle_dir),
                        "jsonl_dir": str(cycle_dir / "jsonl_relabel"),
                        "train_jsonl_dir": str(cycle_dir / "jsonl_relabel"),
                        "quant_path": str(quant),
                        "summary_path": str(cycle_dir / "pipeline_summary.json"),
                        "gate": {"accepted": cycle in (2, 4)},
                    }
                )
            state = {
                "completed_cycles": completed,
                "accepted_models": [
                    {"cycle": 2, "quant_path": str(quant_paths[2])},
                    {"cycle": 4, "quant_path": str(quant_paths[4])},
                ],
                "active_model_path": str(quant_paths[4]),
            }

            report = autopilot._apply_cycle_retention(
                out_root=out_root,
                state=state,
                retain_full_cycles=2,
            )

            self.assertEqual([1], report["deleted_cycles"])
            self.assertEqual([2], report["pruned_cycles"])
            self.assertFalse((cycles_root / "cycle_000001").exists())
            accepted_old = cycles_root / "cycle_000002"
            self.assertTrue(quant_paths[2].is_file())
            self.assertTrue((accepted_old / "pipeline_summary.json").is_file())
            self.assertTrue((accepted_old / "gate_compare.json").is_file())
            self.assertTrue((accepted_old / "retained_cycle.json").is_file())
            self.assertFalse((accepted_old / "selfplay_jsonl").exists())
            self.assertFalse((accepted_old / "jsonl_relabel").exists())
            self.assertFalse((accepted_old / "train").exists())
            self.assertFalse((accepted_old / "nnue_dense.nnue").exists())
            self.assertTrue((cycles_root / "cycle_000003" / "selfplay_jsonl").is_dir())
            self.assertTrue((cycles_root / "cycle_000004" / "selfplay_jsonl").is_dir())
            self.assertTrue(Path(state["active_model_path"]).is_file())
            self.assertEqual("deleted", completed[0]["retention"])
            self.assertEqual("model_only", completed[1]["retention"])
            self.assertEqual("full", completed[2]["retention"])

            # A crash/restart may repeat cleanup; it must be harmless.
            second = autopilot._apply_cycle_retention(
                out_root=out_root,
                state=state,
                retain_full_cycles=2,
            )
            self.assertEqual([], second["deleted_cycles"])
            self.assertEqual([], second["pruned_cycles"])
            self.assertTrue(quant_paths[2].is_file())

    def test_cycle_retention_zero_is_noop_and_unsafe_paths_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            cycle1 = out_root / "cycles" / "cycle_000001"
            cycle2 = out_root / "cycles" / "cycle_000002"
            cycle1.mkdir(parents=True)
            cycle2.mkdir(parents=True)
            (cycle1 / "sentinel").write_text("keep")
            (cycle2 / "sentinel").write_text("keep")
            outside = root / "outside"
            outside.mkdir()
            (outside / "sentinel").write_text("outside")
            state = {
                "completed_cycles": [
                    {"cycle": 1, "out_dir": str(outside), "gate": {"accepted": False}},
                    {"cycle": 2, "out_dir": str(cycle2), "gate": {"accepted": False}},
                ],
                "accepted_models": [],
                "active_model_path": None,
            }

            noop = autopilot._apply_cycle_retention(
                out_root=out_root,
                state=state,
                retain_full_cycles=0,
            )
            self.assertEqual([], noop["deleted_cycles"])
            self.assertTrue((outside / "sentinel").is_file())

            with self.assertRaisesRegex(ValueError, "outside"):
                autopilot._apply_cycle_retention(
                    out_root=out_root,
                    state=state,
                    retain_full_cycles=1,
                )
            self.assertTrue((outside / "sentinel").is_file())
            self.assertTrue((cycle1 / "sentinel").is_file())
            self.assertTrue((cycle2 / "sentinel").is_file())

    def test_cycle_retention_refuses_accepted_quant_outside_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            cycles_root = out_root / "cycles"
            cycle1 = cycles_root / "cycle_000001"
            cycle2 = cycles_root / "cycle_000002"
            cycle1.mkdir(parents=True)
            cycle2.mkdir(parents=True)
            outside_quant = root / "outside.nnue"
            outside_quant.write_bytes(b"do-not-touch")
            state = {
                "completed_cycles": [
                    {
                        "cycle": 1,
                        "out_dir": str(cycle1),
                        "quant_path": str(outside_quant),
                        "gate": {"accepted": True},
                    },
                    {"cycle": 2, "out_dir": str(cycle2), "gate": {"accepted": False}},
                ],
                "accepted_models": [{"cycle": 1, "quant_path": str(outside_quant)}],
                "active_model_path": str(outside_quant),
            }

            with self.assertRaisesRegex(ValueError, "outside"):
                autopilot._apply_cycle_retention(
                    out_root=out_root,
                    state=state,
                    retain_full_cycles=1,
                )
            self.assertEqual(b"do-not-touch", outside_quant.read_bytes())
            self.assertTrue(cycle1.is_dir())

    def test_main_retries_retention_on_restart_before_starting_another_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            cycle1 = out_root / "cycles" / "cycle_000001"
            cycle2 = out_root / "cycles" / "cycle_000002"
            cycle1.mkdir(parents=True)
            cycle2.mkdir(parents=True)
            (cycle1 / "bulk").write_text("old")
            (cycle2 / "bulk").write_text("new")
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 0.0,
                "next_cycle": 3,
                "completed_cycles": [
                    {
                        "cycle": 1,
                        "out_dir": str(cycle1),
                        "gate": {"accepted": False},
                    },
                    {
                        "cycle": 2,
                        "out_dir": str(cycle2),
                        "gate": {"accepted": False},
                    },
                ],
                "accepted_models": [],
                "active_model_path": None,
                "last_error": None,
            }
            (out_root / "autopilot_state.json").write_text(json.dumps(state), encoding="utf-8")

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=AssertionError("expired restart must not start a new cycle"),
            ):
                rc = autopilot.main(
                    [
                        "--out-root",
                        str(out_root),
                        "--retain-full-cycles",
                        "1",
                    ]
                )

            self.assertEqual(0, rc)
            self.assertFalse(cycle1.exists())
            self.assertTrue((cycle2 / "bulk").is_file())
            loaded = json.loads((out_root / "autopilot_state.json").read_text(encoding="utf-8"))
            self.assertEqual("deleted", loaded["completed_cycles"][0]["retention"])
            self.assertEqual("full", loaded["completed_cycles"][1]["retention"])

    def test_blend_percent_ramps_with_number_of_accepted_models(self) -> None:
        self.assertEqual(0, autopilot._active_model_blend_percent({}))
        self.assertEqual(
            25,
            autopilot._active_model_blend_percent({"accepted_models": [{"quant_path": "a.nnue"}]}),
        )
        self.assertEqual(
            50,
            autopilot._active_model_blend_percent(
                {"accepted_models": [{"quant_path": "a.nnue"}, {"quant_path": "b.nnue"}]}
            ),
        )
        self.assertEqual(
            75,
            autopilot._active_model_blend_percent(
                {
                    "accepted_models": [
                        {"quant_path": "a.nnue"},
                        {"quant_path": "b.nnue"},
                        {"quant_path": "c.nnue"},
                    ]
                }
            ),
        )
        self.assertEqual(
            100,
            autopilot._active_model_blend_percent(
                {
                    "accepted_models": [
                        {"quant_path": "a.nnue"},
                        {"quant_path": "b.nnue"},
                        {"quant_path": "c.nnue"},
                        {"quant_path": "d.nnue"},
                    ]
                }
            ),
        )

    def test_cycle_uses_previous_quant_for_bootstrap_after_first_accept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            calls = []
            gate_calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                return {"quant_path": str(quant_path)}

            def _fake_gate(*, base_quant, candidate_quant, **_kwargs):
                gate_calls.append((base_quant, candidate_quant))
                return {"accepted": True}

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
            self.assertEqual(2, len(calls))
            self.assertEqual(2, len(gate_calls))
            first_quant = Path(calls[0]["out_dir"]) / "nnue_quant.nnue"
            self.assertEqual(first_quant, calls[1]["selfplay_nnue_quant_file"])
            self.assertEqual(first_quant, calls[1]["teacher_relabel_nnue_quant_file"])
            self.assertEqual(25, calls[1]["selfplay_nnue_blend_percent"])
            self.assertEqual(25, calls[1]["teacher_relabel_nnue_blend_percent"])
            self.assertIsNone(gate_calls[0][0])

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

    def test_replay_uses_each_cycles_fresh_jsonl_not_its_prior_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh1 = root / "cycle1" / "jsonl_relabel"
            merged1 = root / "cycle1" / "jsonl_train"
            fresh2 = root / "cycle2" / "jsonl_relabel"
            merged2 = root / "cycle2" / "jsonl_train"
            for path in (fresh1, merged1, fresh2, merged2):
                path.mkdir(parents=True, exist_ok=True)
            state = {
                "completed_cycles": [
                    {
                        "cycle": 1,
                        "jsonl_dir": str(fresh1),
                        "train_jsonl_dir": str(merged1),
                    },
                    {
                        "cycle": 2,
                        "jsonl_dir": str(fresh2),
                        "train_jsonl_dir": str(merged2),
                    },
                ]
            }

            self.assertEqual(
                [fresh2, fresh1],
                autopilot._collect_replay_jsonl_dirs(state, 2),
            )

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

    def test_current_state_schema_does_not_fallback_to_last_summary_when_no_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")
            state = {
                "active_model_path": None,
                "last_summary": {"quant_path": str(candidate)},
            }
            self.assertIsNone(autopilot._resolve_active_quant_path(state))

    def test_bootstrap_reject_keeps_default_engine_active(self) -> None:
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
            self.assertEqual(2, len(gate_calls))
            self.assertIsNone(gate_calls[0][0])
            self.assertIsNone(gate_calls[1][0])
            self.assertIsNone(created[0][0]["selfplay_nnue_quant_file"])
            self.assertIsNone(created[1][0]["selfplay_nnue_quant_file"])
            self.assertNotEqual(created[0][0]["selfplay_seed"], created[1][0]["selfplay_seed"])
            self.assertNotEqual(created[0][0]["seed"], created[1][0]["seed"])
            self.assertEqual(
                autopilot._derive_cycle_seed(42, 1, stream=0),
                created[0][0]["selfplay_seed"],
            )
            self.assertEqual(
                autopilot._derive_cycle_seed(42, 2, stream=0),
                created[1][0]["selfplay_seed"],
            )

            state = json.loads((out_root / "autopilot_state.json").read_text(encoding="utf-8"))
            self.assertIsNone(state["active_model_path"])

    def test_model_gate_command_has_one_cargo_separator_and_forwards_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "gate.json"
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")
            commands = []

            def _fake_run(cmd, **kwargs):
                commands.append((cmd, kwargs))
                out_json.write_text(
                    json.dumps(
                        {
                            "games": 6,
                            "points": {"baseline": 2.0, "experimental": 4.0},
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch("training.nnue.autopilot.subprocess.run", side_effect=_fake_run):
                gate = autopilot._run_model_gate(
                    piebot_dir=root,
                    out_json=out_json,
                    base_quant=None,
                    candidate_quant=candidate,
                    games=6,
                    movetime_ms=25,
                    noise_plies=4,
                    noise_topk=3,
                    threads=1,
                    seed=9,
                    min_score_delta=0.0,
                )

            self.assertTrue(gate["accepted"])
            self.assertEqual(1, len(commands))
            cmd, kwargs = commands[0]
            self.assertEqual(1, cmd.count("--"))
            separator = cmd.index("--")
            self.assertEqual(
                ["cargo", "run", "--locked", "--release", "--bin", "compare_play"],
                cmd[:separator],
            )
            self.assertEqual("--games", cmd[separator + 1])
            self.assertEqual("6", cmd[cmd.index("--games") + 1])
            self.assertEqual(str(root), kwargs["cwd"])
            self.assertTrue(kwargs["check"])

    def test_model_gate_cannot_reuse_stale_json_when_runner_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "gate.json"
            out_json.write_text(
                json.dumps({"games": 2, "points": {"baseline": 0.0, "experimental": 2.0}}),
                encoding="utf-8",
            )
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")

            with mock.patch("training.nnue.autopilot.subprocess.run", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    autopilot._run_model_gate(
                        piebot_dir=root,
                        out_json=out_json,
                        base_quant=None,
                        candidate_quant=candidate,
                        games=2,
                        movetime_ms=1,
                        noise_plies=0,
                        noise_topk=1,
                        threads=1,
                        seed=1,
                        min_score_delta=0.0,
                    )

    def test_model_gate_rejects_incomplete_runner_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "gate.json"
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")

            def _write_incomplete(_cmd, **_kwargs):
                out_json.write_text(json.dumps({"games": 2}), encoding="utf-8")

            with mock.patch(
                "training.nnue.autopilot.subprocess.run",
                side_effect=_write_incomplete,
            ):
                with self.assertRaisesRegex(ValueError, "points"):
                    autopilot._run_model_gate(
                        piebot_dir=root,
                        out_json=out_json,
                        base_quant=None,
                        candidate_quant=candidate,
                        games=2,
                        movetime_ms=1,
                        noise_plies=0,
                        noise_topk=1,
                        threads=1,
                        seed=1,
                        min_score_delta=0.0,
                    )

    def test_gate_reject_after_accept_keeps_previous_active_model(self) -> None:
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
            gate_results = iter(
                [
                    {
                        "accepted": True,
                        "baseline_points": 5.0,
                        "experimental_points": 7.0,
                        "delta_points": 2.0,
                    },
                    {
                        "accepted": False,
                        "baseline_points": 7.0,
                        "experimental_points": 5.0,
                        "delta_points": -2.0,
                    },
                ]
            )

            def _fake_gate(*, base_quant, candidate_quant, **_kwargs):
                gate_calls.append((base_quant, candidate_quant))
                return next(gate_results)

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
            self.assertEqual(2, len(gate_calls))
            first_quant = created[0][1]
            second_kwargs = created[1][0]
            self.assertIsNone(gate_calls[0][0])
            self.assertEqual(first_quant, gate_calls[1][0])
            self.assertEqual(first_quant, second_kwargs["selfplay_nnue_quant_file"])
            self.assertEqual(25, second_kwargs["selfplay_nnue_blend_percent"])
            self.assertEqual(25, second_kwargs["teacher_relabel_nnue_blend_percent"])

            state = json.loads((out_root / "autopilot_state.json").read_text(encoding="utf-8"))
            self.assertEqual(str(first_quant), state["active_model_path"])

    def test_second_accept_increases_blend_ramp_for_later_cycles(self) -> None:
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

            def _fake_gate(**_kwargs):
                return {
                    "accepted": True,
                    "baseline_points": 5.0,
                    "experimental_points": 7.0,
                    "delta_points": 2.0,
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
                            "3",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(3, len(created))
            self.assertEqual(25, created[1][0]["selfplay_nnue_blend_percent"])
            self.assertEqual(50, created[2][0]["selfplay_nnue_blend_percent"])
            self.assertEqual(50, created[2][0]["teacher_relabel_nnue_blend_percent"])

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

    def test_main_filters_autopilot_only_kwargs_before_run_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            allowed = set(inspect.signature(autopilot.run_pipeline.run_pipeline).parameters)

            def _strict_run_pipeline(**kwargs):
                unexpected = sorted(set(kwargs) - allowed)
                if unexpected:
                    raise TypeError(f"unexpected kwargs: {unexpected}")
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                return {"quant_path": str(quant_path)}

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_strict_run_pipeline,
            ):
                rc = autopilot.main(
                    [
                        "--out-root",
                        str(out_root),
                        "--hours",
                        "1",
                        "--max-cycles",
                        "1",
                        "--retry-limit",
                        "1",
                        "--retry-backoff-sec",
                        "0",
                        "--gate-games",
                        "0",
                    ]
                )

            self.assertEqual(0, rc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
