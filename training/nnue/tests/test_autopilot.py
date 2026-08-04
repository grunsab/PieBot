import json
import hashlib
import inspect
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import autopilot, train_stub


def _write_fake_checkpoint(out_dir: Path) -> Path:
    checkpoint = out_dir / "train" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"fake": out_dir.name}), encoding="utf-8")
    return checkpoint


def _write_fake_quant(
    path: Path,
    *,
    input_dim: int = 40_960,
    hidden_dim: int = 64,
    output_dim: int = 1,
    marker: bytes = b"",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"PIENNQ01"
        + struct.pack("<IIII", 1, input_dim, hidden_dim, output_dim)
        + marker
    )
    return path


def _paired_gate_payload(pair_deltas: list[float]) -> dict:
    """Build compare_play evidence from experimental-minus-baseline pair deltas."""
    game_results = []
    baseline_wins = 0.0
    experimental_wins = 0.0
    draws = 0
    game_index = 0
    for pair_index, delta in enumerate(pair_deltas):
        # The compact fixtures only need the five legal pentanomial outcomes.
        score_by_delta = {
            -2.0: (0.0, 0.0),
            -1.0: (0.0, 0.5),
            0.0: (0.5, 0.5),
            1.0: (1.0, 0.5),
            2.0: (1.0, 1.0),
        }
        first_exp, second_exp = score_by_delta[float(delta)]
        for experimental_score in (first_exp, second_exp):
            baseline_score = 1.0 - experimental_score
            baseline_wins += float(baseline_score == 1.0)
            experimental_wins += float(experimental_score == 1.0)
            draws += int(experimental_score == 0.5)
            game_results.append(
                {
                    "game_index": game_index,
                    "pair_index": pair_index,
                    "baseline_is_white": game_index % 2 == 0,
                    "baseline_score": baseline_score,
                    "experimental_score": experimental_score,
                    "opening_id": f"opening-{pair_index}",
                }
            )
            game_index += 1
    return {
        "games": len(game_results),
        "paired_openings": True,
        "points": {
            "baseline": baseline_wins,
            "experimental": experimental_wins,
            "draws": draws,
        },
        "game_results": game_results,
    }


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
        self.assertEqual(6, profile["teacher_relabel_depth"])
        self.assertEqual(4, profile["teacher_relabel_every"])
        self.assertGreaterEqual(profile["teacher_relabel_threads"], 32)
        self.assertGreaterEqual(profile["teacher_relabel_hash_mb"], 2048)
        self.assertEqual(0, profile["retain_full_cycles"])
        self.assertTrue(profile["warm_start"])
        self.assertEqual(0.001, profile["warm_start_learning_rate"])
        self.assertEqual(0.5, profile["primary_sample_fraction"])
        self.assertEqual(0.5, profile["teacher_sample_fraction"])
        self.assertEqual(6, profile["min_teacher_depth"])
        self.assertEqual("wdl", profile["loss_kind"])
        self.assertEqual(100.0, profile["huber_delta_cp"])
        self.assertEqual(400.0, profile["wdl_scale_cp"])
        self.assertIsNone(profile["validation_jsonl_dir"])
        self.assertEqual(100_000, profile["max_validation_samples"])
        self.assertEqual(20_260_802, profile["validation_seed"])
        self.assertTrue(profile["continue_optimizer_state"])
        self.assertTrue(profile["validation_require_teacher"])
        self.assertEqual("halfkp-all-pieces-v2", profile["training_feature_set"])
        self.assertEqual(81_920, profile["training_input_dim"])
        self.assertEqual("soft-cp-wdl-v2", profile["training_target_schema"])
        expected_objective = {
            "schema": "nnue-objective-v1",
            "target_schema": "soft-cp-wdl-v2",
            "loss_kind": "wdl",
            "target_cp": 100.0,
            "teacher_mix": 0.8,
            "max_teacher_cp": 1200.0,
            "outcome_decay": 1.0,
            "min_teacher_depth": 6,
            "huber_delta_cp": 100.0,
            "wdl_scale_cp": 400.0,
        }
        self.assertEqual(expected_objective, autopilot._configured_training_objective(profile))
        self.assertEqual(
            train_stub.objective_metadata(
                loss_kind="wdl",
                target_cp=100.0,
                teacher_mix=0.8,
                max_teacher_cp=1200.0,
                outcome_decay=1.0,
                min_teacher_depth=6,
                huber_delta_cp=100.0,
                wdl_scale_cp=400.0,
            ),
            autopilot._configured_training_objective(profile),
        )
        # The production VM's scalar x86 path loses 23% search throughput at
        # 96 units and 40% at 128 versus 64.  The all-piece v2 feature set
        # supplies the added capacity while keeping the search usable.
        self.assertEqual(profile["hidden_dim"], 64)
        self.assertGreaterEqual(profile["max_samples"], 700_000)
        self.assertEqual(0, profile["teacher_lag_cycles"])
        self.assertTrue(profile["gate_paired_openings"])
        self.assertEqual(96, profile["gate_confirmation_games"])
        self.assertEqual(0.0, profile["gate_confirmation_min_score_delta"])
        self.assertEqual(0.95, profile["gate_confidence_level"])
        self.assertGreaterEqual(profile["gate_bootstrap_samples"], 10_000)
        self.assertFalse(profile["gate_require_external_anchor"])
        self.assertIsNone(profile["gate_external_anchor_json"])
        self.assertIsNone(profile["validation_provenance_json"])
        self.assertFalse(profile["initial_checkpoint_weights_only"])
        self.assertIsNone(profile["initial_active_model"])
        self.assertEqual(0, profile["initial_active_model_blend_percent"])


    def test_control_loop_cli_overrides_profile_values(self) -> None:
        validation_dir = Path("fixed-validation")
        args = autopilot._parse_args(
            [
                "--out-root",
                "runs",
                "--primary-sample-fraction",
                "0.7",
                "--teacher-sample-fraction",
                "0.6",
                "--min-teacher-depth",
                "8",
                "--loss-kind",
                "mse",
                "--huber-delta-cp",
                "75",
                "--wdl-scale-cp",
                "300",
                "--validation-jsonl-dir",
                str(validation_dir),
                "--max-validation-samples",
                "12345",
                "--validation-seed",
                "99",
                "--no-continue-optimizer-state",
                "--no-validation-require-teacher",
                "--no-gate-paired-openings",
                "--gate-confirmation-games",
                "48",
                "--gate-confirmation-min-score-delta",
                "3",
            ]
        )

        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )

        self.assertEqual(0.7, resolved["primary_sample_fraction"])
        self.assertEqual(0.6, resolved["teacher_sample_fraction"])
        self.assertEqual(8, resolved["min_teacher_depth"])
        self.assertEqual("mse", resolved["loss_kind"])
        self.assertEqual(75.0, resolved["huber_delta_cp"])
        self.assertEqual(300.0, resolved["wdl_scale_cp"])
        self.assertEqual(validation_dir, resolved["validation_jsonl_dir"])
        self.assertEqual(12_345, resolved["max_validation_samples"])
        self.assertEqual(99, resolved["validation_seed"])
        self.assertFalse(resolved["continue_optimizer_state"])
        self.assertFalse(resolved["validation_require_teacher"])
        self.assertFalse(resolved["gate_paired_openings"])
        self.assertEqual(48, resolved["gate_confirmation_games"])
        self.assertEqual(3.0, resolved["gate_confirmation_min_score_delta"])

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
                        "checkpoint_path": str(cycle_dir / "train" / "checkpoint.json"),
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
            self.assertIsNone(completed[1]["checkpoint_path"])
            self.assertFalse((accepted_old / "nnue_dense.nnue").exists())
            self.assertTrue((cycles_root / "cycle_000003" / "selfplay_jsonl").is_dir())
            self.assertTrue((cycles_root / "cycle_000004" / "selfplay_jsonl").is_dir())
            self.assertTrue(Path(state["active_model_path"]).is_file())
            self.assertEqual("deleted", completed[0]["retention"])
            self.assertEqual("model_only", completed[1]["retention"])
            self.assertEqual("full", completed[2]["retention"])
            self.assertTrue(Path(completed[2]["checkpoint_path"]).is_file())

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

    def test_cycle_retention_preserves_external_initial_active_after_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            external = _write_fake_quant(root / "bootstrap.nnue", marker=b"external")
            metadata = autopilot._initial_active_model_metadata(external, 40)
            completed = []
            for cycle in range(1, 4):
                cycle_dir = out_root / "cycles" / f"cycle_{cycle:06d}"
                cycle_dir.mkdir(parents=True)
                (cycle_dir / "bulk.jsonl").write_text("data\n", encoding="utf-8")
                completed.append(
                    {
                        "cycle": cycle,
                        "out_dir": str(cycle_dir),
                        "gate": {"accepted": False},
                        "retention": "full",
                    }
                )
            state = {
                "completed_cycles": completed,
                "accepted_models": [],
                "initial_active_model": metadata,
                "active_model_path": metadata["path"],
                "active_model_sha256": metadata["sha256"],
                "active_model_blend_percent": metadata["blend_percent"],
                "active_model_identity": metadata["model_identity"],
            }

            report = autopilot._apply_cycle_retention(
                out_root=out_root,
                state=state,
                retain_full_cycles=1,
            )

            self.assertEqual([1, 2], report["deleted_cycles"])
            self.assertTrue(external.is_file())
            self.assertEqual(metadata["path"], state["active_model_path"])
            self.assertTrue((out_root / "cycles" / "cycle_000003").is_dir())

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

    def test_live_restart_marks_run_running_before_launching_next_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            out_root.mkdir()
            state_path = out_root / "autopilot_state.json"
            stale_error = {
                "cycle": 1,
                "attempt": 1,
                "error": "prior transient failure",
                "ts": 123.0,
            }
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 10**12,
                "next_cycle": 2,
                "completed_cycles": [{"cycle": 1}],
                "accepted_models": [],
                "active_model_path": None,
                "training_checkpoint_path": None,
                "training_lineage_start_cycle": 1,
                "last_error": stale_error,
                "status": "complete",
                "finished_at": 456.0,
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")
            launched_states = []

            def _capture_launch_state(**_kwargs):
                launched_states.append(
                    json.loads(state_path.read_text(encoding="utf-8"))
                )
                raise RuntimeError("stop after observing launch state")

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_capture_launch_state,
            ):
                rc = autopilot.main(
                    [
                        "--out-root",
                        str(out_root),
                        "--retry-limit",
                        "1",
                        "--gate-games",
                        "0",
                    ]
                )

            self.assertEqual(2, rc)
            self.assertEqual(1, len(launched_states))
            launched = launched_states[0]
            self.assertEqual("running", launched["status"])
            self.assertNotIn("finished_at", launched)
            self.assertEqual(stale_error, launched["last_error"])
            self.assertEqual(2, launched["current_cycle"]["cycle"])
            self.assertEqual("running", launched["current_cycle"]["status"])

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
        self.assertEqual(25, autopilot._candidate_model_blend_percent({}))
        self.assertEqual(
            50,
            autopilot._candidate_model_blend_percent(
                {"accepted_models": [{"quant_path": "a.nnue"}]}
            ),
        )
        self.assertEqual(
            100,
            autopilot._candidate_model_blend_percent(
                {
                    "accepted_models": [
                        {"quant_path": "a.nnue"},
                        {"quant_path": "b.nnue"},
                        {"quant_path": "c.nnue"},
                    ]
                }
            ),
        )

    def test_quant_identity_includes_runtime_dimensions_and_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant = _write_fake_quant(
                Path(tmp) / "v2.nnue",
                input_dim=81_920,
                hidden_dim=64,
            )
            summary = {
                "metrics": {
                    "feature_set": "halfkp-all-pieces-v2",
                    "target_schema": "soft-cp-wdl-v2",
                    "objective": {
                        "schema": "nnue-objective-v1",
                        "target_schema": "soft-cp-wdl-v2",
                        "loss_kind": "wdl",
                    },
                    "input_dim": 81_920,
                    "hidden_dim": 64,
                }
            }

            self.assertEqual(
                {
                    "quant_format": "PIENNQ01",
                    "quant_version": 1,
                    "input_dim": 81_920,
                    "hidden_dim": 64,
                    "output_dim": 1,
                    "feature_set": "halfkp-all-pieces-v2",
                    "target_schema": "soft-cp-wdl-v2",
                    "objective": {
                        "schema": "nnue-objective-v1",
                        "target_schema": "soft-cp-wdl-v2",
                        "loss_kind": "wdl",
                    },
                },
                autopilot._quant_model_identity(quant, summary=summary),
            )

    def test_active_blend_uses_exact_promoted_blend_for_active_path(self) -> None:
        state = {
            "active_model_path": "legacy.nnue",
            "accepted_models": [
                {
                    "cycle": 1,
                    "quant_path": "older.nnue",
                    "gate": {"experimental_blend_percent": 25},
                },
                {
                    "cycle": 2,
                    "quant_path": "legacy.nnue",
                    "gate": {"experimental_blend_percent": 75},
                },
                {
                    "cycle": 3,
                    "quant_path": "different.nnue",
                    "gate": {"experimental_blend_percent": 100},
                },
            ],
        }

        self.assertEqual(75, autopilot._active_model_blend_percent(state))
        state["active_model_blend_percent"] = 50
        self.assertEqual(50, autopilot._active_model_blend_percent(state))

    def test_different_runtime_identity_restarts_candidate_blend_ramp(self) -> None:
        legacy = {
            "quant_format": "PIENNQ01",
            "quant_version": 1,
            "input_dim": 40_960,
            "hidden_dim": 64,
            "output_dim": 1,
        }
        v2 = {
            **legacy,
            "input_dim": 81_920,
            "feature_set": "halfkp-all-pieces-v2",
        }
        state = {
            "active_model_path": "legacy.nnue",
            "active_model_blend_percent": 75,
            "active_model_identity": legacy,
            "accepted_models": [{}, {}, {}],
        }

        self.assertEqual(
            25,
            autopilot._candidate_model_blend_percent(
                state,
                candidate_identity=v2,
            ),
        )
        self.assertEqual(
            100,
            autopilot._candidate_model_blend_percent(
                state,
                candidate_identity=legacy,
            ),
        )

    def test_different_target_objective_restarts_same_architecture_blend_ramp(self) -> None:
        runtime = {
            "quant_format": "PIENNQ01",
            "quant_version": 1,
            "input_dim": 81_920,
            "hidden_dim": 64,
            "output_dim": 1,
            "feature_set": "halfkp-all-pieces-v2",
        }
        old_identity = {
            **runtime,
            "target_schema": "legacy-hard-wdl-v1",
            "objective": {"schema": "legacy-objective-v1", "loss_kind": "wdl"},
        }
        corrected_identity = {
            **runtime,
            "target_schema": "soft-cp-wdl-v2",
            "objective": {
                "schema": "nnue-objective-v1",
                "target_schema": "soft-cp-wdl-v2",
                "loss_kind": "wdl",
            },
        }
        state = {
            "active_model_path": "old.nnue",
            "active_model_blend_percent": 75,
            "active_model_identity": old_identity,
            "accepted_models": [{}, {}, {}],
        }

        self.assertFalse(
            autopilot._model_identities_same(old_identity, corrected_identity)
        )
        self.assertFalse(
            autopilot._model_identities_same(runtime, corrected_identity)
        )
        self.assertEqual(
            25,
            autopilot._candidate_model_blend_percent(
                state,
                candidate_identity=corrected_identity,
            ),
        )

    def test_legacy_deployment_state_migration_preserves_exact_active_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            active = _write_fake_quant(Path(tmp) / "legacy.nnue")
            active_sha = hashlib.sha256(active.read_bytes()).hexdigest()
            state = {
                "active_model_path": str(active),
                "accepted_models": [
                    {
                        "cycle": 49,
                        "quant_path": "older-25.nnue",
                        "gate": {"experimental_blend_percent": 25},
                    },
                    {
                        "cycle": 55,
                        "quant_path": "older-50.nnue",
                        "gate": {"experimental_blend_percent": 50},
                    },
                    {
                        "cycle": 57,
                        "quant_path": str(active),
                        "quant_sha256": active_sha,
                        "gate": {"experimental_blend_percent": 75},
                    },
                ],
            }

            self.assertTrue(autopilot._migrate_deployment_state(state))
            self.assertEqual(75, state["active_model_blend_percent"])
            self.assertEqual(active_sha, state["active_model_sha256"])
            self.assertEqual(40_960, state["active_model_identity"]["input_dim"])
            self.assertEqual(64, state["active_model_identity"]["hidden_dim"])
            self.assertEqual(3, len(state["accepted_models"]))
            self.assertFalse(autopilot._migrate_deployment_state(state))

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
                checkpoint = _write_fake_checkpoint(out_dir)
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                }

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
                            "--gate-confirmation-games",
                            "0",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(2, len(calls))
            self.assertEqual(3, len(gate_calls))
            first_quant = Path(calls[0]["out_dir"]) / "nnue_quant.nnue"
            self.assertEqual(first_quant, calls[1]["selfplay_nnue_quant_file"])
            self.assertEqual(first_quant, calls[1]["teacher_relabel_nnue_quant_file"])
            self.assertEqual(25, calls[1]["selfplay_nnue_blend_percent"])
            self.assertEqual(25, calls[1]["teacher_relabel_nnue_blend_percent"])
            self.assertIsNone(gate_calls[0][0])
            self.assertEqual(first_quant, gate_calls[1][0])
            self.assertIsNone(gate_calls[2][0], "every promotion gets a pure-PST gate")

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

    def test_replay_lineage_floor_excludes_old_data_without_removing_audit_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = []
            paths = []
            for cycle in range(1, 5):
                path = Path(tmp) / f"cycle-{cycle}" / "jsonl"
                path.mkdir(parents=True)
                paths.append(path)
                completed.append({"cycle": cycle, "jsonl_dir": str(path)})
            state = {
                "training_lineage_start_cycle": 3,
                "completed_cycles": completed,
            }

            self.assertEqual(
                [paths[3], paths[2]],
                autopilot._collect_replay_jsonl_dirs(state, 4),
            )
            self.assertEqual([1, 2, 3, 4], [c["cycle"] for c in completed])

    def test_validation_partition_migration_cuts_replay_without_resetting_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            completed = []
            for cycle in range(1, 5):
                path = Path(tmp) / f"cycle-{cycle}" / "jsonl"
                path.mkdir(parents=True)
                paths.append(path)
                completed.append({"cycle": cycle, "jsonl_dir": str(path)})
            state = {
                "next_cycle": 5,
                "training_lineage_start_cycle": 1,
                "training_checkpoint_path": "/live/cycle-4/train/checkpoint.json",
                "training_checkpoint_sha256": "checkpoint-sha",
                "training_model_identity": {"input_dim": 81_920},
                "completed_cycles": completed,
                "accepted_models": [],
            }

            self.assertTrue(autopilot._migrate_deployment_state(state))

            self.assertEqual(
                autopilot.run_pipeline.train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
                state["validation_partition_schema"],
            )
            self.assertEqual(5, state["validation_partition_start_cycle"])
            self.assertEqual(
                "/live/cycle-4/train/checkpoint.json",
                state["training_checkpoint_path"],
            )
            self.assertEqual("checkpoint-sha", state["training_checkpoint_sha256"])
            self.assertEqual({"input_dim": 81_920}, state["training_model_identity"])
            self.assertEqual([], autopilot._collect_replay_jsonl_dirs(state, 6))
            migration = state["validation_partition_migration"]
            self.assertEqual(5, migration["start_cycle"])
            self.assertEqual(
                "/live/cycle-4/train/checkpoint.json",
                migration["preserved_checkpoint_path"],
            )

            clean_path = Path(tmp) / "cycle-5" / "jsonl"
            clean_path.mkdir(parents=True)
            state["completed_cycles"].append(
                {"cycle": 5, "jsonl_dir": str(clean_path)}
            )
            state["next_cycle"] = 6
            self.assertEqual(
                [clean_path],
                autopilot._collect_replay_jsonl_dirs(state, 6),
            )
            self.assertFalse(autopilot._migrate_deployment_state(state))

    def test_atomic_training_lineage_reset_clears_warm_start_but_keeps_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "autopilot_state.json"
            state = {
                "next_cycle": 77,
                "training_lineage_start_cycle": 1,
                "training_checkpoint_path": "/old/checkpoint.json",
                "training_checkpoint_sha256": "old-sha",
                "training_model_identity": {"input_dim": 40_960},
                "completed_cycles": [{"cycle": 1}, {"cycle": 76}],
                "accepted_models": [{"cycle": 57}],
            }

            reset, changed = autopilot._atomic_reset_training_lineage(
                state_path=state_path,
                state=state,
                start_cycle=77,
            )

            self.assertTrue(changed)
            self.assertEqual(77, reset["training_lineage_start_cycle"])
            self.assertIsNone(reset["training_checkpoint_path"])
            self.assertIsNone(reset["training_checkpoint_sha256"])
            self.assertIsNone(reset["training_model_identity"])
            self.assertEqual([1, 76], [c["cycle"] for c in reset["completed_cycles"]])
            self.assertEqual([57], [m["cycle"] for m in reset["accepted_models"]])
            self.assertEqual("/old/checkpoint.json", reset["training_lineage_reset"]["prior_checkpoint_path"])
            self.assertEqual(reset, json.loads(state_path.read_text()))

            repeated, repeated_changed = autopilot._atomic_reset_training_lineage(
                state_path=state_path,
                state=reset,
                start_cycle=77,
            )
            self.assertFalse(repeated_changed)
            self.assertEqual(reset, repeated)

            progressed = dict(reset)
            progressed["next_cycle"] = 78
            progressed["training_checkpoint_path"] = "/new/checkpoint.json"
            progressed["training_model_identity"] = {"input_dim": 81_920}
            restarted, restarted_changed = autopilot._atomic_reset_training_lineage(
                state_path=state_path,
                state=progressed,
                start_cycle=77,
            )
            self.assertFalse(restarted_changed)
            self.assertEqual("/new/checkpoint.json", restarted["training_checkpoint_path"])

    def test_training_lineage_floor_and_checkpoint_identity_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_lineage_start_cycle"):
            autopilot._validate_training_lineage_floor(
                {"next_cycle": 5, "training_lineage_start_cycle": 0}
            )
        with self.assertRaisesRegex(ValueError, "next_cycle"):
            autopilot._validate_training_lineage_floor(
                {"next_cycle": 5, "training_lineage_start_cycle": 6}
            )

        state = {
            "training_checkpoint_path": "/legacy/checkpoint.json",
            "training_model_identity": {
                "input_dim": 40_960,
                "hidden_dim": 64,
                "feature_set": "halfkp-own-pieces-v1",
            },
        }
        with self.assertRaisesRegex(ValueError, "reset-training-lineage"):
            autopilot._validate_training_checkpoint_identity(
                state,
                {
                    "training_input_dim": 81_920,
                    "hidden_dim": 64,
                    "training_feature_set": "halfkp-all-pieces-v2",
                },
            )

        current_architecture_broken_objective = {
            "training_checkpoint_path": "/broken-wdl/checkpoint.json",
            "training_model_identity": {
                "input_dim": 81_920,
                "hidden_dim": 64,
                "feature_set": "halfkp-all-pieces-v2",
                "target_schema": "legacy-wdl-target-v1",
                "objective": {"schema": "old-objective"},
            },
            "next_cycle": 77,
        }
        defaults = autopilot.zen5_9755_7d_profile()
        with self.assertRaisesRegex(ValueError, "target_schema|objective"):
            autopilot._validate_training_checkpoint_identity(
                current_architecture_broken_objective,
                defaults,
            )

    def test_main_applies_explicit_lineage_reset_before_checkpoint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            out_root.mkdir()
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 0.0,
                "next_cycle": 77,
                "training_lineage_start_cycle": 1,
                "training_checkpoint_path": "/legacy/checkpoint.json",
                "training_checkpoint_sha256": "legacy-sha",
                "training_model_identity": {
                    "input_dim": 40_960,
                    "hidden_dim": 64,
                    "feature_set": "halfkp-own-pieces-v1",
                },
                "completed_cycles": [{"cycle": 76}],
                "accepted_models": [],
                "active_model_path": None,
                "last_error": None,
            }
            (out_root / "autopilot_state.json").write_text(json.dumps(state), encoding="utf-8")

            rc = autopilot.main(
                [
                    "--out-root",
                    str(out_root),
                    "--reset-training-lineage-at-cycle",
                    "77",
                ]
            )

            self.assertEqual(0, rc)
            loaded = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(77, loaded["training_lineage_start_cycle"])
            self.assertIsNone(loaded["training_checkpoint_path"])
            self.assertIsNone(loaded["training_model_identity"])
            self.assertEqual([76], [c["cycle"] for c in loaded["completed_cycles"]])

    def test_training_checkpoint_resolver_migrates_legacy_state_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "cycle" / "train" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            legacy_state = {
                "completed_cycles": [{"cycle": 1, "out_dir": str(root / "missing")}],
                "last_summary": {"checkpoint_path": str(checkpoint)},
            }
            self.assertEqual(
                checkpoint,
                autopilot._resolve_training_checkpoint_path(legacy_state, None),
            )

            verified_state = {
                "training_checkpoint_path": str(checkpoint),
                "training_checkpoint_sha256": checkpoint_sha,
            }
            self.assertEqual(
                checkpoint,
                autopilot._resolve_training_checkpoint_path(verified_state, None),
            )
            checkpoint.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                autopilot._resolve_training_checkpoint_path(verified_state, None)

            missing = root / "gone" / "checkpoint.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                autopilot._resolve_training_checkpoint_path(
                    {"training_checkpoint_path": str(missing)},
                    None,
                )

    def test_explicit_null_training_checkpoint_starts_fresh_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_checkpoint = Path(tmp) / "old" / "train" / "checkpoint.json"
            legacy_checkpoint.parent.mkdir(parents=True)
            legacy_checkpoint.write_text("{}", encoding="utf-8")
            state = {
                "training_checkpoint_path": None,
                "completed_cycles": [
                    {
                        "cycle": 1,
                        "checkpoint_path": str(legacy_checkpoint),
                    }
                ],
                "last_summary": {"checkpoint_path": str(legacy_checkpoint)},
            }

            self.assertIsNone(
                autopilot._resolve_training_checkpoint_path(
                    state,
                    legacy_checkpoint,
                )
            )

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

    def test_teacher_lag_selects_older_model_and_its_promoted_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m1 = Path(tmp) / "m1.nnue"
            m2 = Path(tmp) / "m2.nnue"
            m3 = Path(tmp) / "m3.nnue"
            m1.write_bytes(b"PIENNQ01dummy")
            m2.write_bytes(b"PIENNQ01dummy")
            m3.write_bytes(b"PIENNQ01dummy")
            state = {
                "accepted_models": [
                    {
                        "cycle": 1,
                        "quant_path": str(m1),
                        "gate": {"experimental_blend_percent": 25},
                    },
                    {
                        "cycle": 2,
                        "quant_path": str(m2),
                        "gate": {"experimental_blend_percent": 50},
                    },
                    {
                        "cycle": 3,
                        "quant_path": str(m3),
                        "gate": {"experimental_blend_percent": 75},
                    },
                ]
            }
            self.assertEqual(
                (m2, 50),
                autopilot._resolve_teacher_quant_and_blend(state, 1),
            )
            self.assertEqual(
                (m1, 25),
                autopilot._resolve_teacher_quant_and_blend(state, 2),
            )
            self.assertEqual(m2, autopilot._resolve_teacher_quant_path(state, 1))

    def test_current_state_schema_does_not_fallback_to_last_summary_when_no_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")
            state = {
                "active_model_path": None,
                "last_summary": {"quant_path": str(candidate)},
            }
            self.assertIsNone(autopilot._resolve_active_quant_path(state))

    def test_active_model_resolution_fails_closed_on_missing_or_changed_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.nnue"
            with self.assertRaisesRegex(ValueError, "active model is missing"):
                autopilot._resolve_active_quant_path(
                    {"active_model_path": str(missing), "accepted_models": []}
                )

            active = root / "active.nnue"
            active.write_bytes(b"PIENNQ01original")
            expected_sha = hashlib.sha256(active.read_bytes()).hexdigest()
            state = {
                "active_model_path": str(active),
                "accepted_models": [
                    {"quant_path": str(active), "quant_sha256": expected_sha}
                ],
            }
            self.assertEqual(active, autopilot._resolve_active_quant_path(state))
            active.write_bytes(b"PIENNQ01changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                autopilot._resolve_active_quant_path(state)

    def test_bootstrap_reject_keeps_default_engine_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            created = []

            def _fake_run_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(f"PIENNQ01dummy-{len(created)}".encode())
                created.append((kwargs, quant_path))
                checkpoint = _write_fake_checkpoint(out_dir)
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                    "jsonl_dir": str(out_dir / "jsonl_relabel"),
                }

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
                            "--gate-confirmation-games",
                            "0",
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

    def test_rejected_candidate_warm_starts_the_next_training_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            calls = []
            gate_bases = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                train_dir = out_dir / "train"
                train_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = train_dir / "checkpoint.json"
                checkpoint_path.write_text(
                    json.dumps({"cycle": len(calls)}),
                    encoding="utf-8",
                )
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(f"PIENNQ01dummy-{len(calls)}".encode())
                return {
                    "checkpoint_path": str(checkpoint_path),
                    "quant_path": str(quant_path),
                    "jsonl_dir": str(out_dir / "jsonl_relabel"),
                }

            def _reject_gate(*, base_quant, **_kwargs):
                gate_bases.append(base_quant)
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
                    side_effect=_reject_gate,
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
            first_checkpoint = Path(calls[0]["out_dir"]) / "train" / "checkpoint.json"
            second_checkpoint = Path(calls[1]["out_dir"]) / "train" / "checkpoint.json"
            self.assertIsNone(calls[0]["initial_checkpoint"])
            self.assertEqual(first_checkpoint, calls[1]["initial_checkpoint"])
            self.assertEqual(0.003, calls[0]["learning_rate"])
            self.assertEqual(0.001, calls[1]["learning_rate"])
            self.assertEqual([None, None], gate_bases)
            self.assertIsNone(calls[1]["selfplay_nnue_quant_file"])
            self.assertIsNone(calls[1]["teacher_relabel_nnue_quant_file"])

            state = json.loads((out_root / "autopilot_state.json").read_text(encoding="utf-8"))
            self.assertIsNone(state["active_model_path"])
            self.assertEqual(str(second_checkpoint), state["training_checkpoint_path"])

    def test_explicit_initial_checkpoint_bootstraps_first_training_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            initial_checkpoint = root / "bootstrap-checkpoint.json"
            initial_checkpoint.write_text(json.dumps({"bootstrap": True}), encoding="utf-8")
            calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                train_dir = out_dir / "train"
                train_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = train_dir / "checkpoint.json"
                checkpoint_path.write_text(json.dumps({"cycle": 1}), encoding="utf-8")
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                return {
                    "checkpoint_path": str(checkpoint_path),
                    "quant_path": str(quant_path),
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    return_value={"accepted": False},
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--hours",
                            "1",
                            "--max-cycles",
                            "1",
                            "--initial-checkpoint",
                            str(initial_checkpoint),
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(initial_checkpoint, calls[0]["initial_checkpoint"])

    def test_external_checkpoint_is_weights_only_once_then_optimizer_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            external_checkpoint = root / "bootstrap-checkpoint.json"
            external_checkpoint.write_text(
                json.dumps({"external": True}),
                encoding="utf-8",
            )
            calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                checkpoint = _write_fake_checkpoint(out_dir)
                optimizer = checkpoint.parent / "optimizer.pt"
                optimizer.write_bytes(f"optimizer-{len(calls)}".encode())
                quant = _write_fake_quant(
                    out_dir / "nnue_quant.nnue",
                    input_dim=81_920,
                    marker=f"candidate-{len(calls)}".encode(),
                )
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant),
                }

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
                        "--initial-checkpoint",
                        str(external_checkpoint),
                        "--initial-checkpoint-weights-only",
                    ]
                )

            self.assertEqual(0, rc)
            self.assertEqual(2, len(calls))
            first_candidate = Path(calls[0]["out_dir"]) / "train" / "checkpoint.json"
            self.assertEqual(external_checkpoint, calls[0]["initial_checkpoint"])
            self.assertTrue(calls[0]["initial_checkpoint_weights_only"])
            self.assertFalse(calls[0]["continue_optimizer_state"])
            self.assertIsNone(calls[0].get("initial_optimizer_state"))
            self.assertEqual(first_candidate, calls[1]["initial_checkpoint"])
            self.assertFalse(calls[1]["initial_checkpoint_weights_only"])
            self.assertTrue(calls[1]["continue_optimizer_state"])

    def test_initial_active_model_seeds_cycle_one_selfplay_and_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            initial_active = _write_fake_quant(
                root / "initial-active.nnue",
                input_dim=81_920,
                marker=b"bootstrap",
            )
            initial_sha = hashlib.sha256(initial_active.read_bytes()).hexdigest()
            calls = []
            state_writes = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                checkpoint = _write_fake_checkpoint(out_dir)
                quant = _write_fake_quant(
                    out_dir / "nnue_quant.nnue",
                    input_dim=81_920,
                    marker=b"candidate",
                )
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant),
                }

            real_atomic_write = autopilot._atomic_write_json

            def _record_atomic_write(path, value):
                if Path(path).name == "autopilot_state.json":
                    state_writes.append(json.loads(json.dumps(value)))
                return real_atomic_write(path, value)

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._atomic_write_json",
                    side_effect=_record_atomic_write,
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--hours",
                            "1",
                            "--max-cycles",
                            "1",
                            "--gate-games",
                            "0",
                            "--initial-active-model",
                            str(initial_active),
                            "--initial-active-model-blend-percent",
                            "40",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(1, len(calls))
            resolved_initial = initial_active.resolve()
            self.assertEqual(resolved_initial, calls[0]["selfplay_nnue_quant_file"])
            self.assertEqual(resolved_initial, calls[0]["teacher_relabel_nnue_quant_file"])
            self.assertEqual(40, calls[0]["selfplay_nnue_blend_percent"])
            self.assertEqual(40, calls[0]["teacher_relabel_nnue_blend_percent"])
            self.assertGreater(len(state_writes), 0)
            self.assertEqual(
                initial_active.resolve().as_posix(),
                state_writes[0]["active_model_path"],
            )
            self.assertEqual(initial_sha, state_writes[0]["active_model_sha256"])
            self.assertEqual(40, state_writes[0]["active_model_blend_percent"])
            self.assertEqual(
                initial_sha,
                state_writes[0]["initial_active_model"]["sha256"],
            )
            state = json.loads(
                (out_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            initial_metadata = state["initial_active_model"]
            self.assertEqual(resolved_initial.as_posix(), initial_metadata["path"])
            self.assertEqual(initial_sha, initial_metadata["sha256"])
            self.assertEqual(40, initial_metadata["blend_percent"])
            self.assertEqual("PIENNQ01", initial_metadata["model_identity"]["quant_format"])

    def test_initial_active_model_validation_is_strict_but_not_current_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = _write_fake_quant(root / "initial.nnue", marker=b"initial")
            other = _write_fake_quant(root / "other.nnue", marker=b"other")
            invalid = root / "invalid.nnue"
            invalid.write_bytes(b"not-a-quant-model")

            with self.assertRaisesRegex(ValueError, "without.*model"):
                autopilot._initial_active_model_metadata(None, 25)
            for blend in (-1, 101):
                with self.subTest(blend=blend), self.assertRaisesRegex(
                    ValueError,
                    "between 0 and 100",
                ):
                    autopilot._initial_active_model_metadata(initial, blend)
            with self.assertRaisesRegex(ValueError, "missing"):
                autopilot._initial_active_model_metadata(root / "missing.nnue", 25)
            with self.assertRaisesRegex(ValueError, "quantized model format"):
                autopilot._initial_active_model_metadata(invalid, 25)

            configured = autopilot._initial_active_model_metadata(initial, 25)
            advanced_identity = autopilot._quant_model_identity(other)
            state = {
                "initial_active_model": configured,
                "active_model_path": str(other),
                "active_model_sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
                "active_model_blend_percent": 50,
                "active_model_identity": advanced_identity,
            }
            self.assertFalse(
                autopilot._initialize_or_validate_initial_active_model(
                    state,
                    initial_active_model=initial,
                    blend_percent=25,
                    fresh=False,
                )
            )
            self.assertEqual(str(other), state["active_model_path"])
            with self.assertRaisesRegex(ValueError, "bootstrap identity"):
                autopilot._initialize_or_validate_initial_active_model(
                    state,
                    initial_active_model=other,
                    blend_percent=25,
                    fresh=False,
                )

    def test_unchanged_training_checkpoint_is_not_repeatedly_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            gate_calls = []
            quant_bytes = b"PIENNQ01same"
            quant_sha = hashlib.sha256(quant_bytes).hexdigest()

            def _fake_run_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _write_fake_checkpoint(out_dir)
                checkpoint.write_text(json.dumps({"same": True}), encoding="utf-8")
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                quant = out_dir / "nnue_quant.nnue"
                quant.write_bytes(quant_bytes)
                return {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "quant_path": str(quant),
                    "quant_sha256": quant_sha,
                }

            def _fake_gate(**kwargs):
                gate_calls.append(kwargs)
                return {
                    "accepted": False,
                    "baseline_points": 7.0,
                    "experimental_points": 5.0,
                    "delta_points": -2.0,
                    "games": 12,
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
            state = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(quant_sha, state["last_gated_quant_sha256"])
            self.assertEqual("unchanged-training-checkpoint", state["last_gate"]["reason"])

    def test_saturated_active_model_is_not_reaccepted_when_candidate_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            out_root.mkdir()
            parent_checkpoint = root / "parent-checkpoint.json"
            parent_checkpoint.write_text(json.dumps({"parent": True}), encoding="utf-8")
            active_quant = root / "active.nnue"
            active_quant.write_bytes(b"PIENNQ01identical-model")
            quant_sha = hashlib.sha256(active_quant.read_bytes()).hexdigest()
            completed = [
                {
                    "cycle": cycle,
                    "out_dir": str(root / f"old-cycle-{cycle}"),
                    "checkpoint_path": str(parent_checkpoint) if cycle == 4 else None,
                    "gate": {"accepted": True},
                }
                for cycle in range(1, 5)
            ]
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 10**12,
                "next_cycle": 5,
                "completed_cycles": completed,
                "accepted_models": [
                    {
                        "cycle": cycle,
                        "quant_path": str(active_quant),
                        "quant_sha256": quant_sha,
                    }
                    for cycle in range(1, 5)
                ],
                "active_model_path": str(active_quant),
                "training_checkpoint_path": str(parent_checkpoint),
                "last_error": None,
            }
            (out_root / "autopilot_state.json").write_text(json.dumps(state), encoding="utf-8")

            def _fake_run_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _write_fake_checkpoint(out_dir)
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                quant = out_dir / "nnue_quant.nnue"
                quant.write_bytes(active_quant.read_bytes())
                return {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "quant_path": str(quant),
                    "quant_sha256": quant_sha,
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    side_effect=AssertionError("identical active model must not be re-gated"),
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--max-cycles",
                            "5",
                            "--retry-limit",
                            "1",
                            "--retry-backoff-sec",
                            "0",
                        ]
                    )

            self.assertEqual(0, rc)
            loaded = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(4, len(loaded["accepted_models"]))
            self.assertEqual("candidate-identical-to-active-model", loaded["last_gate"]["reason"])

    def test_failed_completion_commit_retries_without_duplicate_or_self_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _write_fake_checkpoint(out_dir)
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                quant = out_dir / "nnue_quant.nnue"
                quant.write_bytes(b"PIENNQ01transaction")
                quant_sha = hashlib.sha256(quant.read_bytes()).hexdigest()
                return {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "quant_path": str(quant),
                    "quant_sha256": quant_sha,
                }

            real_atomic_write = autopilot._atomic_write_json
            failed_once = False

            def _fail_first_completion(path, payload):
                nonlocal failed_once
                if (
                    not failed_once
                    and len(payload.get("completed_cycles", [])) == 1
                    and payload.get("last_error") is None
                ):
                    failed_once = True
                    raise OSError("simulated completion commit failure")
                return real_atomic_write(path, payload)

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._atomic_write_json",
                    side_effect=_fail_first_completion,
                ):
                    with mock.patch("training.nnue.autopilot.time.sleep"):
                        rc = autopilot.main(
                            [
                                "--out-root",
                                str(out_root),
                                "--hours",
                                "1",
                                "--max-cycles",
                                "1",
                                "--gate-games",
                                "0",
                                "--retry-limit",
                                "2",
                                "--retry-backoff-sec",
                                "0",
                            ]
                        )

            self.assertEqual(0, rc)
            self.assertTrue(failed_once)
            self.assertEqual(2, len(calls))
            self.assertIsNone(calls[0]["initial_checkpoint"])
            self.assertIsNone(calls[1]["initial_checkpoint"])
            state = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual([1], [cycle["cycle"] for cycle in state["completed_cycles"]])
            self.assertEqual([], state["accepted_models"])
            self.assertEqual(
                "gate-disabled-promotion-ineligible", state["last_gate"]["reason"]
            )
            self.assertEqual(2, state["next_cycle"])

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
                    json.dumps(_paired_gate_payload([2.0, 2.0, 2.0])),
                    encoding="utf-8",
                )

            with mock.patch("training.nnue.autopilot.subprocess.run", side_effect=_fake_run):
                gate = autopilot._run_model_gate(
                    piebot_dir=root,
                    out_json=out_json,
                    base_quant=candidate,
                    candidate_quant=candidate,
                    games=6,
                    movetime_ms=25,
                    noise_plies=4,
                    noise_topk=3,
                    threads=1,
                    seed=9,
                    min_score_delta=0.0,
                    base_blend_percent=25,
                    candidate_blend_percent=50,
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
            self.assertEqual("25", cmd[cmd.index("--base-blend") + 1])
            self.assertEqual("50", cmd[cmd.index("--exp-blend") + 1])
            self.assertIn("--paired-openings", cmd)
            self.assertEqual(str(root), kwargs["cwd"])
            self.assertTrue(kwargs["check"])

    def test_gate_statistics_require_complete_game_level_pairs(self) -> None:
        payload = _paired_gate_payload([2.0, 2.0])
        without_games = dict(payload)
        without_games.pop("game_results")
        status = autopilot._paired_gate_statistics(
            without_games,
            confidence_level=0.95,
            bootstrap_samples=2_000,
            seed=17,
            minimum_mean_pair_delta=0.0,
        )
        self.assertFalse(status["eligible"])
        self.assertFalse(status["accepted"])
        self.assertEqual("missing-game-level-evidence", status["reason"])

        incomplete = _paired_gate_payload([2.0, 2.0])
        incomplete["game_results"].pop()
        with self.assertRaisesRegex(ValueError, "complete.*pair"):
            autopilot._paired_gate_statistics(
                incomplete,
                confidence_level=0.95,
                bootstrap_samples=2_000,
                seed=17,
                minimum_mean_pair_delta=0.0,
            )

    def test_paired_bootstrap_is_reproducible_and_requires_positive_lower_bound(self) -> None:
        strong = _paired_gate_payload([2.0] * 12)
        first = autopilot._paired_gate_statistics(
            strong,
            confidence_level=0.95,
            bootstrap_samples=2_000,
            seed=99,
            minimum_mean_pair_delta=0.0,
        )
        second = autopilot._paired_gate_statistics(
            strong,
            confidence_level=0.95,
            bootstrap_samples=2_000,
            seed=99,
            minimum_mean_pair_delta=0.0,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["eligible"])
        self.assertTrue(first["accepted"])
        self.assertGreater(first["confidence_interval"]["lower"], 0.0)

        inconclusive = autopilot._paired_gate_statistics(
            _paired_gate_payload([2.0, -2.0] * 8),
            confidence_level=0.95,
            bootstrap_samples=2_000,
            seed=99,
            minimum_mean_pair_delta=0.0,
        )
        self.assertFalse(inconclusive["accepted"])
        self.assertEqual("confidence-lower-bound-not-positive", inconclusive["reason"])

        drawn = autopilot._paired_gate_statistics(
            _paired_gate_payload([0.0] * 4),
            confidence_level=0.95,
            bootstrap_samples=500,
            seed=99,
            minimum_mean_pair_delta=0.0,
        )
        self.assertTrue(drawn["eligible"])
        self.assertFalse(drawn["accepted"])
        self.assertEqual(4, drawn["complete_pairs"])

    def test_model_gate_missing_game_evidence_is_promotion_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "gate.json"
            candidate = _write_fake_quant(root / "candidate.nnue")

            def _fake_run(_cmd, **_kwargs):
                out_json.write_text(
                    json.dumps(
                        {
                            "games": 6,
                            "points": {"baseline": 0.0, "experimental": 6.0},
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
                    movetime_ms=1,
                    noise_plies=0,
                    noise_topk=1,
                    threads=1,
                    seed=1,
                    min_score_delta=0.0,
                )
            self.assertFalse(gate["accepted"])
            self.assertFalse(gate["evidence_eligible"])
            self.assertEqual("missing-game-level-evidence", gate["reason"])

    def test_relative_candidate_must_also_pass_pure_pst_absolute_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = _write_fake_quant(root / "active.nnue", marker=b"active")
            candidate = _write_fake_quant(root / "candidate.nnue", marker=b"candidate")
            calls = []
            results = iter(
                [
                    {"accepted": True, "baseline_kind": "active-model"},
                    {"accepted": True, "baseline_kind": "active-model"},
                    {
                        "accepted": False,
                        "baseline_kind": "pure-pst",
                        "reason": "confidence-lower-bound-not-positive",
                    },
                ]
            )

            def _gate(**kwargs):
                calls.append(kwargs)
                return next(results)

            with mock.patch("training.nnue.autopilot._run_model_gate", side_effect=_gate):
                attempt = autopilot._run_confirmed_gate_attempt(
                    piebot_dir=root,
                    screen_json=root / "screen.json",
                    confirmation_json=root / "confirmation.json",
                    base_quant=active,
                    candidate_quant=candidate,
                    screen_games=24,
                    confirmation_games=96,
                    movetime_ms=100,
                    noise_plies=12,
                    noise_topk=5,
                    threads=1,
                    seed=7,
                    screen_min_score_delta=0.0,
                    confirmation_min_score_delta=0.0,
                    base_blend_percent=25,
                    candidate_blend_percent=50,
                    paired_openings=True,
                )

            self.assertFalse(attempt["accepted"])
            self.assertEqual("absolute-pst-rejected", attempt["reason"])
            self.assertEqual(3, len(calls))
            self.assertIsNone(calls[-1]["base_quant"])
            self.assertEqual("pure-pst", attempt["absolute"]["baseline_kind"])

    def test_validation_provenance_never_calls_piebot_teacher_data_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation"
            validation.mkdir()
            (validation / "shard.jsonl").write_text(
                json.dumps({"fen": "startpos", "teacher_depth": 6}) + "\n",
                encoding="utf-8",
            )
            piebot_metadata = root / "piebot-validation.json"
            piebot_metadata.write_text(
                json.dumps(
                    {
                        "schema": "piebot-validation-provenance-v1",
                        "independent_of_piebot": False,
                        "source": {"kind": "piebot-relabel", "name": "PieBot"},
                    }
                ),
                encoding="utf-8",
            )
            circular = autopilot._validation_strength_status(
                validation_jsonl_dir=validation,
                provenance_json=piebot_metadata,
            )
            self.assertFalse(circular["absolute_strength_eligible"])
            self.assertEqual("circular-piebot-validation", circular["reason"])

            missing = autopilot._validation_strength_status(
                validation_jsonl_dir=validation,
                provenance_json=None,
            )
            self.assertFalse(missing["absolute_strength_eligible"])
            self.assertEqual("validation-provenance-unverified", missing["reason"])

    def test_independent_validation_and_external_anchor_metadata_are_bound_by_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation"
            validation.mkdir()
            (validation / "shard.jsonl").write_text(
                json.dumps({"fen": "startpos", "teacher_depth": 18}) + "\n",
                encoding="utf-8",
            )
            dataset_sha = autopilot._jsonl_dataset_sha256(validation)
            metadata = root / "validation.json"
            metadata.write_text(
                json.dumps(
                    {
                        "schema": "piebot-validation-provenance-v1",
                        "independent_of_piebot": True,
                        "dataset_sha256": dataset_sha,
                        "source": {
                            "kind": "stockfish",
                            "name": "Stockfish 17",
                            "binary_sha256": "a" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            status = autopilot._validation_strength_status(
                validation_jsonl_dir=validation,
                provenance_json=metadata,
            )
            self.assertTrue(status["absolute_strength_eligible"])
            self.assertEqual("independent-validation-verified", status["reason"])

            candidate_sha = "b" * 64
            anchor = root / "anchor.json"
            anchor.write_text(
                json.dumps(
                    {
                        "schema": "piebot-uci-elo-arena-v1",
                        "config": {
                            "piebot": {"model_sha256": candidate_sha},
                            "stockfish": {"options": {"UCI_Elo": 2200}},
                        },
                        "games": [
                            {"game_index": 0, "pair_index": 0, "piebot_score": 1.0},
                            {"game_index": 1, "pair_index": 0, "piebot_score": 0.5},
                        ],
                        "summary": {"complete_pairs": 1, "games": 2},
                    }
                ),
                encoding="utf-8",
            )
            verified = autopilot._external_anchor_status(
                anchor_json=anchor,
                candidate_sha256=candidate_sha,
            )
            self.assertTrue(verified["eligible"])
            mismatch = autopilot._external_anchor_status(
                anchor_json=anchor,
                candidate_sha256="c" * 64,
            )
            self.assertFalse(mismatch["eligible"])
            self.assertEqual("external-anchor-model-sha-mismatch", mismatch["reason"])

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

    def test_paired_model_gate_rejects_odd_game_count_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")

            with mock.patch("training.nnue.autopilot.subprocess.run") as runner:
                with self.assertRaisesRegex(ValueError, "even"):
                    autopilot._run_model_gate(
                        piebot_dir=root,
                        out_json=root / "gate.json",
                        base_quant=None,
                        candidate_quant=candidate,
                        games=3,
                        movetime_ms=1,
                        noise_plies=0,
                        noise_topk=1,
                        threads=1,
                        seed=1,
                        min_score_delta=0.0,
                        paired_openings=True,
                    )
            runner.assert_not_called()

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

    def test_passing_screen_still_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = _write_fake_quant(root / "candidate.nnue")
            calls = []
            results = iter(
                [
                    {"accepted": True, "delta_points": 1.0, "games": 24},
                    {"accepted": False, "delta_points": 0.0, "games": 96},
                ]
            )

            def _gate(**kwargs):
                calls.append(kwargs)
                return next(results)

            with mock.patch(
                "training.nnue.autopilot._run_model_gate",
                side_effect=_gate,
            ):
                attempt = autopilot._run_confirmed_gate_attempt(
                    piebot_dir=root,
                    screen_json=root / "screen.json",
                    confirmation_json=root / "confirmation.json",
                    base_quant=None,
                    candidate_quant=quant,
                    screen_games=24,
                    confirmation_games=96,
                    movetime_ms=100,
                    noise_plies=12,
                    noise_topk=5,
                    threads=1,
                    seed=7,
                    screen_min_score_delta=0.0,
                    confirmation_min_score_delta=2.0,
                    base_blend_percent=0,
                    candidate_blend_percent=25,
                    paired_openings=True,
                )

            self.assertFalse(attempt["accepted"])
            self.assertEqual("confirmation-rejected", attempt["reason"])
            self.assertTrue(attempt["screen"]["accepted"])
            self.assertFalse(attempt["confirmation"]["accepted"])
            self.assertEqual([24, 96], [call["games"] for call in calls])
            self.assertEqual([0.0, 2.0], [call["min_score_delta"] for call in calls])
            self.assertTrue(all(call["paired_openings"] for call in calls))

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
                checkpoint = _write_fake_checkpoint(out_dir)
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                    "jsonl_dir": str(out_dir / "jsonl_relabel"),
                }

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
                            "--gate-confirmation-games",
                            "0",
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
                checkpoint = _write_fake_checkpoint(out_dir)
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                    "jsonl_dir": str(out_dir / "jsonl_relabel"),
                }

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
                            "--teacher-lag-cycles",
                            "1",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(3, len(created))
            self.assertEqual(25, created[1][0]["selfplay_nnue_blend_percent"])
            self.assertEqual(50, created[2][0]["selfplay_nnue_blend_percent"])
            self.assertEqual(25, created[2][0]["teacher_relabel_nnue_blend_percent"])

    def test_cross_lineage_candidate_starts_at_25_and_keeps_legacy_active_on_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            out_root.mkdir()
            active = _write_fake_quant(root / "legacy.nnue", input_dim=40_960)
            active_sha = hashlib.sha256(active.read_bytes()).hexdigest()
            parent_checkpoint = _write_fake_checkpoint(root / "parent")
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 10**12,
                "next_cycle": 4,
                "completed_cycles": [{"cycle": n} for n in range(1, 4)],
                "accepted_models": [
                    {
                        "cycle": 1,
                        "quant_path": "old-25.nnue",
                        "gate": {"experimental_blend_percent": 25},
                    },
                    {
                        "cycle": 2,
                        "quant_path": "old-50.nnue",
                        "gate": {"experimental_blend_percent": 50},
                    },
                    {
                        "cycle": 3,
                        "quant_path": str(active),
                        "quant_sha256": active_sha,
                        "gate": {"experimental_blend_percent": 75},
                    },
                ],
                "active_model_path": str(active),
                "training_checkpoint_path": str(parent_checkpoint),
                "last_error": None,
            }
            (out_root / "autopilot_state.json").write_text(json.dumps(state), encoding="utf-8")
            pipeline_calls = []
            gate_calls = []

            def _fake_pipeline(**kwargs):
                pipeline_calls.append(kwargs)
                out_dir = Path(kwargs["out_dir"])
                checkpoint = _write_fake_checkpoint(out_dir)
                quant = _write_fake_quant(
                    out_dir / "nnue_quant.nnue",
                    input_dim=81_920,
                    marker=b"v2",
                )
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant),
                    "metrics": {
                        "feature_set": "halfkp-all-pieces-v2",
                        "input_dim": 81_920,
                        "hidden_dim": 64,
                    },
                }

            def _reject(**kwargs):
                gate_calls.append(kwargs)
                return {
                    "accepted": False,
                    "baseline_points": 8.0,
                    "experimental_points": 4.0,
                    "delta_points": -4.0,
                    "games": 12,
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    side_effect=_reject,
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--max-cycles",
                            "4",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(75, pipeline_calls[0]["selfplay_nnue_blend_percent"])
            self.assertEqual(75, pipeline_calls[0]["teacher_relabel_nnue_blend_percent"])
            self.assertEqual([25], [call["candidate_blend_percent"] for call in gate_calls])
            loaded = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(str(active), loaded["active_model_path"])
            self.assertEqual(75, loaded["active_model_blend_percent"])
            self.assertEqual(40_960, loaded["active_model_identity"]["input_dim"])
            self.assertEqual(
                81_920,
                loaded["last_gate_identity"]["candidate_model_identity"]["input_dim"],
            )
            self.assertEqual([25], loaded["last_gate_identity"]["candidate_blend_percents"])

    def test_same_lineage_ramp_failure_falls_back_to_current_blend_and_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            out_root.mkdir()
            active = _write_fake_quant(
                root / "active-v2.nnue",
                input_dim=81_920,
                marker=b"active",
            )
            active_sha = hashlib.sha256(active.read_bytes()).hexdigest()
            identity = {
                "quant_format": "PIENNQ01",
                "quant_version": 1,
                "input_dim": 81_920,
                "hidden_dim": 64,
                "output_dim": 1,
                "feature_set": "halfkp-all-pieces-v2",
            }
            parent_checkpoint = _write_fake_checkpoint(root / "parent")
            state = {
                "version": 1,
                "profile": "zen5_9755_7d",
                "started_at": 0.0,
                "deadline_ts": 10**12,
                "next_cycle": 2,
                "completed_cycles": [{"cycle": 1}],
                "accepted_models": [
                    {
                        "cycle": 1,
                        "quant_path": str(active),
                        "quant_sha256": active_sha,
                        "blend_percent": 25,
                        "model_identity": identity,
                        "gate": {"experimental_blend_percent": 25},
                    }
                ],
                "active_model_path": str(active),
                "active_model_sha256": active_sha,
                "active_model_blend_percent": 25,
                "active_model_identity": identity,
                "training_checkpoint_path": str(parent_checkpoint),
                "last_error": None,
            }
            (out_root / "autopilot_state.json").write_text(json.dumps(state), encoding="utf-8")
            candidate_paths = []
            gate_calls = []

            def _fake_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                checkpoint = _write_fake_checkpoint(out_dir)
                quant = _write_fake_quant(
                    out_dir / "nnue_quant.nnue",
                    input_dim=81_920,
                    marker=b"improved",
                )
                candidate_paths.append(quant)
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant),
                    "metrics": {
                        "feature_set": "halfkp-all-pieces-v2",
                        "input_dim": 81_920,
                        "hidden_dim": 64,
                    },
                }

            gate_results = iter(
                [
                    {
                        "accepted": False,
                        "baseline_points": 8.0,
                        "experimental_points": 4.0,
                        "delta_points": -4.0,
                        "games": 12,
                    },
                    {
                        "accepted": True,
                        "baseline_points": 5.0,
                        "experimental_points": 7.0,
                        "delta_points": 2.0,
                        "games": 12,
                    },
                    {
                        "accepted": True,
                        "baseline_points": 45.0,
                        "experimental_points": 51.0,
                        "delta_points": 6.0,
                        "games": 96,
                    },
                    {
                        "accepted": True,
                        "baseline_kind": "pure-pst",
                        "baseline_points": 42.0,
                        "experimental_points": 54.0,
                        "delta_points": 12.0,
                        "games": 96,
                    },
                ]
            )

            def _gate(**kwargs):
                gate_calls.append(kwargs)
                return next(gate_results)

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    side_effect=_gate,
                ):
                    rc = autopilot.main(
                        [
                            "--out-root",
                            str(out_root),
                            "--max-cycles",
                            "2",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(
                [50, 25, 25, 25],
                [call["candidate_blend_percent"] for call in gate_calls],
            )
            self.assertEqual("gate_compare.json", Path(gate_calls[0]["out_json"]).name)
            self.assertEqual(
                "gate_compare_same_blend.json",
                Path(gate_calls[1]["out_json"]).name,
            )
            self.assertEqual(
                "gate_compare_same_blend_confirmation.json",
                Path(gate_calls[2]["out_json"]).name,
            )
            self.assertEqual(96, gate_calls[2]["games"])
            self.assertEqual(0.0, gate_calls[2]["min_score_delta"])
            self.assertIsNone(gate_calls[3]["base_quant"])
            self.assertEqual(
                "gate_compare_same_blend_confirmation_absolute_pst.json",
                Path(gate_calls[3]["out_json"]).name,
            )
            loaded = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(str(candidate_paths[0]), loaded["active_model_path"])
            self.assertEqual(25, loaded["active_model_blend_percent"])
            self.assertEqual(identity, loaded["active_model_identity"])
            promoted = loaded["accepted_models"][-1]
            self.assertEqual(25, promoted["blend_percent"])
            self.assertEqual(identity, promoted["model_identity"])
            self.assertEqual(25, promoted["gate"]["experimental_blend_percent"])
            self.assertEqual(2, len(promoted["gate"]["attempts"]))
            self.assertTrue(promoted["gate"]["confirmation"]["accepted"])
            self.assertEqual([50, 25], loaded["last_gate_identity"]["candidate_blend_percents"])

    def test_main_passes_control_loop_options_to_run_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "runs"
            validation_dir = Path(tmp) / "validation"
            validation_dir.mkdir()
            calls = []

            def _control_aware_run_pipeline(
                *,
                out_dir,
                primary_sample_fraction,
                teacher_sample_fraction,
                min_teacher_depth,
                loss_kind,
                huber_delta_cp,
                wdl_scale_cp,
                validation_jsonl_dir,
                max_validation_samples,
                validation_seed,
                validation_require_teacher,
                continue_optimizer_state,
                **_kwargs,
            ):
                calls.append(
                    {
                        "primary_sample_fraction": primary_sample_fraction,
                        "teacher_sample_fraction": teacher_sample_fraction,
                        "min_teacher_depth": min_teacher_depth,
                        "loss_kind": loss_kind,
                        "huber_delta_cp": huber_delta_cp,
                        "wdl_scale_cp": wdl_scale_cp,
                        "validation_jsonl_dir": validation_jsonl_dir,
                        "max_validation_samples": max_validation_samples,
                        "validation_seed": validation_seed,
                        "validation_require_teacher": validation_require_teacher,
                        "continue_optimizer_state": continue_optimizer_state,
                    }
                )
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _write_fake_checkpoint(out_dir)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01control-options")
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                new=_control_aware_run_pipeline,
            ):
                rc = autopilot.main(
                    [
                        "--out-root",
                        str(out_root),
                        "--hours",
                        "1",
                        "--max-cycles",
                        "1",
                        "--gate-games",
                        "0",
                        "--primary-sample-fraction",
                        "0.6",
                        "--teacher-sample-fraction",
                        "0.55",
                        "--min-teacher-depth",
                        "7",
                        "--loss-kind",
                        "huber",
                        "--huber-delta-cp",
                        "80",
                        "--wdl-scale-cp",
                        "350",
                        "--validation-jsonl-dir",
                        str(validation_dir),
                        "--max-validation-samples",
                        "54321",
                        "--validation-seed",
                        "123",
                        "--validation-require-teacher",
                        "--continue-optimizer-state",
                    ]
                )

            self.assertEqual(0, rc)
            self.assertEqual(
                [
                    {
                        "primary_sample_fraction": 0.6,
                        "teacher_sample_fraction": 0.55,
                        "min_teacher_depth": 7,
                        "loss_kind": "huber",
                        "huber_delta_cp": 80.0,
                        "wdl_scale_cp": 350.0,
                        "validation_jsonl_dir": validation_dir,
                        "max_validation_samples": 54_321,
                        "validation_seed": 123,
                        "validation_require_teacher": True,
                        "continue_optimizer_state": True,
                    }
                ],
                calls,
            )

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
                checkpoint = _write_fake_checkpoint(out_dir)
                quant_path = out_dir / "nnue_quant.nnue"
                quant_path.write_bytes(b"PIENNQ01dummy")
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant_path),
                }

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
