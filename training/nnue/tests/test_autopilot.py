import json
import hashlib
import inspect
import math
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
        "parallel_games_requested": 1,
        "parallel_games": 1,
        "parallelism_schema": "bounded-pair-workers-v1",
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
        self.assertEqual(1, profile["gate_parallel_games"])
        self.assertEqual(96, profile["gate_confirmation_games"])
        self.assertEqual(0.0, profile["gate_confirmation_min_score_delta"])
        self.assertEqual(
            "strict-superiority",
            profile["gate_incremental_pst_policy"],
        )
        self.assertEqual(0.0, profile["gate_pst_veto_margin"])
        self.assertEqual(0.95, profile["gate_confidence_level"])
        self.assertGreaterEqual(profile["gate_bootstrap_samples"], 10_000)
        self.assertFalse(profile["gate_require_external_anchor"])
        self.assertIsNone(profile["gate_external_anchor_json"])
        self.assertIsNone(profile["validation_provenance_json"])
        self.assertFalse(profile["initial_checkpoint_weights_only"])
        self.assertIsNone(profile["initial_active_model"])
        self.assertEqual(0, profile["initial_active_model_blend_percent"])


    def test_profile_exposes_selfplay_openings_and_teacher_node_cap(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertIsNone(profile["selfplay_openings"])
        self.assertEqual(0, profile["teacher_relabel_max_nodes"])

    def test_profile_and_cli_expose_adjudication_knobs(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertEqual(900.0, profile["selfplay_resign_cp"])
        self.assertEqual(8, profile["selfplay_resign_plies"])
        self.assertEqual(0.15, profile["selfplay_no_resign_fraction"])
        self.assertEqual(10.0, profile["selfplay_draw_adj_cp"])
        self.assertEqual(40, profile["selfplay_draw_adj_plies"])
        self.assertEqual(80, profile["selfplay_draw_adj_min_ply"])

        args = autopilot._parse_args(
            [
                "--out-root",
                "runs",
                "--selfplay-resign-cp",
                "0",
                "--selfplay-no-resign-fraction",
                "0.5",
                "--selfplay-draw-adj-min-ply",
                "100",
            ]
        )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertEqual(0.0, resolved["selfplay_resign_cp"])
        self.assertEqual(0.5, resolved["selfplay_no_resign_fraction"])
        self.assertEqual(100, resolved["selfplay_draw_adj_min_ply"])

    def test_cli_overrides_selfplay_openings_and_teacher_node_cap(self) -> None:
        args = autopilot._parse_args(
            [
                "--out-root",
                "runs",
                "--selfplay-openings",
                "book/openings.fen",
                "--teacher-relabel-max-nodes",
                "250000",
            ]
        )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertEqual(Path("book/openings.fen"), resolved["selfplay_openings"])
        self.assertEqual(250_000, resolved["teacher_relabel_max_nodes"])

    def test_cli_overrides_selfplay_temperature_moves(self) -> None:
        args = autopilot._parse_args(
            ["--out-root", "runs", "--selfplay-temperature-moves", "12"]
        )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertEqual(12, resolved["selfplay_temperature_moves"])
        bare = autopilot._parse_args(["--out-root", "runs"])
        untouched = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), bare
        )
        self.assertEqual(24, untouched["selfplay_temperature_moves"])

    def test_profile_and_cli_expose_actor_budget_knobs(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertEqual(0, profile["selfplay_actor_tt_mb"])
        self.assertEqual(10_000, profile["selfplay_policy_node_cap"])
        self.assertEqual(20_000, profile["selfplay_bestmove_node_cap"])

        args = autopilot._parse_args(
            [
                "--out-root",
                "runs",
                "--selfplay-actor-tt-mb",
                "256",
                "--selfplay-policy-node-cap",
                "50000",
                "--selfplay-bestmove-node-cap",
                "100000",
            ]
        )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertEqual(256, resolved["selfplay_actor_tt_mb"])
        self.assertEqual(50_000, resolved["selfplay_policy_node_cap"])
        self.assertEqual(100_000, resolved["selfplay_bestmove_node_cap"])

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
                "--gate-parallel-games",
                "7",
                "--gate-confirmation-games",
                "48",
                "--gate-confirmation-min-score-delta",
                "3",
                "--gate-incremental-pst-policy",
                "regression-veto",
                "--gate-pst-veto-margin",
                "0.125",
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
        self.assertEqual(7, resolved["gate_parallel_games"])
        self.assertEqual(48, resolved["gate_confirmation_games"])
        self.assertEqual(3.0, resolved["gate_confirmation_min_score_delta"])
        self.assertEqual("regression-veto", resolved["gate_incremental_pst_policy"])
        self.assertEqual(0.125, resolved["gate_pst_veto_margin"])

    def test_incremental_pst_policy_configuration_is_fail_closed(self) -> None:
        autopilot._validate_gate_promotion_policy(
            {
                "gate_incremental_pst_policy": "strict-superiority",
                "gate_pst_veto_margin": 0.0,
            }
        )
        autopilot._validate_gate_promotion_policy(
            {
                "gate_incremental_pst_policy": "regression-veto",
                "gate_pst_veto_margin": 0.25,
            }
        )
        with self.assertRaisesRegex(ValueError, "unsupported incremental PST policy"):
            autopilot._validate_gate_promotion_policy(
                {"gate_incremental_pst_policy": "always-promote"}
            )
        for invalid in (-0.01, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "finite non-negative",
            ):
                autopilot._validate_gate_promotion_policy(
                    {
                        "gate_incremental_pst_policy": "regression-veto",
                        "gate_pst_veto_margin": invalid,
                    }
                )

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
                for name in (
                    "gate_compare_fallback.json",
                    "gate_compare_fallback_confirmation.json",
                    "gate_compare_fallback_confirmation_absolute_pst.json",
                ):
                    (cycle_dir / name).write_text(
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
            self.assertTrue((accepted_old / "gate_compare_fallback.json").is_file())
            self.assertTrue(
                (accepted_old / "gate_compare_fallback_confirmation.json").is_file()
            )
            self.assertTrue(
                (
                    accepted_old
                    / "gate_compare_fallback_confirmation_absolute_pst.json"
                ).is_file()
            )
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
            self.assertEqual(5, state["deployment_state_version"])
            self.assertEqual(
                "paired-bootstrap-pst-v2",
                state["promotion_evidence_schema"],
            )
            self.assertTrue(
                all(
                    model["promotion_evidence_status"] == "legacy-unverified"
                    for model in state["accepted_models"]
                )
            )
            self.assertFalse(autopilot._migrate_deployment_state(state))

    def test_acceptance_records_incremental_pst_not_vetoed_evidence_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant = _write_fake_quant(Path(tmp) / "candidate.nnue")
            state = {"accepted_models": []}
            gate = {
                "confirmation": {"accepted": True},
                "absolute": {
                    "pst_decision": {
                        "schema": "incremental-pst-regression-veto-v1",
                        "accepted": True,
                        "vetoed": False,
                        "strict_pst_superiority_passed": False,
                        "proves_pst_non_inferiority": False,
                    }
                }
            }

            autopilot._record_acceptance(
                state=state,
                cycle_idx=9,
                quant_path=quant,
                quant_sha256="candidate-sha",
                gate=gate,
                blend_percent=50,
                model_identity={"input_dim": 81_920},
            )

            self.assertEqual(
                "paired-bootstrap-relative-verified-pst-not-vetoed",
                state["accepted_models"][-1]["promotion_evidence_status"],
            )

            strict_state = {"accepted_models": []}
            autopilot._record_acceptance(
                state=strict_state,
                cycle_idx=1,
                quant_path=quant,
                quant_sha256="candidate-sha",
                gate={
                    "absolute": {
                        "pst_decision": {
                            "schema": "strict-pst-superiority-v1",
                            "accepted": True,
                            "strict_pst_superiority_passed": True,
                        }
                    }
                },
                blend_percent=25,
                model_identity={"input_dim": 81_920},
            )
            self.assertEqual(
                "paired-bootstrap-pst-superior",
                strict_state["accepted_models"][-1]["promotion_evidence_status"],
            )

    def test_v4_strict_accepted_evidence_migrates_to_v5_superiority(self) -> None:
        state = {
            "deployment_state_version": 4,
            "promotion_evidence_schema": "paired-bootstrap-pst-v1",
            "next_cycle": 2,
            "completed_cycles": [{"cycle": 1}],
            "accepted_models": [
                {
                    "cycle": 1,
                    "promotion_evidence_status": "paired-bootstrap-pst-verified",
                    "gate": {
                        "confirmation": {"accepted": True},
                        "absolute": {
                            "accepted": True,
                            "statistics": {
                                "schema": "paired-bootstrap-gate-v1",
                                "eligible": True,
                                "accepted": True,
                                "confidence_interval": {
                                    "lower": 0.0625,
                                    "upper": 0.4375,
                                },
                            },
                        },
                    },
                },
                {"cycle": 0, "gate": {"reason": "legacy-aggregate-only"}},
            ],
            "active_model_path": None,
        }

        self.assertTrue(autopilot._migrate_deployment_state(state))
        self.assertEqual(5, state["deployment_state_version"])
        self.assertEqual("paired-bootstrap-pst-v2", state["promotion_evidence_schema"])
        self.assertEqual(
            "paired-bootstrap-pst-superior",
            state["accepted_models"][0]["promotion_evidence_status"],
        )
        self.assertEqual(
            "legacy-unverified",
            state["accepted_models"][1]["promotion_evidence_status"],
        )
        self.assertFalse(autopilot._migrate_deployment_state(state))

    def test_v5_pst_not_vetoed_evidence_migration_is_idempotent(self) -> None:
        partition_schema = (
            autopilot.run_pipeline.train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA
        )
        state = {
            "deployment_state_version": 5,
            "promotion_evidence_schema": "paired-bootstrap-pst-v2",
            "next_cycle": 2,
            "completed_cycles": [{"cycle": 1}],
            "accepted_models": [
                {
                    "cycle": 1,
                    "gate": {
                        "confirmation": {"accepted": True},
                        "absolute": {
                            "accepted": True,
                            "strict_pst_superiority_passed": False,
                            "pst_decision": {
                                "schema": "incremental-pst-regression-veto-v1",
                                "accepted": True,
                                "vetoed": False,
                                "strict_pst_superiority_passed": False,
                                "proves_pst_non_inferiority": False,
                            },
                        },
                    },
                }
            ],
            "training_lineage_start_cycle": 1,
            "validation_partition_schema": partition_schema,
            "validation_partition_start_cycle": 1,
            "training_model_identity": None,
            "active_model_path": None,
            "active_model_sha256": None,
            "active_model_blend_percent": 0,
            "active_model_identity": None,
        }

        self.assertTrue(autopilot._migrate_deployment_state(state))
        self.assertEqual(
            "paired-bootstrap-relative-verified-pst-not-vetoed",
            state["accepted_models"][0]["promotion_evidence_status"],
        )
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

    def test_profile_default_and_cli_override_for_external_teacher_quant(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertIn("teacher_external_quant_file", profile)
        self.assertIsNone(profile["teacher_external_quant_file"])

        try:
            args = autopilot._parse_args(
                [
                    "--out-root",
                    "runs",
                    "--teacher-external-quant-file",
                    "nets/h128_best.nnue",
                ]
            )
        except SystemExit:
            self.fail(
                "--teacher-external-quant-file is not a recognized autopilot flag"
            )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertEqual(
            Path("nets/h128_best.nnue"), resolved["teacher_external_quant_file"]
        )

    def test_external_teacher_resolver_is_fail_closed_and_piebot_lineage_only(
        self,
    ) -> None:
        self.assertTrue(
            hasattr(autopilot, "_resolve_external_teacher_quant"),
            "autopilot lacks the C8 external-teacher resolver",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(
                autopilot._resolve_external_teacher_quant(
                    {"teacher_external_quant_file": None}
                )
            )
            teacher = _write_fake_quant(
                root / "teacher.nnue",
                input_dim=81_920,
                marker=b"external-teacher",
            )
            path, sha = autopilot._resolve_external_teacher_quant(
                {"teacher_external_quant_file": teacher}
            )
            self.assertEqual(teacher, path)
            self.assertEqual(
                hashlib.sha256(teacher.read_bytes()).hexdigest(), sha
            )
            with self.assertRaisesRegex(ValueError, "external teacher"):
                autopilot._resolve_external_teacher_quant(
                    {"teacher_external_quant_file": root / "missing-teacher.nnue"}
                )
            # INVARIANT: only a PieBot-lineage quant net may teach; a file
            # that is not a PieBot quantized model is rejected outright
            # (Stockfish/external-engine labels remain forbidden).
            invalid = root / "not-a-piebot-net.nnue"
            invalid.write_bytes(b"stockfish-labels-forbidden")
            with self.assertRaisesRegex(ValueError, "PieBot"):
                autopilot._resolve_external_teacher_quant(
                    {"teacher_external_quant_file": invalid}
                )

    def test_external_teacher_overrides_state_teacher_but_not_actor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            initial_active = _write_fake_quant(
                root / "initial-active.nnue",
                input_dim=81_920,
                marker=b"actor",
            )
            external_teacher = _write_fake_quant(
                root / "external-teacher.nnue",
                input_dim=81_920,
                marker=b"external-teacher",
            )
            calls = []

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

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                try:
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
                            "--teacher-external-quant-file",
                            str(external_teacher),
                        ]
                    )
                except SystemExit:
                    self.fail(
                        "--teacher-external-quant-file is not wired into autopilot"
                    )

            self.assertEqual(0, rc)
            self.assertEqual(1, len(calls))
            # The ACTOR (selfplay model) remains the state-derived active model.
            self.assertEqual(
                initial_active.resolve(), calls[0]["selfplay_nnue_quant_file"]
            )
            self.assertEqual(40, calls[0]["selfplay_nnue_blend_percent"])
            # Only the relabel teacher decouples, at full NNUE strength.
            self.assertEqual(
                external_teacher, calls[0]["teacher_relabel_nnue_quant_file"]
            )
            self.assertEqual(100, calls[0]["teacher_relabel_nnue_blend_percent"])

    def test_missing_external_teacher_is_hard_error_before_cycle_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            calls = []

            def _fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                raise AssertionError(
                    "run_pipeline must not run without a verified external teacher"
                )

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                try:
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
                            "--teacher-external-quant-file",
                            str(root / "missing-teacher.nnue"),
                        ]
                    )
                except SystemExit:
                    self.fail(
                        "--teacher-external-quant-file is not wired into autopilot"
                    )

            self.assertEqual(2, rc)
            self.assertEqual([], calls)
            state = json.loads(
                (out_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual([], state["completed_cycles"])
            self.assertIn("external teacher", state["last_error"]["error"])

    def test_default_none_external_teacher_keeps_legacy_kwargs_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            initial_active = _write_fake_quant(
                root / "initial-active.nnue",
                input_dim=81_920,
                marker=b"actor",
            )
            calls = []

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
            # Legacy behavior: the state-derived model both acts and teaches.
            self.assertEqual(
                initial_active.resolve(), calls[0]["selfplay_nnue_quant_file"]
            )
            self.assertEqual(
                initial_active.resolve(),
                calls[0]["teacher_relabel_nnue_quant_file"],
            )
            self.assertEqual(40, calls[0]["selfplay_nnue_blend_percent"])
            self.assertEqual(40, calls[0]["teacher_relabel_nnue_blend_percent"])
            self.assertNotIn("teacher_external_quant_file", calls[0])
            state = json.loads(
                (out_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            record = state["completed_cycles"][0]
            self.assertNotIn("teacher_external_quant_file", record)
            self.assertNotIn("teacher_external_quant_sha256", record)

    def test_external_teacher_sha256_recorded_in_cycle_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "runs"
            external_teacher = _write_fake_quant(
                root / "external-teacher.nnue",
                input_dim=81_920,
                marker=b"external-teacher",
            )

            def _fake_run_pipeline(**kwargs):
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

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                try:
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
                            "--teacher-external-quant-file",
                            str(external_teacher),
                        ]
                    )
                except SystemExit:
                    self.fail(
                        "--teacher-external-quant-file is not wired into autopilot"
                    )

            self.assertEqual(0, rc)
            state = json.loads(
                (out_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            record = state["completed_cycles"][0]
            self.assertEqual(
                str(external_teacher), record["teacher_external_quant_file"]
            )
            self.assertEqual(
                hashlib.sha256(external_teacher.read_bytes()).hexdigest(),
                record["teacher_external_quant_sha256"],
            )

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

    def test_gate_parallelism_is_cache_bound_and_change_forces_fresh_gate(self) -> None:
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
                            "--gate-parallel-games",
                            "3",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(1, len(gate_calls))
            state = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(quant_sha, state["last_gated_quant_sha256"])
            self.assertEqual("unchanged-training-checkpoint", state["last_gate"]["reason"])
            self.assertEqual(3, state["last_gate_identity"]["parallel_games_requested"])
            self.assertEqual(
                "bounded-pair-workers-v1",
                state["last_gate_identity"]["parallelism_schema"],
            )
            self.assertEqual(
                "strict-pst-superiority-v1",
                state["last_gate_identity"]["absolute_decision_rule"],
            )
            self.assertEqual(
                "strict-superiority",
                state["last_gate_identity"]["incremental_pst_policy"],
            )
            self.assertEqual(0.0, state["last_gate_identity"]["pst_veto_margin"])
            self.assertEqual(3, gate_calls[0]["parallel_games"])

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
                            "--gate-parallel-games",
                            "4",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(2, len(gate_calls))
            state = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(4, state["last_gate_identity"]["parallel_games_requested"])
            self.assertEqual(4, gate_calls[-1]["parallel_games"])

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
                payload = _paired_gate_payload([2.0, 2.0, 2.0])
                payload["parallel_games_requested"] = 3
                out_json.write_text(json.dumps(payload), encoding="utf-8")

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
                    parallel_games=3,
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
            self.assertEqual("3", cmd[cmd.index("--parallel-games") + 1])
            self.assertIn("--paired-openings", cmd)
            self.assertEqual(3, gate["parallel_games_requested"])
            self.assertEqual(1, gate["parallel_games"])
            self.assertEqual("bounded-pair-workers-v1", gate["parallelism_schema"])
            self.assertEqual(str(root), kwargs["cwd"])
            self.assertTrue(kwargs["check"])

    def test_model_gate_parallel_games_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")
            with mock.patch("training.nnue.autopilot.subprocess.run") as runner:
                with self.assertRaisesRegex(ValueError, "parallel games must be positive"):
                    autopilot._run_model_gate(
                        piebot_dir=root,
                        out_json=root / "gate.json",
                        base_quant=None,
                        candidate_quant=candidate,
                        games=2,
                        movetime_ms=1,
                        noise_plies=0,
                        noise_topk=1,
                        threads=1,
                        seed=1,
                        min_score_delta=0.0,
                        parallel_games=0,
                    )
            runner.assert_not_called()

    def test_model_gate_rejects_parallelism_evidence_from_another_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "gate.json"
            candidate = root / "candidate.nnue"
            candidate.write_bytes(b"PIENNQ01dummy")

            def _fake_run(_cmd, **_kwargs):
                payload = _paired_gate_payload([2.0])
                payload["parallel_games_requested"] = 1
                out_json.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch(
                "training.nnue.autopilot.subprocess.run", side_effect=_fake_run
            ):
                with self.assertRaisesRegex(ValueError, "does not match the request"):
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
                        parallel_games=2,
                    )

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
                            "parallel_games_requested": 1,
                            "parallel_games": 1,
                            "parallelism_schema": "bounded-pair-workers-v1",
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

    def test_incremental_successor_defaults_to_strict_pst_superiority(self) -> None:
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
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "minimum_mean_pair_delta": 0.0,
                            "confidence_interval": {"lower": -0.25, "upper": 0.25},
                        },
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
                    parallel_games=5,
                )

            self.assertFalse(attempt["accepted"])
            self.assertEqual("absolute-pst-rejected", attempt["reason"])
            self.assertEqual(3, len(calls))
            self.assertEqual([5, 5, 5], [call["parallel_games"] for call in calls])
            self.assertIsNone(calls[-1]["base_quant"])
            self.assertEqual("pure-pst", attempt["absolute"]["baseline_kind"])
            self.assertEqual(
                "strict-pst-superiority-v1",
                attempt["absolute"]["pst_decision"]["schema"],
            )

    def test_incremental_successor_passes_within_configured_pst_regression_margin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = _write_fake_quant(root / "active.nnue", marker=b"active")
            candidate = _write_fake_quant(root / "candidate.nnue", marker=b"candidate")
            results = iter(
                [
                    {"accepted": True, "baseline_kind": "active-model"},
                    {"accepted": True, "baseline_kind": "active-model"},
                    {
                        "accepted": False,
                        "baseline_kind": "pure-pst",
                        "reason": "confidence-lower-bound-not-positive",
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "minimum_mean_pair_delta": 0.25,
                            "confidence_interval": {
                                "lower": -0.3125,
                                "upper": -0.05,
                            },
                        },
                    },
                ]
            )

            with mock.patch(
                "training.nnue.autopilot._run_model_gate",
                side_effect=lambda **_kwargs: next(results),
            ):
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
                    confirmation_min_score_delta=0.25,
                    incremental_pst_policy="regression-veto",
                    pst_veto_margin=0.1,
                    base_blend_percent=50,
                    candidate_blend_percent=50,
                    paired_openings=True,
                )

            self.assertTrue(attempt["accepted"])
            self.assertEqual("confirmation-accepted-pst-not-vetoed", attempt["reason"])
            absolute = attempt["absolute"]
            self.assertFalse(absolute["strict_pst_superiority_passed"])
            self.assertTrue(absolute["accepted"])
            self.assertEqual(
                "incremental-pst-regression-veto-v1",
                absolute["pst_decision"]["schema"],
            )
            self.assertFalse(absolute["pst_decision"]["vetoed"])
            self.assertEqual(0.1, absolute["pst_decision"]["pst_veto_margin"])
            self.assertEqual(-0.1, absolute["pst_decision"]["veto_threshold"])
            self.assertEqual(
                0.25,
                absolute["pst_decision"]["strict_minimum_mean_pair_delta"],
            )
            self.assertFalse(
                absolute["pst_decision"]["proves_pst_non_inferiority"]
            )

    def test_incremental_successor_is_vetoed_when_confidently_worse_than_pst(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = _write_fake_quant(root / "active.nnue", marker=b"active")
            candidate = _write_fake_quant(root / "candidate.nnue", marker=b"candidate")
            results = iter(
                [
                    {"accepted": True, "baseline_kind": "active-model"},
                    {"accepted": True, "baseline_kind": "active-model"},
                    {
                        "accepted": False,
                        "baseline_kind": "pure-pst",
                        "reason": "confidence-lower-bound-not-positive",
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "minimum_mean_pair_delta": 0.0,
                            "confidence_interval": {
                                "lower": -0.375,
                                "upper": 0.0,
                            },
                        },
                    },
                ]
            )

            with mock.patch(
                "training.nnue.autopilot._run_model_gate",
                side_effect=lambda **_kwargs: next(results),
            ):
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
                    incremental_pst_policy="regression-veto",
                    pst_veto_margin=0.0,
                    base_blend_percent=50,
                    candidate_blend_percent=50,
                    paired_openings=True,
                )

            self.assertFalse(attempt["accepted"])
            self.assertEqual("absolute-pst-regression-vetoed", attempt["reason"])
            self.assertFalse(attempt["absolute"]["strict_pst_superiority_passed"])
            self.assertTrue(attempt["absolute"]["pst_decision"]["vetoed"])

    def test_incremental_pst_regression_veto_requires_eligible_absolute_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = _write_fake_quant(root / "active.nnue", marker=b"active")
            candidate = _write_fake_quant(root / "candidate.nnue", marker=b"candidate")
            results = iter(
                [
                    {"accepted": True, "baseline_kind": "active-model"},
                    {"accepted": True, "baseline_kind": "active-model"},
                    {"accepted": True, "baseline_kind": "pure-pst"},
                ]
            )

            with mock.patch(
                "training.nnue.autopilot._run_model_gate",
                side_effect=lambda **_kwargs: next(results),
            ):
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
                    base_blend_percent=50,
                    candidate_blend_percent=50,
                    paired_openings=True,
                    incremental_pst_policy="regression-veto",
                    pst_veto_margin=0.0,
                )

            self.assertFalse(attempt["accepted"])
            self.assertEqual("absolute-pst-rejected", attempt["reason"])
            self.assertFalse(attempt["absolute"]["pst_decision"]["eligible"])
            self.assertEqual(
                "pst-regression-veto-evidence-ineligible",
                attempt["absolute"]["reason"],
            )

    def test_incremental_pst_regression_veto_never_bypasses_relative_confirmation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = _write_fake_quant(root / "active.nnue", marker=b"active")
            candidate = _write_fake_quant(root / "candidate.nnue", marker=b"candidate")
            calls = []
            results = iter(
                [
                    {"accepted": True, "baseline_kind": "active-model"},
                    {
                        "accepted": False,
                        "baseline_kind": "active-model",
                        "reason": "confidence-lower-bound-not-positive",
                    },
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
                    incremental_pst_policy="regression-veto",
                    pst_veto_margin=0.0,
                    base_blend_percent=50,
                    candidate_blend_percent=50,
                    paired_openings=True,
                )

            self.assertEqual([24, 96], [call["games"] for call in calls])
            self.assertFalse(attempt["accepted"])
            self.assertEqual("confirmation-rejected", attempt["reason"])
            self.assertIsNone(attempt["absolute"])

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
                    {
                        "accepted": False,
                        "delta_points": 0.0,
                        "games": 96,
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "confidence_interval": {"lower": -1.0, "upper": 3.0},
                        },
                    },
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
            self.assertIsNone(attempt["absolute"])
            self.assertEqual([24, 96], [call["games"] for call in calls])
            self.assertEqual([0.0, 2.0], [call["min_score_delta"] for call in calls])
            self.assertTrue(all(call["paired_openings"] for call in calls))

    def test_positive_mean_screen_reaches_strict_confirmation_despite_wide_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = _write_fake_quant(root / "candidate.nnue")
            calls = []
            results = iter(
                [
                    {
                        "accepted": False,
                        "reason": "confidence-lower-bound-not-positive",
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "mean_pair_delta": 1.0 / 12.0,
                            "minimum_mean_pair_delta": 0.0,
                            "confidence_interval": {
                                "lower": -0.25,
                                "upper": 5.0 / 12.0,
                            },
                        },
                    },
                    {
                        "accepted": False,
                        "reason": "confidence-lower-bound-not-positive",
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": False,
                            "mean_pair_delta": 0.0,
                            "minimum_mean_pair_delta": 0.0,
                        },
                    },
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
                    confirmation_min_score_delta=0.0,
                    base_blend_percent=0,
                    candidate_blend_percent=25,
                    paired_openings=True,
                )

            self.assertEqual([24, 96], [call["games"] for call in calls])
            self.assertTrue(attempt["screen"]["accepted"])
            self.assertFalse(attempt["screen"]["confidence_gate_accepted"])
            self.assertEqual(
                "mean-pair-delta-screen-v1",
                attempt["screen"]["screen_filter"]["schema"],
            )
            self.assertFalse(attempt["confirmation"]["accepted"])
            self.assertFalse(attempt["accepted"])
            self.assertEqual("confirmation-rejected", attempt["reason"])

    def test_gsprt_decision_accepts_strong_positive_pair_deltas(self) -> None:
        record = autopilot._gsprt_decision(
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 6,
            delta1=0.25,
            alpha=0.05,
            beta=0.05,
        )
        self.assertEqual("accept", record["decision"])
        self.assertEqual(48, record["pairs"])
        self.assertGreaterEqual(record["llr"], record["upper_bound"])
        self.assertAlmostEqual(0.75, record["mean_pair_delta"])
        self.assertGreater(record["variance"], 0.0)

    def test_gsprt_decision_rejects_strong_negative_pair_deltas(self) -> None:
        record = autopilot._gsprt_decision(
            [-1.0, -1.0, 0.0, -1.0] * 12,
            delta1=0.25,
            alpha=0.05,
            beta=0.05,
        )
        self.assertEqual("reject", record["decision"])
        self.assertEqual(48, record["pairs"])
        self.assertLessEqual(record["llr"], record["lower_bound"])

    def test_gsprt_decision_requires_minimum_pairs_before_deciding(self) -> None:
        for deltas in ([], [2.0]):
            with self.subTest(pairs=len(deltas)):
                record = autopilot._gsprt_decision(
                    deltas,
                    delta1=0.25,
                    alpha=0.05,
                    beta=0.05,
                )
                self.assertEqual("continue", record["decision"])
                self.assertEqual(len(deltas), record["pairs"])

    def test_gsprt_decision_guards_zero_variance_with_floor(self) -> None:
        constant_positive = autopilot._gsprt_decision(
            [0.5] * 10,
            delta1=0.25,
            alpha=0.05,
            beta=0.05,
        )
        self.assertEqual("accept", constant_positive["decision"])
        self.assertTrue(math.isfinite(constant_positive["llr"]))
        self.assertGreaterEqual(constant_positive["variance"], 1e-6)

        constant_zero = autopilot._gsprt_decision(
            [0.0] * 10,
            delta1=0.25,
            alpha=0.05,
            beta=0.05,
        )
        self.assertEqual("reject", constant_zero["decision"])
        self.assertTrue(math.isfinite(constant_zero["llr"]))
        self.assertGreaterEqual(constant_zero["variance"], 1e-6)

        with self.assertRaises(ValueError):
            autopilot._gsprt_decision(
                [float("nan")] * 4,
                delta1=0.25,
                alpha=0.05,
                beta=0.05,
            )

    def test_gsprt_decision_bounds_are_monotonic_in_alpha_and_beta(self) -> None:
        # Mean 0.1667 over 48 pairs sits between the loose and strict accept
        # bounds, so tightening alpha/beta must flip accept back to continue.
        deltas = [1.0] * 8 + [0.0] * 40
        loose = autopilot._gsprt_decision(deltas, delta1=0.25, alpha=0.2, beta=0.2)
        strict = autopilot._gsprt_decision(deltas, delta1=0.25, alpha=0.01, beta=0.01)
        self.assertGreater(strict["upper_bound"], loose["upper_bound"])
        self.assertLess(strict["lower_bound"], loose["lower_bound"])
        self.assertAlmostEqual(loose["llr"], strict["llr"])
        self.assertEqual("accept", loose["decision"])
        self.assertEqual("continue", strict["decision"])

    def test_gsprt_accepts_historical_promotion_magnitudes_at_defaults(self) -> None:
        # Cycle 94 confirmed at mean +0.3125 and cycle 98 at +0.25 over 48
        # pairs; the SPRT defaults must reproduce both LCB acceptances.
        cycle94 = [1.0] * 15 + [0.0] * 33
        cycle98 = [1.0] * 12 + [0.0] * 36
        for name, deltas in (("cycle94", cycle94), ("cycle98", cycle98)):
            with self.subTest(cycle=name):
                record = autopilot._gsprt_decision(
                    deltas,
                    delta1=0.25,
                    alpha=0.05,
                    beta=0.05,
                )
                self.assertEqual("accept", record["decision"])
                self.assertEqual(48, record["pairs"])

    def test_gsprt_mean_zero_evidence_rejects_at_defaults(self) -> None:
        record = autopilot._gsprt_decision(
            [1.0, -1.0] * 150,
            delta1=0.25,
            alpha=0.05,
            beta=0.05,
        )
        self.assertEqual("reject", record["decision"])
        self.assertEqual(300, record["pairs"])

    def test_sprt_batch_merge_reindexes_overlapping_indices_for_validation(self) -> None:
        # Both batches use raw indices 0..47 / pairs 0..23; the merge must
        # re-index them so the strict paired evidence validator still passes.
        first = _paired_gate_payload([1.0] * 24)
        second = _paired_gate_payload([0.0] * 24)
        merged = autopilot._merge_paired_gate_batches([first, second])
        self.assertEqual(96, merged["games"])
        self.assertIs(True, merged["paired_openings"])
        self.assertEqual(
            set(range(96)),
            {game["game_index"] for game in merged["game_results"]},
        )
        self.assertEqual(
            set(range(48)),
            {game["pair_index"] for game in merged["game_results"]},
        )
        self.assertEqual(
            {"baseline": 0.0, "experimental": 24.0, "draws": 72},
            merged["points"],
        )
        statistics = autopilot._paired_gate_statistics(
            merged,
            confidence_level=0.95,
            bootstrap_samples=500,
            seed=7,
            minimum_mean_pair_delta=0.0,
        )
        self.assertTrue(statistics["eligible"])
        self.assertEqual(48, statistics["complete_pairs"])
        self.assertAlmostEqual(0.5, statistics["mean_pair_delta"])

    def test_sprt_confirmation_accepts_after_min_pairs_and_merges_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "confirmation.json"
            quant = _write_fake_quant(root / "candidate.nnue")
            commands = []
            payloads = iter(
                [
                    _paired_gate_payload([1.0] * 8 + [0.0] * 16),
                    _paired_gate_payload([1.0] * 7 + [0.0] * 17),
                ]
            )

            def _fake_run(cmd, **_kwargs):
                commands.append(cmd)
                batch_json = Path(cmd[cmd.index("--json-out") + 1])
                batch_json.write_text(json.dumps(next(payloads)), encoding="utf-8")

            with mock.patch(
                "training.nnue.autopilot.subprocess.run", side_effect=_fake_run
            ):
                result = autopilot._run_sprt_confirmation(
                    piebot_dir=root,
                    out_json=out_json,
                    base_quant=None,
                    candidate_quant=quant,
                    movetime_ms=25,
                    noise_plies=4,
                    noise_topk=3,
                    threads=1,
                    seed=9001,
                    min_score_delta=0.0,
                    base_blend_percent=0,
                    candidate_blend_percent=25,
                    confidence_level=0.95,
                    bootstrap_samples=500,
                    parallel_games=1,
                    delta1=0.25,
                    alpha=0.05,
                    beta=0.05,
                    min_pairs=48,
                    batch_pairs=24,
                    max_pairs=300,
                )

            self.assertEqual(2, len(commands))
            games = [cmd[cmd.index("--games") + 1] for cmd in commands]
            self.assertEqual(["48", "48"], games)
            seeds = [cmd[cmd.index("--seed") + 1] for cmd in commands]
            self.assertEqual(["9001", "9002"], seeds)
            batch_paths = [cmd[cmd.index("--json-out") + 1] for cmd in commands]
            self.assertEqual(2, len(set(batch_paths)))
            self.assertNotIn(str(out_json), batch_paths)

            self.assertTrue(result["accepted"])
            self.assertEqual("sprt-accepted", result["reason"])
            self.assertEqual("accept", result["sprt"]["decision"])
            self.assertEqual(48, result["sprt"]["pairs"])
            self.assertEqual(96, result["games"])
            self.assertEqual(str(out_json), result["json_path"])
            statistics = result["statistics"]
            self.assertEqual("paired-bootstrap-gate-v1", statistics["schema"])
            self.assertTrue(statistics["eligible"])
            self.assertEqual(48, statistics["complete_pairs"])
            self.assertAlmostEqual(0.3125, statistics["mean_pair_delta"])
            self.assertIn("confidence_interval", statistics)

            merged = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(96, merged["games"])
            self.assertIs(True, merged["paired_openings"])
            self.assertEqual(
                set(range(96)),
                {game["game_index"] for game in merged["game_results"]},
            )
            self.assertEqual(
                {"baseline": 0.0, "experimental": 15.0, "draws": 81},
                merged["points"],
            )

    def test_sprt_confirmation_inconclusive_at_max_pairs_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def _fake_gate(**kwargs):
                calls.append(kwargs)
                pairs = int(kwargs["games"]) // 2
                payload = _paired_gate_payload([2.0, -2.0] * (pairs // 2))
                statistics = autopilot._paired_gate_statistics(
                    payload,
                    confidence_level=0.95,
                    bootstrap_samples=50,
                    seed=3,
                    minimum_mean_pair_delta=0.0,
                )
                return {
                    "accepted": bool(statistics["accepted"]),
                    "reason": statistics["reason"],
                    "evidence_eligible": True,
                    "evidence_schema": statistics["schema"],
                    "baseline_points": payload["points"]["baseline"],
                    "experimental_points": payload["points"]["experimental"],
                    "delta_points": 0.0,
                    "games": payload["games"],
                    "json_path": str(kwargs["out_json"]),
                    "statistics": statistics,
                    "pair_outcomes": statistics["pair_outcomes"],
                    "game_results": statistics["game_results"],
                }

            with mock.patch(
                "training.nnue.autopilot._run_model_gate", side_effect=_fake_gate
            ):
                result = autopilot._run_sprt_confirmation(
                    piebot_dir=root,
                    out_json=root / "confirmation.json",
                    base_quant=None,
                    candidate_quant=root / "candidate.nnue",
                    movetime_ms=25,
                    noise_plies=4,
                    noise_topk=3,
                    threads=1,
                    seed=7,
                    min_score_delta=0.0,
                    base_blend_percent=0,
                    candidate_blend_percent=25,
                    confidence_level=0.95,
                    bootstrap_samples=50,
                    parallel_games=1,
                    delta1=0.25,
                    alpha=0.05,
                    beta=0.05,
                    min_pairs=48,
                    batch_pairs=24,
                    max_pairs=300,
                )

            self.assertEqual([48] * 12 + [24], [int(call["games"]) for call in calls])
            self.assertEqual(list(range(7, 20)), [int(call["seed"]) for call in calls])
            self.assertEqual(13, len({str(call["out_json"]) for call in calls}))
            self.assertFalse(result["accepted"])
            self.assertEqual("max-pairs-inconclusive", result["reason"])
            self.assertEqual("reject", result["sprt"]["decision"])
            self.assertEqual(300, result["sprt"]["pairs"])
            self.assertEqual(600, result["games"])
            self.assertEqual(300, result["statistics"]["complete_pairs"])

    def test_sprt_enabled_confirmation_uses_gsprt_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = _write_fake_quant(root / "candidate.nnue")
            screen_calls = []
            sprt_calls = []

            def _screen(**kwargs):
                screen_calls.append(kwargs)
                return {
                    "accepted": False,
                    "reason": "confidence-lower-bound-not-positive",
                    "evidence_eligible": True,
                    "statistics": {
                        "eligible": True,
                        "accepted": False,
                        "mean_pair_delta": 0.4,
                    },
                }

            def _sprt(**kwargs):
                sprt_calls.append(kwargs)
                return {
                    "accepted": True,
                    "reason": "sprt-accepted",
                    "evidence_eligible": True,
                    "games": 96,
                    "statistics": {
                        "eligible": True,
                        "accepted": True,
                        "schema": "paired-bootstrap-gate-v1",
                    },
                    "sprt": {"decision": "accept", "pairs": 48},
                }

            with mock.patch(
                "training.nnue.autopilot._run_model_gate", side_effect=_screen
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_sprt_confirmation",
                    side_effect=_sprt,
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
                        confirmation_min_score_delta=0.0,
                        base_blend_percent=0,
                        candidate_blend_percent=25,
                        paired_openings=True,
                        sprt=True,
                        sprt_delta1=0.3,
                        sprt_alpha=0.02,
                        sprt_beta=0.04,
                        sprt_min_pairs=50,
                        sprt_batch_pairs=20,
                        sprt_max_pairs=200,
                    )

            self.assertEqual(1, len(screen_calls))
            self.assertEqual(24, screen_calls[0]["games"])
            self.assertEqual(1, len(sprt_calls))
            call = sprt_calls[0]
            self.assertEqual(0.3, call["delta1"])
            self.assertEqual(0.02, call["alpha"])
            self.assertEqual(0.04, call["beta"])
            self.assertEqual(50, call["min_pairs"])
            self.assertEqual(20, call["batch_pairs"])
            self.assertEqual(200, call["max_pairs"])
            self.assertEqual(7 + 1_000_003, call["seed"])
            self.assertEqual(root / "confirmation.json", call["out_json"])
            self.assertTrue(attempt["accepted"])
            self.assertEqual("confirmation-accepted", attempt["reason"])
            self.assertEqual("accept", attempt["confirmation"]["sprt"]["decision"])
            self.assertEqual(0, attempt["confirmation"]["baseline_blend_percent"])
            self.assertEqual(25, attempt["confirmation"]["experimental_blend_percent"])

    def test_sprt_confirmation_requires_paired_openings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = _write_fake_quant(root / "candidate.nnue")
            with mock.patch("training.nnue.autopilot._run_model_gate") as gate:
                with self.assertRaisesRegex(ValueError, "paired openings"):
                    autopilot._run_confirmed_gate_attempt(
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
                        confirmation_min_score_delta=0.0,
                        base_blend_percent=0,
                        candidate_blend_percent=25,
                        paired_openings=False,
                        sprt=True,
                    )
            gate.assert_not_called()

    def test_profile_has_sprt_gate_knobs_default_off(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertIs(False, profile["gate_sprt"])
        self.assertEqual(0.25, profile["gate_sprt_delta1"])
        self.assertEqual(0.05, profile["gate_sprt_alpha"])
        self.assertEqual(0.05, profile["gate_sprt_beta"])
        self.assertEqual(48, profile["gate_sprt_min_pairs"])
        self.assertEqual(24, profile["gate_sprt_batch_pairs"])
        self.assertEqual(300, profile["gate_sprt_max_pairs"])

    def test_cli_overrides_map_sprt_gate_knobs(self) -> None:
        args = autopilot._parse_args(
            [
                "--out-root",
                "runs",
                "--gate-sprt",
                "--gate-sprt-delta1",
                "0.3",
                "--gate-sprt-alpha",
                "0.02",
                "--gate-sprt-beta",
                "0.04",
                "--gate-sprt-min-pairs",
                "60",
                "--gate-sprt-batch-pairs",
                "30",
                "--gate-sprt-max-pairs",
                "240",
            ]
        )
        resolved = autopilot._apply_cli_overrides(
            autopilot.zen5_9755_7d_profile(), args
        )
        self.assertIs(True, resolved["gate_sprt"])
        self.assertEqual(0.3, resolved["gate_sprt_delta1"])
        self.assertEqual(0.02, resolved["gate_sprt_alpha"])
        self.assertEqual(0.04, resolved["gate_sprt_beta"])
        self.assertEqual(60, resolved["gate_sprt_min_pairs"])
        self.assertEqual(30, resolved["gate_sprt_batch_pairs"])
        self.assertEqual(240, resolved["gate_sprt_max_pairs"])

    def test_sprt_disabled_keeps_legacy_confirmation_flow(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        bare = autopilot._parse_args(["--out-root", "runs"])
        self.assertEqual(profile, autopilot._apply_cli_overrides(profile, bare))
        self.assertIs(
            False,
            inspect.signature(autopilot._run_confirmed_gate_attempt)
            .parameters["sprt"]
            .default,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = _write_fake_quant(root / "candidate.nnue")
            calls = []
            results = iter(
                [
                    {
                        "accepted": True,
                        "evidence_eligible": True,
                        "statistics": {
                            "eligible": True,
                            "accepted": True,
                            "mean_pair_delta": 0.5,
                        },
                    },
                    {"accepted": False, "reason": "confidence-lower-bound-not-positive"},
                ]
            )

            def _gate(**kwargs):
                calls.append(kwargs)
                return next(results)

            with mock.patch(
                "training.nnue.autopilot._run_model_gate", side_effect=_gate
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_sprt_confirmation"
                ) as sprt:
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
                        confirmation_min_score_delta=0.0,
                        base_blend_percent=0,
                        candidate_blend_percent=25,
                        paired_openings=True,
                    )

            sprt.assert_not_called()
            self.assertEqual([24, 96], [call["games"] for call in calls])
            self.assertFalse(attempt["accepted"])
            self.assertEqual("confirmation-rejected", attempt["reason"])

    def test_main_sprt_gate_flags_flow_into_confirmation_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sprt_root = Path(tmp) / "runs_sprt"
            legacy_root = Path(tmp) / "runs_legacy"
            sprt_calls = []

            def _fake_run_pipeline(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _write_fake_checkpoint(out_dir)
                quant = _write_fake_quant(out_dir / "nnue_quant.nnue")
                return {
                    "checkpoint_path": str(checkpoint),
                    "quant_path": str(quant),
                }

            def _screen(**_kwargs):
                return {
                    "accepted": False,
                    "reason": "confidence-lower-bound-not-positive",
                    "evidence_eligible": True,
                    "statistics": {
                        "eligible": True,
                        "accepted": False,
                        "mean_pair_delta": 0.5,
                    },
                }

            def _sprt(**kwargs):
                sprt_calls.append(kwargs)
                return {
                    "accepted": True,
                    "reason": "sprt-accepted",
                    "evidence_eligible": True,
                    "games": 96,
                    "statistics": {
                        "eligible": True,
                        "accepted": True,
                        "schema": "paired-bootstrap-gate-v1",
                    },
                    "sprt": {"decision": "accept", "pairs": 48},
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate", side_effect=_screen
                ):
                    with mock.patch(
                        "training.nnue.autopilot._run_sprt_confirmation",
                        side_effect=_sprt,
                    ):
                        rc = autopilot.main(
                            [
                                "--out-root",
                                str(sprt_root),
                                "--hours",
                                "1",
                                "--max-cycles",
                                "1",
                                "--gate-sprt",
                                "--gate-sprt-delta1",
                                "0.3",
                                "--gate-sprt-alpha",
                                "0.02",
                                "--gate-sprt-beta",
                                "0.04",
                                "--gate-sprt-min-pairs",
                                "60",
                                "--gate-sprt-batch-pairs",
                                "30",
                                "--gate-sprt-max-pairs",
                                "240",
                            ]
                        )

            self.assertEqual(0, rc)
            self.assertEqual(1, len(sprt_calls))
            call = sprt_calls[0]
            self.assertEqual(0.3, call["delta1"])
            self.assertEqual(0.02, call["alpha"])
            self.assertEqual(0.04, call["beta"])
            self.assertEqual(60, call["min_pairs"])
            self.assertEqual(30, call["batch_pairs"])
            self.assertEqual(240, call["max_pairs"])
            state = json.loads(
                (sprt_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            identity = state["last_gate_identity"]
            self.assertEqual(
                "gsprt-pair-delta-v1", identity["confirmation_decision_rule"]
            )
            self.assertEqual(
                {
                    "delta1": 0.3,
                    "alpha": 0.02,
                    "beta": 0.04,
                    "min_pairs": 60,
                    "batch_pairs": 30,
                    "max_pairs": 240,
                },
                identity["sprt"],
            )
            self.assertIsNotNone(state.get("active_model_path"))

            def _legacy_reject(**_kwargs):
                return {
                    "accepted": False,
                    "reason": "confidence-lower-bound-not-positive",
                    "evidence_eligible": True,
                    "statistics": {
                        "eligible": True,
                        "accepted": False,
                        "mean_pair_delta": -0.5,
                    },
                }

            with mock.patch(
                "training.nnue.autopilot.run_pipeline.run_pipeline",
                side_effect=_fake_run_pipeline,
            ):
                with mock.patch(
                    "training.nnue.autopilot._run_model_gate",
                    side_effect=_legacy_reject,
                ):
                    with mock.patch(
                        "training.nnue.autopilot._run_sprt_confirmation"
                    ) as legacy_sprt:
                        rc = autopilot.main(
                            [
                                "--out-root",
                                str(legacy_root),
                                "--hours",
                                "1",
                                "--max-cycles",
                                "1",
                            ]
                        )

            self.assertEqual(0, rc)
            legacy_sprt.assert_not_called()
            state = json.loads(
                (legacy_root / "autopilot_state.json").read_text(encoding="utf-8")
            )
            identity = state["last_gate_identity"]
            self.assertNotIn("confirmation_decision_rule", identity)
            self.assertNotIn("sprt", identity)

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

    def test_cross_lineage_checks_equal_blend_before_conservative_fallback(self) -> None:
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
                            "--gate-incremental-pst-policy",
                            "regression-veto",
                            "--gate-pst-veto-margin",
                            "0.0",
                        ]
                    )

            self.assertEqual(0, rc)
            self.assertEqual(75, pipeline_calls[0]["selfplay_nnue_blend_percent"])
            self.assertEqual(75, pipeline_calls[0]["teacher_relabel_nnue_blend_percent"])
            self.assertEqual(
                [75, 25],
                [call["candidate_blend_percent"] for call in gate_calls],
            )
            self.assertEqual(
                "gate_compare_same_blend.json",
                Path(gate_calls[0]["out_json"]).name,
            )
            self.assertEqual(
                "gate_compare_fallback.json",
                Path(gate_calls[1]["out_json"]).name,
            )
            loaded = json.loads((out_root / "autopilot_state.json").read_text())
            self.assertEqual(str(active), loaded["active_model_path"])
            self.assertEqual(75, loaded["active_model_blend_percent"])
            self.assertEqual(40_960, loaded["active_model_identity"]["input_dim"])
            self.assertEqual(
                81_920,
                loaded["last_gate_identity"]["candidate_model_identity"]["input_dim"],
            )
            self.assertEqual(
                [75, 25],
                loaded["last_gate_identity"]["candidate_blend_percents"],
            )
            self.assertEqual(
                "mean-pair-delta-screen-v1",
                loaded["last_gate_identity"]["screen_decision_rule"],
            )
            self.assertEqual(
                "incremental-pst-regression-veto-v1",
                loaded["last_gate_identity"]["absolute_decision_rule"],
            )
            self.assertEqual(
                "regression-veto",
                loaded["last_gate_identity"]["incremental_pst_policy"],
            )
            self.assertEqual(
                0.0,
                loaded["last_gate_identity"]["pst_veto_margin"],
            )

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
