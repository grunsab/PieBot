#!/usr/bin/env python3
"""Static contract checks for the campaign_v2 Vast.ai deployment (plan section 4)."""

from __future__ import annotations

import configparser
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_vast_campaign_v2.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "piebot_campaign_v2.conf"
BOOK = ROOT / "books" / "openings_v1.fen"


class CampaignV2DeploymentTests(unittest.TestCase):
    def _launcher(self) -> str:
        return LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_is_executable_and_has_valid_bash_syntax(self) -> None:
        self.assertTrue(LAUNCHER.is_file())
        self.assertTrue(os.access(LAUNCHER, os.X_OK))
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_launcher_is_piebot_self_teacher_only(self) -> None:
        launcher = self._launcher()
        self.assertNotIn("--teacher-relabel-engine", launcher)
        external_engine_name = "".join(("stock", "fish"))
        self.assertNotIn(external_engine_name, launcher.lower())

    def test_root_and_deadline_are_minted_for_the_full_campaign(self) -> None:
        launcher = self._launcher()
        self.assertIn('OUT_ROOT="${OUT_ROOT:-/workspace/piebot_campaign_v2}"', launcher)
        # 60 days = campaign + 15-day reserve; deadline is minted exactly once.
        self.assertIn('HOURS="${HOURS:-1440}"', launcher)
        self.assertIn(
            "/workspace/piebot_runs/main_72h_self_teacher_repair_v1", launcher
        )
        for forbidden_root in (
            'OUT_ROOT:-/workspace/piebot_runs/main_72h_self_teacher_repair_v1',
            'OUT_ROOT:-/workspace/piebot_runs/main_48h_20260802T081500Z',
        ):
            self.assertNotIn(forbidden_root, launcher)

    def test_bootstrap_sources_are_the_verified_artifacts(self) -> None:
        launcher = self._launcher()
        self.assertIn(
            'PRIOR_RUN_ROOT="/workspace/piebot_runs/main_72h_self_teacher_repair_v1"',
            launcher,
        )
        self.assertIn(
            "$PRIOR_RUN_ROOT/bootstrap/cycle_000086_checkpoint.json",
            launcher,
        )
        self.assertIn(
            "0ce48cc1299d5750bd43512793e843d8363e1e09a5c4a72c3b22e024951f367c",
            launcher,
        )
        self.assertIn(
            "$PRIOR_RUN_ROOT/cycles/cycle_000098/nnue_quant.nnue",
            launcher,
        )
        self.assertIn(
            "3fa9bae3127319930ec16ebb1ee3117656abe7001984f6c8655108a08d278c3a",
            launcher,
        )
        self.assertIn('"--initial-checkpoint-weights-only"', launcher)
        self.assertIn(
            'INITIAL_ACTIVE_MODEL_BLEND_PERCENT="${INITIAL_ACTIVE_MODEL_BLEND_PERCENT:-25}"',
            launcher,
        )

    def test_teacher_depth_is_a_measured_configuration(self) -> None:
        launcher = self._launcher()
        self.assertIn('RELABEL_DEPTH="${RELABEL_DEPTH:-7}"', launcher)
        # Two calibrated teacher shapes exist: depth 7 capped at the measured
        # p95 of depth-5 cost (144k), and depth 9 capped at the measured p95
        # of depth-7 cost (2.5M). Anything else is an unmeasured teacher.
        self.assertIn('[[ "$RELABEL_DEPTH" -eq 7 || "$RELABEL_DEPTH" -eq 9 ]]', launcher)
        # The node cap has no default: it must come from a measured node-cost
        # distribution, so an unset value refuses to launch.
        self.assertIn('RELABEL_MAX_NODES="${RELABEL_MAX_NODES:-}"', launcher)
        self.assertIn("RELABEL_MAX_NODES must be set", launcher)
        self.assertIn('"--teacher-relabel-max-nodes" "$RELABEL_MAX_NODES"', launcher)
        self.assertIn('require_autopilot_flag "--teacher-relabel-max-nodes"', launcher)
        # min_teacher_depth stays 5 (objective-identity field; achieved-depth
        # stamping keeps it honest under the cap).
        self.assertIn('"--min-teacher-depth" "5"', launcher)

    def test_launcher_supports_fresh_init_lineage(self) -> None:
        launcher = self._launcher()
        # v5: a new-width lineage starts from fresh random weights. FRESH_INIT=1
        # skips checkpoint staging and omits the warm-start source flags so
        # train_torch initializes at --hidden-dim (a width-mismatched warm
        # start would hard-fail in train_torch, and rightly so). An empty
        # INITIAL_CHECKPOINT_SOURCE cannot express this because :- defaults
        # swallow empty strings.
        self.assertIn('FRESH_INIT="${FRESH_INIT:-0}"', launcher)
        self.assertIn('if [[ "$FRESH_INIT" != "1" ]]', launcher)

    def test_opening_book_is_staged_by_sha_and_wired(self) -> None:
        launcher = self._launcher()
        self.assertIn("books/openings_v1.fen", launcher)
        self.assertIn(
            "d35b81a1a75d03d6172c40f94c9e8626e3f3b6ed8995f935f5bce1e1c5550294",
            launcher,
        )
        self.assertIn('"--selfplay-openings" "$SELFPLAY_OPENINGS"', launcher)
        self.assertIn('require_autopilot_flag "--selfplay-openings"', launcher)

    def test_book_on_disk_matches_the_pinned_sha(self) -> None:
        import hashlib

        digest = hashlib.sha256(BOOK.read_bytes()).hexdigest()
        self.assertEqual(
            "d35b81a1a75d03d6172c40f94c9e8626e3f3b6ed8995f935f5bce1e1c5550294",
            digest,
        )

    def test_sprt_gate_is_enabled_with_plan_parameters(self) -> None:
        launcher = self._launcher()
        self.assertIn('require_autopilot_flag "--gate-sprt"', launcher)
        self.assertIn('"--gate-sprt"', launcher)
        self.assertIn('"--gate-sprt-delta1" "0.25"', launcher)
        self.assertIn('"--gate-sprt-alpha" "0.05"', launcher)
        self.assertIn('"--gate-sprt-beta" "0.05"', launcher)
        self.assertIn('"--gate-sprt-min-pairs" "48"', launcher)
        self.assertIn('"--gate-sprt-batch-pairs" "24"', launcher)
        self.assertIn('"--gate-sprt-max-pairs" "300"', launcher)

    def test_adjudication_is_explicit_with_plan_values(self) -> None:
        launcher = self._launcher()
        self.assertIn('RESIGN_CP="${RESIGN_CP:-900}"', launcher)
        self.assertIn('RESIGN_PLIES="${RESIGN_PLIES:-8}"', launcher)
        self.assertIn('NO_RESIGN_FRACTION="${NO_RESIGN_FRACTION:-0.15}"', launcher)
        self.assertIn('DRAW_ADJ_CP="${DRAW_ADJ_CP:-10}"', launcher)
        self.assertIn('DRAW_ADJ_PLIES="${DRAW_ADJ_PLIES:-40}"', launcher)
        self.assertIn('DRAW_ADJ_MIN_PLY="${DRAW_ADJ_MIN_PLY:-80}"', launcher)
        self.assertIn('"--selfplay-resign-cp" "$RESIGN_CP"', launcher)
        self.assertIn('"--selfplay-no-resign-fraction" "$NO_RESIGN_FRACTION"', launcher)
        self.assertIn('"--selfplay-draw-adj-min-ply" "$DRAW_ADJ_MIN_PLY"', launcher)
        self.assertIn('require_autopilot_flag "--selfplay-resign-cp"', launcher)

    def test_actor_depth_is_raised_to_five(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        # 2026-08-07/08 (user-directed): depth 2 -> 4 -> 5. Depth-4 cut
        # threefold rows to the campaign-best 38%; depth 5 continues the
        # push, still well under the 80k bestmove node cap.
        self.assertIn('SELFPLAY_DEPTH="5"', environment)

    def test_actor_budget_is_deployed_with_measured_values(self) -> None:
        launcher = self._launcher()
        # Pilot C: relabel dominates cycle time, so a 4x actor raise is cheap;
        # fixes the failing threefold data-shape gate (47% at depth-2/10k).
        self.assertIn('ACTOR_TT_MB="${ACTOR_TT_MB:-128}"', launcher)
        self.assertIn('POLICY_NODE_CAP="${POLICY_NODE_CAP:-40000}"', launcher)
        self.assertIn('BESTMOVE_NODE_CAP="${BESTMOVE_NODE_CAP:-80000}"', launcher)
        self.assertIn('"--selfplay-actor-tt-mb" "$ACTOR_TT_MB"', launcher)
        self.assertIn('"--selfplay-policy-node-cap" "$POLICY_NODE_CAP"', launcher)
        self.assertIn('"--selfplay-bestmove-node-cap" "$BESTMOVE_NODE_CAP"', launcher)
        self.assertIn('require_autopilot_flag "--selfplay-actor-tt-mb"', launcher)

    def test_temperature_window_is_tightened(self) -> None:
        launcher = self._launcher()
        # Cycle-18 tripwire response: threefold rows stuck ~42% because tau 1.0
        # noise over 24 plies dilutes even the upgraded actor. Halve the window.
        self.assertIn('TEMPERATURE_MOVES="${TEMPERATURE_MOVES:-12}"', launcher)
        self.assertIn('"--selfplay-temperature-moves" "$TEMPERATURE_MOVES"', launcher)
        self.assertIn('require_autopilot_flag "--selfplay-temperature-moves"', launcher)

    def test_external_teacher_is_supported_and_staged_when_set(self) -> None:
        launcher = self._launcher()
        # C8: empty default keeps the knob dormant; when set, the file is
        # staged by content hash and passed to autopilot.
        self.assertIn('TEACHER_EXTERNAL_QUANT_FILE="${TEACHER_EXTERNAL_QUANT_FILE:-}"', launcher)
        self.assertIn('TEACHER_EXTERNAL_QUANT_SHA256="${TEACHER_EXTERNAL_QUANT_SHA256:-}"', launcher)
        self.assertIn('"--teacher-external-quant-file"', launcher)
        self.assertIn('require_autopilot_flag "--teacher-external-quant-file"', launcher)

    def test_v5_conf_mints_h128_deep_teacher_lineage(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        # v5 pivot (2026-08-07): the pure-network blunder protocol showed the
        # v4 h64 learner REGRESSING vs the cycle-98 incumbent (ACPL 36.6 vs
        # 34.0, 2.15 vs 1.77 blunders/game) — the h64 loop cannot outrun its
        # own teacher. New lineage: hidden-128 student from fresh random
        # weights (measured cost: 19.1% NPS, evidence/h128_speed_probe),
        # taught by cycle-98 searching depth 9 (median 4.1M nodes measured)
        # capped at the depth-7 p95, relabeling every 6th ply.
        self.assertIn('OUT_ROOT="/workspace/piebot_campaign_v5"', environment)
        self.assertIn('HIDDEN_DIM="128"', environment)
        self.assertIn('FRESH_INIT="1"', environment)
        self.assertNotIn("INITIAL_CHECKPOINT_SOURCE", environment)
        self.assertIn('RELABEL_DEPTH="9"', environment)
        self.assertIn('RELABEL_EVERY="6"', environment)
        self.assertIn('TARGET_CP="250"', environment)
        self.assertNotIn("TEACHER_EXTERNAL_QUANT_FILE", environment)
        # Actor, teacher, and gate incumbent stay the accepted cycle-98 h64.
        self.assertIn(
            'INITIAL_ACTIVE_MODEL_SOURCE="/workspace/campaign_v3_bootstrap/cycle_000098_nnue_quant.nnue"',
            environment,
        )

    def test_outcome_target_env_is_wired(self) -> None:
        launcher = self._launcher()
        self.assertIn('TARGET_CP="${TARGET_CP:-100}"', launcher)
        self.assertIn('"--target-cp" "$TARGET_CP"', launcher)
        self.assertIn('require_autopilot_flag "--target-cp"', launcher)

    def test_slot_partition_reserves_arena_and_ab_lanes(self) -> None:
        launcher = self._launcher()
        self.assertIn('SELFPLAY_PARALLEL_GAMES="${SELFPLAY_PARALLEL_GAMES:-32}"', launcher)
        self.assertIn('RELABEL_THREADS="${RELABEL_THREADS:-32}"', launcher)

    def test_learning_rates_and_width_are_explicit(self) -> None:
        launcher = self._launcher()
        self.assertIn('LEARNING_RATE="${LEARNING_RATE:-0.002}"', launcher)
        self.assertIn('WARM_START_LEARNING_RATE="${WARM_START_LEARNING_RATE:-0.001}"', launcher)
        self.assertIn('HIDDEN_DIM="${HIDDEN_DIM:-64}"', launcher)
        self.assertIn('"--hidden-dim" "$HIDDEN_DIM"', launcher)
        self.assertIn('"--learning-rate" "$LEARNING_RATE"', launcher)

    def test_retention_keeps_replay_window_covered(self) -> None:
        launcher = self._launcher()
        self.assertIn('RETAIN_FULL_CYCLES="${RETAIN_FULL_CYCLES:-8}"', launcher)
        self.assertIn('REPLAY_WINDOW_CYCLES="${REPLAY_WINDOW_CYCLES:-6}"', launcher)
        self.assertIn("REPLAY_WINDOW_CYCLES must not exceed RETAIN_FULL_CYCLES", launcher)

    def test_supervisor_conf_is_restart_safe(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        section = "program:piebot_campaign_v2"
        self.assertIn(section, parser)
        self.assertEqual("unexpected", parser[section]["autorestart"])
        self.assertEqual("0", parser[section]["exitcodes"])
        self.assertIn("run_vast_campaign_v2.sh", parser[section]["command"])
        # 2026-08-07: a stop left the python autopilot orphaned because only
        # the launcher shell received the signal. Stops must hit the group.
        self.assertEqual("true", parser[section]["stopasgroup"])
        self.assertEqual("true", parser[section]["killasgroup"])

    def test_supervisor_conf_supplies_the_measured_node_cap(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        # v5 deep teacher: ask depth 9, budget the measured p95 of depth-7
        # cost (2.49M nodes, 2026-08-06 probe) -> 2500000, mirroring how the
        # depth-7 teacher was capped at depth-5's p95 (144k).
        self.assertIn('RELABEL_MAX_NODES="2500000"', environment)

    def test_source_pin_is_written_only_after_all_preflights_pass(self) -> None:
        launcher = self._launcher()
        # 2026-08-07 incident: the pin was written before the GPU preflight,
        # so a failed preflight left a poisoned root that refused the fixed
        # launcher. The pin write must be the last step before autopilot.
        pin_write = launcher.index("SOURCE_COMMIT_TMP")
        self.assertGreater(pin_write, launcher.index("production minimum"))
        self.assertGreater(
            pin_write, launcher.index("building optimized production binaries")
        )
        self.assertLess(pin_write, launcher.index('"${AUTOPILOT_ARGS[@]}"'))

    def test_gpu_preflight_admits_a_24gb_marketed_card(self) -> None:
        launcher = self._launcher()
        # An RTX 4090 reports 24,080 MiB — under 24 binary GiB. The trainer
        # uses a few GB; 20 GiB still rejects genuinely undersized GPUs.
        self.assertIn("20 * 1024**3", launcher)
        self.assertNotIn("24 * 1024**3", launcher)

    def test_supervisor_conf_carries_the_threadripper_resource_profile(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        # 7995WX host: 184 effective threads; 160 for the training lane,
        # 24 reserved for arena/A-B lanes and SSH responsiveness. Gate knobs
        # are identity-frozen and must NOT appear here.
        self.assertIn('SELFPLAY_PARALLEL_GAMES="160"', environment)
        self.assertIn('RELABEL_THREADS="160"', environment)
        self.assertIn('RELABEL_HASH_MB="8192"', environment)
        self.assertNotIn("GATE_", environment)


if __name__ == "__main__":
    unittest.main()
