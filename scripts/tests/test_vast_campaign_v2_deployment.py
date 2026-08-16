#!/usr/bin/env python3
"""Static contract checks for the campaign_v2 Vast.ai deployment (plan section 4)."""

from __future__ import annotations

import configparser
import math
import os
import re
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
        # min_teacher_depth is env-wired (objective-identity field) and must
        # exceed the actor depth: self-play stamps teacher_depth = actor
        # depth on every row, so equality lets actor self-labels masquerade
        # as teacher labels (discovered 2026-08-08; silently present in v4).
        self.assertIn('MIN_TEACHER_DEPTH="${MIN_TEACHER_DEPTH:-5}"', launcher)
        self.assertIn('"--min-teacher-depth" "$MIN_TEACHER_DEPTH"', launcher)
        self.assertIn("(( MIN_TEACHER_DEPTH > SELFPLAY_DEPTH ))", launcher)
        # Teacher sample fraction must be configurable to match the relabel
        # cadence (every-Nth-ply relabeling yields ~1/N teacher rows).
        self.assertIn(
            'TEACHER_SAMPLE_FRACTION="${TEACHER_SAMPLE_FRACTION:-0.5}"', launcher
        )
        self.assertIn(
            '"--teacher-sample-fraction" "$TEACHER_SAMPLE_FRACTION"', launcher
        )

    def test_fresh_init_leaves_no_unguarded_checkpoint_reference(self) -> None:
        # 2026-08-08 deploy incident: FRESH_INIT guarded staging but a later
        # preflight verify_sha256 still dereferenced $INITIAL_CHECKPOINT and
        # crash-looped the supervisor. Every use of the checkpoint artifact
        # outside its declaration must sit inside a FRESH_INIT guard.
        launcher = self._launcher()
        guarded = 0
        for i, line in enumerate(launcher.splitlines()):
            if '"$INITIAL_CHECKPOINT"' not in line:
                continue
            # The reference must appear inside an `if [[ "$FRESH_INIT" != "1" ]]`
            # block: scan backwards for the guard before any closing `fi`.
            preceding = launcher.splitlines()[:i]
            depth = 0
            ok = False
            for prev in reversed(preceding):
                stripped = prev.strip()
                if stripped == "fi":
                    depth += 1
                elif stripped.startswith("if "):
                    if depth == 0:
                        ok = 'FRESH_INIT" != "1"' in stripped
                        break
                    depth -= 1
            self.assertTrue(ok, f"unguarded $INITIAL_CHECKPOINT use: {line.strip()}")
            guarded += 1
        self.assertGreaterEqual(guarded, 3)

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
        self.assertIn('"--gate-sprt-delta1" "0.0575"', launcher)
        self.assertIn('"--gate-sprt-alpha" "0.05"', launcher)
        self.assertIn('"--gate-sprt-beta" "0.05"', launcher)
        self.assertIn('"--gate-sprt-min-pairs" "48"', launcher)
        # Frozen constant, NOT env-overridable: the supervisor conf may size
        # gate throughput per host but may never move a statistical knob.
        self.assertIn("GATE_SPRT_BATCH_PAIRS=180", launcher)
        self.assertNotIn("GATE_SPRT_BATCH_PAIRS=${", launcher)
        self.assertIn('"--gate-sprt-batch-pairs" "$GATE_SPRT_BATCH_PAIRS"', launcher)
        self.assertIn('"--gate-sprt-max-pairs" "1600"', launcher)

    def test_launcher_warns_but_never_dies_on_gate_worker_clamp(self) -> None:
        """A batch smaller than GATE_PARALLEL_GAMES silently runs fewer workers
        than configured -- the original defect. Warn about it, but NEVER die():
        GATE_PARALLEL_GAMES is host-tunable in the supervisor conf, supervisor
        has autorestart, and a die() on a throughput preference would turn a
        harmless host edit into a crash-loop that halts a live lineage. The real
        safety check is EFFECTIVE_CPUS >= REQUIRED_CPUS.
        """
        launcher = self._launcher()
        self.assertIn(
            "if (( GATE_SPRT_BATCH_PAIRS < GATE_PARALLEL_GAMES )); then", launcher
        )
        clamp_block = launcher.split(
            "if (( GATE_SPRT_BATCH_PAIRS < GATE_PARALLEL_GAMES )); then", 1
        )[1].split("fi", 1)[0]
        self.assertIn("WARNING", clamp_block)
        self.assertNotIn("die ", clamp_block)

    def test_sprt_batch_does_not_clamp_the_configured_worker_count(self) -> None:
        """The batch must carry at least as many pairs as there are workers.

        compare_play dispatches opening PAIRS through rayon par_iter and bounds
        workers at min(parallel_games, available_cores, work_units) where
        work_units = games/2 (compare_play.rs:600-620). A batch with fewer pairs
        than GATE_PARALLEL_GAMES therefore runs fewer workers than configured --
        which is exactly the original defect: 24-pair batches pinned the gate to
        24 of 184 cores while the conf asked for 48.

        Measured 2026-08-15 (150 ms movetime, same net, CPU lane free):
            24 pairs / 24 workers   48 games   48.2 s   0.996 games/s
            90 pairs / 45 workers  180 games   90.8 s   1.981 games/s
           180 pairs / 90 workers  360 games   95.7 s   3.762 games/s
        Per-worker throughput is flat (0.0415 / 0.0440 / 0.0418), so aggregate
        throughput tracks WORKER COUNT and the batch is what makes workers
        reachable. 90 workers is the largest count measured, not a proven
        optimum. See evidence/gate_sprt_work_granularity_20260815.json.
        """
        launcher = self._launcher()
        batch_match = re.search(r"GATE_SPRT_BATCH_PAIRS=([0-9]+)", launcher)
        self.assertIsNotNone(batch_match, "launcher must set GATE_SPRT_BATCH_PAIRS")
        assert batch_match is not None
        batch_pairs = int(batch_match.group(1))

        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        match = re.search(r'GATE_PARALLEL_GAMES="([0-9]+)"', environment)
        self.assertIsNotNone(match, "supervisor conf must set GATE_PARALLEL_GAMES")
        assert match is not None
        workers = int(match.group(1))

        self.assertGreaterEqual(
            batch_pairs,
            workers,
            f"batch of {batch_pairs} pairs would clamp the gate to "
            f"{batch_pairs} workers although GATE_PARALLEL_GAMES={workers}",
        )
        # The container is capped by cgroup quota at 184, NOT the 192 that
        # nproc reports (192 is SMT threads on 96 physical cores). The launcher
        # preflight dies when REQUIRED_CPUS exceeds the effective count, and
        # supervisor autorestart turns that into a crash-loop.
        self.assertLessEqual(
            workers,
            184,
            f"GATE_PARALLEL_GAMES={workers} exceeds the 184-CPU cgroup cap and "
            "would crash-loop the launcher at preflight",
        )

    def _gate_flag(self, name: str) -> str:
        launcher = self._launcher()
        match = re.search(rf'"--{re.escape(name)}" "([0-9.]+)"', launcher)
        self.assertIsNotNone(match, f"launcher does not set --{name}")
        assert match is not None
        return match.group(1)

    def test_sprt_h1_targets_the_conventional_small_patch_elo_band(self) -> None:
        """delta1 must sit in the [0, 10]-Elo band conventional for this tier.

        Pair delta and logistic Elo are related by ``delta = 4p - 2``. The gate
        shipped with delta1 = 0.25, i.e. an H1 of +43.7 Elo and an indifference
        point of +21.8 Elo, which is far above any gain this trainer produces
        per promotion attempt.
        """
        delta1 = float(self._gate_flag("gate-sprt-delta1"))
        h1_elo = 400.0 * math.log10(
            ((delta1 + 2.0) / 4.0) / (1.0 - (delta1 + 2.0) / 4.0)
        )
        self.assertGreater(h1_elo, 5.0, "H1 below +5 Elo needs an infeasible sample")
        self.assertLess(
            h1_elo,
            15.0,
            f"H1 is {h1_elo:.1f} Elo; conventional bounds for this tier are [0, 10]",
        )

    def test_sprt_max_pairs_can_reach_a_verdict_at_the_design_point(self) -> None:
        """max_pairs and delta1 must be changed together, never delta1 alone.

        ``autopilot._gsprt_decision`` accumulates
        ``llr = pairs * delta1 * (mean - delta1 / 2) / variance`` and
        ``_run_model_gate`` converts an inconclusive SPRT into a **reject**
        (``autopilot.py`` "max-pairs-inconclusive"). So a delta1 small enough to
        detect real gains, paired with a max_pairs too small to accumulate the
        log-likelihood, turns the gate into an unconditional reject.

        At the H1 design point (``mean == delta1``) the test must be able to
        reach the accept bound before truncation. Variance is the paired-game
        variance measured over 500 pairs in
        ``evidence/gate_power_and_unpromoted_progress_20260814.json``.
        """
        delta1 = float(self._gate_flag("gate-sprt-delta1"))
        alpha = float(self._gate_flag("gate-sprt-alpha"))
        beta = float(self._gate_flag("gate-sprt-beta"))
        max_pairs = float(self._gate_flag("gate-sprt-max-pairs"))

        measured_pair_variance = 0.8504
        accept_bound = math.log((1.0 - beta) / alpha)
        pairs_needed = (
            2.0 * measured_pair_variance * accept_bound / (delta1 * delta1)
        )
        self.assertLess(
            pairs_needed,
            max_pairs,
            f"delta1={delta1} needs ~{pairs_needed:.0f} pairs to accept at its own "
            f"design point but max_pairs={max_pairs:.0f} truncates first, and a "
            f"truncated SPRT is recorded as a REJECT",
        )

    def test_gate_screen_filters_before_the_expensive_confirmation(self) -> None:
        """The screen is a resource filter, so its threshold must be positive.

        With a tight delta1 a *null* candidate needs as many pairs to reject as
        a real one needs to accept, so every cycle would pay the full
        confirmation. A positive screen threshold keeps that cost off the
        cycles that have nothing to promote.
        """
        threshold = float(self._gate_flag("gate-min-score-delta"))
        self.assertGreater(
            threshold, 0.0, "a zero screen threshold escalates ~half of all null cycles"
        )
        delta1 = float(self._gate_flag("gate-sprt-delta1"))
        self.assertLess(
            threshold,
            delta1,
            "screen must not reject candidates the confirmation would accept",
        )

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

    def test_v6_conf_mints_arch_v2_1024_lineage(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        # v8 (2026-08-16) superseded v7, which promoted 11 models and then
        # stalled for 28 cycles at +4.91 Elo [+1.66, +8.14] over its frozen
        # cycle-118 active model (12,800 gate games). The loop had NOT run out
        # of signal: a depth-7 search still disagreed with its own net by 617 cp
        # on average and on 57.5% of best moves. What ran out was the
        # OBJECTIVE'S ABILITY TO SEE that signal -- BCE through sigmoid(cp/400)
        # retains 28% of its gradient at 1000 cp and 18% at 1200, while 28% of
        # the labels sit above |600| cp, so the same signal shrank 26.5% when
        # viewed through the WDL target. See
        # evidence/objective_saturation_20260816.json.
        #
        # v8 keeps the architecture and the weights and changes only the
        # target: CP_LOSS_WEIGHT adds Huber on the normalised cp error, and
        # TEACHER_MIX drops below 1.0 so game outcome -- the one signal not
        # derived from the net's own search -- re-enters. Both are
        # objective-identity fields, hence a new OUT_ROOT with a weights-only
        # bootstrap from cycle 146 and fresh Adam (train_torch.py:235-237
        # permits the objective transition only when weights_only; :436-437
        # refuses to carry optimizer state across it).
        self.assertIn('OUT_ROOT="/workspace/piebot_campaign_v8"', environment)
        self.assertIn('TRAIN_ARCH="v2"', environment)
        self.assertIn('HIDDEN_DIM="1024"', environment)
        # Weights-only bootstrap, NOT fresh random: 146 cycles of learned
        # representation are kept; only the optimizer moments are discarded.
        self.assertIn('FRESH_INIT="0"', environment)
        self.assertIn("campaign_v8_bootstrap/cycle_000146_checkpoint.json", environment)
        self.assertIn('CP_LOSS_WEIGHT="1.0"', environment)
        # The outcome signal is back on.
        self.assertNotIn('TEACHER_MIX="1.0"', environment)
        # Held-out loss rose after epoch 1 in 8/8 measured cycles and epoch 3
        # was selected 0/8, so EPOCHS=3 discarded ~2/3 of GPU train time.
        self.assertIn('EPOCHS="1"', environment)
        self.assertIn('RELABEL_DEPTH="7"', environment)
        # Teacher signal density raised 2026-08-09. At every-6 relabeling with a
        # 0.15 teacher fraction only ~12% of the gradient was search evaluation
        # and the other ~88% was the outcome of a depth-5 self-play game, which
        # the net cannot predict (train accuracy sat at 0.509, chance). Neither
        # knob is an objective-identity field, so this stays inside the v6
        # lineage: weights and Adam state carry over.
        self.assertIn('RELABEL_EVERY="1"', environment)
        # Teacher/actor separation: actor depth 5 rows must NOT count as
        # teacher rows, and the teacher fraction matches the every-2 cadence.
        self.assertIn('MIN_TEACHER_DEPTH="6"', environment)
        self.assertIn('TEACHER_SAMPLE_FRACTION="1.0"', environment)
        self.assertIn('TARGET_CP="250"', environment)
        self.assertNotIn("TEACHER_EXTERNAL_QUANT_FILE", environment)
        # v8: actor, teacher and gate incumbent start as the cycle-146 arch-v2
        # net at blend 75 -- the strongest model v7 produced, not the h64 v1
        # net v7 bootstrapped from. Staged out of the campaign tree because
        # autopilot retention keeps only 8 cycles and would delete it.
        self.assertIn(
            'INITIAL_ACTIVE_MODEL_SOURCE="/workspace/campaign_v8_bootstrap/cycle_000146_nnue_quant.nnue"',
            environment,
        )
        self.assertIn('INITIAL_ACTIVE_MODEL_BLEND_PERCENT="75"', environment)

    def test_launcher_wires_train_arch(self) -> None:
        launcher = self._launcher()
        self.assertIn('TRAIN_ARCH="${TRAIN_ARCH:-v1}"', launcher)
        self.assertIn('"--train-arch" "$TRAIN_ARCH"', launcher)
        self.assertIn('require_autopilot_flag "--train-arch"', launcher)

    def test_teacher_mix_env_is_wired(self) -> None:
        launcher = self._launcher()
        self.assertIn('TEACHER_MIX="${TEACHER_MIX:-0.8}"', launcher)
        self.assertIn('"--teacher-mix" "$TEACHER_MIX"', launcher)
        self.assertIn('require_autopilot_flag "--teacher-mix"', launcher)

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
        # v7 reverted to the depth-7 teacher capped at depth-5's measured p95
        # (144k). The depth-9 shape consumed 86% of campaign compute for a
        # 0.0021-nat gain, so that budget was moved into optimizer steps.
        self.assertIn('RELABEL_MAX_NODES="144000"', environment)

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
        # 7995WX host: 184 effective threads. campaign_v7 runs the training
        # lane at 112 and leaves ~48 for the search-arm A/B farm plus ~24 for
        # SSH responsiveness (the v6 profile used 160 and starved that farm).
        self.assertIn('SELFPLAY_PARALLEL_GAMES="112"', environment)
        self.assertIn('RELABEL_THREADS="112"', environment)
        self.assertIn('RELABEL_HASH_MB="8192"', environment)

    def test_supervisor_conf_carries_no_statistical_gate_knob(self) -> None:
        """Only gate *throughput* may be tuned per host; the bar may not be.

        Anything that changes the promotion bar (SPRT bounds, screen size,
        screen threshold) is frozen in the launcher where it is reviewed and
        version-controlled with its justification. Allowing those in the
        supervisor conf would let a host-level edit manufacture an acceptance.
        """
        parser = configparser.ConfigParser()
        parser.read(SUPERVISOR)
        environment = parser["program:piebot_campaign_v2"]["environment"]
        for frozen in ("GATE_GAMES", "GATE_SPRT", "GATE_MIN_SCORE", "GATE_SEARCH_THREADS"):
            self.assertNotIn(frozen, environment, f"{frozen} must stay in the launcher")
        # Throughput only: the gate is the tail of every cycle, so it is sized
        # to the host's spare cores.
        self.assertIn('GATE_PARALLEL_GAMES="90"', environment)


if __name__ == "__main__":
    unittest.main()
