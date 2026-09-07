#!/usr/bin/env python3
"""Focused contracts for the resumable PieBot/Stockfish UCI arena."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import uci_elo_arena as arena


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


class _FirstLegalEngine:
    def __init__(self, *, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.closed = False

    def play(self, board, _limit, *, game=None):
        del game
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(move=next(iter(board.legal_moves)))

    def close(self) -> None:
        self.closed = True


class ArenaPlanningTests(unittest.TestCase):
    def test_game_plans_are_deterministic_and_reverse_colors_per_opening(self) -> None:
        first = arena.build_game_plans([START_FEN, E4_FEN], games=6, seed=73)
        second = arena.build_game_plans([START_FEN, E4_FEN], games=6, seed=73)

        self.assertEqual(first, second)
        for pair_index in range(3):
            outbound, return_game = first[pair_index * 2 : pair_index * 2 + 2]
            self.assertEqual(outbound.pair_index, return_game.pair_index)
            self.assertEqual(outbound.opening_id, return_game.opening_id)
            self.assertEqual(outbound.opening_fen, return_game.opening_fen)
            self.assertEqual(outbound.piebot_color, "white")
            self.assertEqual(return_game.piebot_color, "black")

    def test_odd_game_count_is_rejected_for_strict_pairing(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            arena.build_game_plans([START_FEN], games=3, seed=1)

    def test_extracts_unique_final_fens_from_compare_play_json(self) -> None:
        payload = {
            "pairing": {
                "opening_policy": "neutral-pst-topk-v2",
                "openings": [
                    {"positions": [START_FEN, E4_FEN]},
                    {"positions": [START_FEN, E4_FEN]},
                    {"positions": [START_FEN]},
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(arena.load_opening_fens(path), [E4_FEN, START_FEN])


class ArenaStatisticsTests(unittest.TestCase):
    def test_logistic_elo_and_paired_bootstrap_are_reproducible(self) -> None:
        records = [
            {
                "pair_index": 0,
                "game_index": 0,
                "piebot_score": 1.0,
                "termination": "chess_checkmate",
            },
            {
                "pair_index": 0,
                "game_index": 1,
                "piebot_score": 0.0,
                "termination": "stockfish_time_forfeit",
            },
            {
                "pair_index": 1,
                "game_index": 2,
                "piebot_score": 1.0,
                "termination": "chess_checkmate",
            },
            {
                "pair_index": 1,
                "game_index": 3,
                "piebot_score": 1.0,
                "termination": "chess_checkmate",
            },
        ]

        summary_a = arena.summarize_results(records, bootstrap_samples=2_000, seed=991)
        summary_b = arena.summarize_results(records, bootstrap_samples=2_000, seed=991)

        self.assertEqual(summary_a, summary_b)
        self.assertEqual(summary_a["wins"], 3)
        self.assertEqual(summary_a["draws"], 0)
        self.assertEqual(summary_a["losses"], 1)
        self.assertEqual(summary_a["complete_pairs"], 2)
        self.assertAlmostEqual(summary_a["score_rate"], 0.75)
        self.assertAlmostEqual(
            summary_a["elo_difference"], 400.0 * math.log10(0.75 / 0.25)
        )
        self.assertEqual(summary_a["score_95_ci"], [0.5, 1.0])
        self.assertEqual(summary_a["elo_95_ci"][0], 0.0)
        self.assertTrue(math.isinf(summary_a["elo_95_ci"][1]))
        self.assertEqual(
            summary_a["termination_counts"],
            {"chess_checkmate": 3, "stockfish_time_forfeit": 1},
        )
        self.assertEqual(
            summary_a["pair_score_counts"],
            {"0.0": 0, "0.5": 0, "1.0": 1, "1.5": 0, "2.0": 1},
        )
        self.assertEqual(summary_a["pentanomial"], [0, 0, 1, 0, 1])

    def test_bootstrap_uses_only_complete_opening_pairs(self) -> None:
        records = [
            {"pair_index": 0, "game_index": 0, "piebot_score": 1.0},
            {"pair_index": 0, "game_index": 1, "piebot_score": 0.0},
            {"pair_index": 1, "game_index": 2, "piebot_score": 1.0},
        ]
        summary = arena.summarize_results(records, bootstrap_samples=100, seed=4)
        self.assertEqual(summary["complete_pairs"], 1)
        self.assertEqual(summary["score_95_ci"], [0.5, 0.5])


class ArenaResumeTests(unittest.TestCase):
    def test_state_resume_skips_completed_games_and_rejects_config_drift(self) -> None:
        plans = arena.build_game_plans([START_FEN], games=2, seed=7)
        config = {"games": 2, "seed": 7, "model_sha256": "abc"}

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "arena.json"
            state = arena.load_or_create_state(state_path, config, plans)
            state["games"].append(
                {
                    "game_index": 0,
                    "pair_index": 0,
                    "opening_id": plans[0].opening_id,
                    "piebot_color": "white",
                    "piebot_score": 0.5,
                }
            )
            arena.save_state(state_path, state)

            resumed = arena.load_or_create_state(state_path, config, plans)
            self.assertEqual([plan.game_index for plan in arena.pending_plans(resumed, plans)], [1])

            changed = dict(config, model_sha256="changed")
            with self.assertRaisesRegex(ValueError, "configuration"):
                arena.load_or_create_state(state_path, changed, plans)


class ArenaUciSafetyTests(unittest.TestCase):
    def test_piebot_options_pin_single_thread_nnue_and_model_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "candidate.nnue"
            model.write_bytes(b"known model bytes")
            options, digest = arena.piebot_uci_options(model, blend=75, hash_mb=64)

        self.assertEqual(options["Threads"], 1)
        self.assertEqual(options["Hash"], 64)
        self.assertEqual(options["UseNNUE"], True)
        self.assertEqual(options["EvalBlend"], 75)
        self.assertTrue(Path(options["NNUEQuantFile"]).is_absolute())
        self.assertEqual(digest, arena.sha256_bytes(b"known model bytes"))

    def test_stockfish_options_disable_ponder_multipv_and_tablebases(self) -> None:
        options = arena.stockfish_uci_options(elo=2600, hash_mb=64)
        self.assertEqual(options["Threads"], 1)
        self.assertEqual(options["Hash"], 64)
        self.assertEqual(options["UCI_LimitStrength"], True)
        self.assertEqual(options["UCI_Elo"], 2600)
        self.assertEqual(options["Ponder"], False)
        self.assertEqual(options["MultiPV"], 1)
        self.assertEqual(options["SyzygyProbeLimit"], 0)
        self.assertTrue(
            {"Ponder", "MultiPV", "SyzygyProbeLimit"}
            <= arena.STOCKFISH_REQUIRED_OPTIONS
        )

    def test_raw_preflight_fails_on_silent_piebot_nnue_fallback(self) -> None:
        fake_uci = textwrap.dedent(
            """
            import sys
            for raw in sys.stdin:
                line = raw.strip()
                if line == "uci":
                    print("id name Fake PieBot", flush=True)
                    for name in ("Threads", "Hash", "UseNNUE", "NNUEQuantFile", "EvalBlend"):
                        print(f"option name {name} type string default", flush=True)
                    print("uciok", flush=True)
                elif line.startswith("setoption name NNUEQuantFile"):
                    print("info string failed to load NNUEQuantFile: corrupt fixture", flush=True)
                elif line == "isready":
                    print("readyok", flush=True)
                elif line == "quit":
                    break
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_uci.py"
            script.write_text(fake_uci, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed to load NNUEQuantFile"):
                arena.run_uci_preflight(
                    [sys.executable, "-u", str(script)],
                    {
                        "Threads": 1,
                        "Hash": 64,
                        "NNUEQuantFile": "/tmp/model.nnue",
                        "UseNNUE": True,
                        "EvalBlend": 75,
                    },
                    required_options={
                        "Threads",
                        "Hash",
                        "UseNNUE",
                        "NNUEQuantFile",
                        "EvalBlend",
                    },
                    failure_markers=("failed to load NNUEQuantFile",),
                    timeout_s=2.0,
                )

    def test_default_wall_cap_leaves_headroom_for_60s_increment_games(self) -> None:
        # Campaign anchor conditions freeze the wall cap at 900s: at 60+0.5 a
        # long game can legitimately exceed 300s, and wall-cap draws bias the
        # calibration. This value is frozen for the campaign's lifetime.
        self.assertEqual(900.0, arena.DEFAULT_GAME_WALL_TIME_S)

    def test_preflight_rejects_spin_values_outside_advertised_range(self) -> None:
        fake_uci = textwrap.dedent(
            """
            import sys
            for raw in sys.stdin:
                line = raw.strip()
                if line == "uci":
                    print("id name Fake Stockfish", flush=True)
                    print(
                        "option name UCI_Elo type spin default 1320 min 1320 max 3190",
                        flush=True,
                    )
                    print(
                        "option name UCI_LimitStrength type check default false",
                        flush=True,
                    )
                    print("uciok", flush=True)
                elif line == "isready":
                    print("readyok", flush=True)
                elif line == "quit":
                    break
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_sf.py"
            script.write_text(fake_uci, encoding="utf-8")
            command = [sys.executable, "-u", str(script)]
            required = {"UCI_Elo", "UCI_LimitStrength"}

            with self.assertRaisesRegex(RuntimeError, r"UCI_Elo.*1000.*1320.*3190"):
                arena.run_uci_preflight(
                    command,
                    {"UCI_LimitStrength": True, "UCI_Elo": 1000},
                    required_options=required,
                    failure_markers=(),
                    timeout_s=5.0,
                )

            with self.assertRaisesRegex(RuntimeError, r"UCI_Elo.*3400.*1320.*3190"):
                arena.run_uci_preflight(
                    command,
                    {"UCI_LimitStrength": True, "UCI_Elo": 3400},
                    required_options=required,
                    failure_markers=(),
                    timeout_s=5.0,
                )

            # In-range values pass and the advertised range is reported.
            report = arena.run_uci_preflight(
                command,
                {"UCI_LimitStrength": True, "UCI_Elo": 2500},
                required_options=required,
                failure_markers=(),
                timeout_s=5.0,
            )
            self.assertEqual(
                {"min": 1320, "max": 3190},
                report["spin_ranges"]["UCI_Elo"],
            )

    def test_python_chess_startup_timeout_does_not_cap_legal_think_time(self) -> None:
        calls = []

        class FakeConfiguredEngine:
            timeout = None

            def configure(self, options) -> None:
                self.options = options

        configured = FakeConfiguredEngine()

        class FakeSimpleEngine:
            @staticmethod
            def popen_uci(command, timeout):
                calls.append((command, timeout))
                return configured

        modules = (object(), SimpleNamespace(SimpleEngine=FakeSimpleEngine))
        with patch.object(arena, "_chess_modules", return_value=modules):
            result = arena._open_python_chess_engine(
                ["/engine"],
                {"Threads": 1},
                startup_timeout_s=10.0,
                command_timeout_s=301.0,
            )

        self.assertIs(result, configured)
        self.assertEqual(calls, [(["/engine"], 10.0)])
        self.assertEqual(configured.timeout, 301.0)


class GenericOpponentTests(unittest.TestCase):
    def test_generic_opponent_applies_only_supported_full_strength_options(self):
        options = arena.generic_opponent_uci_options(
            ["Hash", "Threads", "Ponder", "MultiPV", "UCI_LimitStrength", "UCI_Elo", "OwnBook", "SyzygyPath"],
            hash_mb=64, single_thread_qualified=False,
        )
        self.assertEqual(options, {"Threads": 1, "Hash": 64, "Ponder": False,
            "MultiPV": 1, "UCI_LimitStrength": False, "OwnBook": False, "SyzygyPath": ""})
        self.assertNotIn("UCI_Elo", options)
        with self.assertRaisesRegex(ValueError, "single.thread"):
            arena.generic_opponent_uci_options(["Hash"], hash_mb=64, single_thread_qualified=False)
        self.assertEqual(arena.generic_opponent_uci_options(
            ["Hash"], hash_mb=64, single_thread_qualified=True), {"Hash": 64})
        self.assertEqual(arena.generic_opponent_uci_options(
            ["Hash", "Skill Level"], hash_mb=64, single_thread_qualified=True,
            spin_ranges={"Skill Level": {"min": 0, "max": 20}}), {"Hash": 64, "Skill Level": 20})

    def test_generic_raw_handshake_works_with_only_hash_option(self):
        fake_uci = textwrap.dedent("""
            import sys
            for raw in sys.stdin:
                line = raw.strip()
                if line == "uci":
                    print("id name Minimal UCI", flush=True)
                    print("option name Hash type spin default 16 min 1 max 256", flush=True)
                    print("uciok", flush=True)
                elif line.startswith("setoption") and line != "setoption name Hash value 64":
                    print("info string could not load unsupported option", flush=True)
                elif line == "isready":
                    print("readyok", flush=True)
                elif line == "quit":
                    break
        """)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "engine.py"
            script.write_text(fake_uci)
            import shlex
            arguments = arena.build_parser().parse_args([
                "--piebot-nnue", "/unused", "--results", str(Path(tmp) / "unused.json"),
                "--opponent-command", shlex.join([sys.executable, "-u", str(script)]),
                "--opponent-sha256", arena.sha256_file(Path(sys.executable)),
                "--opponent-name", "Minimal UCI", "--opponent-rating", "2700",
                "--opponent-rating-source", "https://example.test/rating",
                "--opponent-rating-snapshot", "2026-09-07", "--opponent-single-thread-qualified"])
            _, options, metadata = arena.prepare_generic_opponent(arguments)
            self.assertEqual(options, {"Hash": 64})
            self.assertEqual(metadata["uci_id"], ["id name Minimal UCI"])
            self.assertEqual(metadata["argument_files"], [{"path": str(script.resolve()), "sha256": arena.sha256_file(script)}])

    def test_identity_validation_rejects_changed_executable_network_or_wrapper_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "engine"
            network = Path(tmp) / "net.nnue"
            script = Path(tmp) / "engine.py"
            for path in (binary, network, script):
                path.write_bytes(path.name.encode())
            identity = arena.pinned_command_identity([str(binary), str(script)],
                                                    expected_sha256=arena.sha256_file(binary))
            network_pin = {"path": str(network), "sha256": arena.sha256_file(network)}
            arena.verify_pinned_artifacts(identity, [network_pin])
            for path in (binary, network, script):
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    arena.verify_pinned_artifacts(identity, [network_pin])
                path.write_bytes(original)

    def test_results_lock_prevents_two_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            with arena.results_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with arena.results_lock(path):
                        self.fail("second writer acquired arena lock")

    @staticmethod
    def pair(pair_index, fen, scores, termination="chess_checkmate"):
        return [{"game_index": 2 * pair_index + i, "pair_index": pair_index,
                 "opening_fen": fen, "opening_id": str(fen),
                 "piebot_color": color, "piebot_score": score, "termination": termination}
                for i, (color, score) in enumerate(zip(("white", "black"), scores))]

    def test_generic_statistics_exclude_incomplete_pairs_from_point_estimate(self):
        records = self.pair(0, START_FEN, [1, 0]) + self.pair(1, E4_FEN, [1, 1])[:1]
        summary = arena.summarize_generic_results(records, bootstrap_samples=100, seed=1, expected_games=4)
        self.assertEqual(summary["score_rate"], 0.5)
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["recorded_games"], 3)
        self.assertFalse(summary["measurement_eligible"])
        self.assertIn("incomplete-pairs", summary["ineligible_reasons"])

    def test_generic_statistics_cluster_repeated_exact_fens(self):
        records = self.pair(0, START_FEN, [1, 1]) + self.pair(1, START_FEN, [1, 1]) + self.pair(2, E4_FEN, [0, 0])
        summary = arena.summarize_generic_results(records, bootstrap_samples=2000, seed=13, expected_games=6)
        self.assertEqual(summary["complete_pairs"], 3)
        self.assertEqual(summary["complete_opening_clusters"], 2)
        self.assertEqual(summary["bootstrap_unit"], "exact-fen-cluster-v1")
        self.assertAlmostEqual(summary["score_rate"], 2 / 3)
        self.assertEqual(summary["score_95_ci"], [0.0, 1.0])
        self.assertTrue(summary["measurement_eligible"])

    def test_capped_failed_or_forfeited_matches_are_flagged(self):
        for termination in ("max_plies", "game_wall_time_cap", "opponent_start_failure", "opponent_time_forfeit"):
            with self.subTest(termination=termination):
                records = self.pair(0, START_FEN, [1, 0], termination) + self.pair(1, E4_FEN, [0.5, 0.5])
                summary = arena.summarize_generic_results(records, bootstrap_samples=100, seed=1, expected_games=4)
                self.assertFalse(summary["measurement_eligible"])
                self.assertIn("abnormal-terminations", summary["ineligible_reasons"])

    def test_generic_cli_runs_minimal_opponent_without_sf_options_and_pins_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, binary, network = root / "model.nnue", root / "opponent", root / "network.nnue"
            for path in (model, binary, network):
                path.write_bytes(path.name.encode())
            runtime = root / "runtime.dll"
            runtime.write_bytes(b"pinned runtime")
            binary.chmod(0o755)
            book = root / "book.fen"
            book.write_text(START_FEN + "\n" + E4_FEN + "\n")
            output = root / "arena.json"
            calls = []
            def preflight(command, options, **kwargs):
                calls.append((command, dict(options), kwargs))
                return {"id": ["id name Fixture"], "advertised_options": ["Hash", "EvalFile"], "spin_ranges": {}}
            def play(plan, settings, **kwargs):
                return dict(plan.as_dict(), piebot_score=0.5, termination="chess_stalemate", plies=0)
            arguments = ["--piebot-command", sys.executable, "--piebot-nnue", str(model),
                "--opponent-command", str(binary), "--opponent-sha256", arena.sha256_file(binary),
                "--opponent-name", "Fixture 1.0", "--opponent-rating", "2700",
                "--opponent-rating-source", "https://example.test/ratings",
                "--opponent-rating-snapshot", "2026-09-07", "--opponent-single-thread-qualified",
                "--opponent-network", "EvalFile=" + str(network),
                "--opponent-network-sha256", arena.sha256_file(network),
                "--opponent-asset", str(runtime), "--opponent-asset-sha256", arena.sha256_file(runtime),
                "--games", "4", "--openings-file", str(book), "--results", str(output), "--bootstrap-samples", "100"]
            with patch.object(arena, "run_uci_preflight", side_effect=preflight), \
                 patch.object(arena, "play_isolated_game", side_effect=play):
                self.assertEqual(arena.main(arguments), 0)
            state = json.loads(output.read_text())
            self.assertNotIn("stockfish", state["config"])
            opponent = state["config"]["opponent"]
            self.assertEqual(opponent["rating"], 2700)
            self.assertEqual(opponent["options"], {"Hash": 64, "EvalFile": str(network.resolve())})
            self.assertEqual(opponent["assets"], [{"path": str(runtime.resolve()), "sha256": arena.sha256_file(runtime)}])
            self.assertTrue(state["summary"]["measurement_eligible"])
            self.assertTrue(all("UCI_Elo" not in options for _, options, _ in calls))
            with patch.object(arena, "run_uci_preflight", side_effect=preflight), \
                 patch.object(arena, "play_isolated_game") as no_replay:
                self.assertEqual(arena.main(arguments), 0)
                no_replay.assert_not_called()
            # A static dependency changed while a game was running: refuse to
            # persist that game's result under the old binary/runtime identity.
            live_args = list(arguments)
            live_output = root / "during-game.json"
            live_args[live_args.index("--results") + 1] = str(live_output)
            def mutate_runtime(plan, settings, **kwargs):
                runtime.write_bytes(b"modified runtime")
                return play(plan, settings, **kwargs)
            with patch.object(arena, "run_uci_preflight", side_effect=preflight), \
                 patch.object(arena, "play_isolated_game", side_effect=mutate_runtime):
                self.assertEqual(arena.main(live_args), 2)
            self.assertFalse(live_output.exists())
            runtime.write_bytes(b"pinned runtime")
            network.write_bytes(b"changed")
            with patch.object(arena, "run_uci_preflight", side_effect=preflight):
                self.assertEqual(arena.main(arguments), 2)

    def test_stockfish_cli_keeps_existing_options_configuration_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, results = root / "model.nnue", root / "sf.json"
            model.write_bytes(b"NNUE fixture")
            arguments = ["--piebot-command", sys.executable, "--stockfish-command", sys.executable,
                "--piebot-nnue", str(model), "--stockfish-elo", "2400", "--games", "2",
                "--results", str(results), "--bootstrap-samples", "100"]
            calls = []
            def preflight(command, options, **kwargs):
                calls.append((dict(options), kwargs["required_options"]))
                return {"id": ["id name Legacy fixture"]}
            def play(plan, settings, **kwargs):
                return dict(plan.as_dict(), piebot_score=0.5, termination="chess_stalemate", plies=0)
            with patch.object(arena, "run_uci_preflight", side_effect=preflight), \
                 patch.object(arena, "play_isolated_game", side_effect=play):
                self.assertEqual(arena.main(arguments), 0)
            state = json.loads(results.read_text())
            self.assertEqual(calls[1], (arena.stockfish_uci_options(elo=2400, hash_mb=64), arena.STOCKFISH_REQUIRED_OPTIONS))
            self.assertIn("stockfish", state["config"])
            self.assertNotIn("opponent", state["config"])
            self.assertNotIn("measurement_mode", state["config"])
            self.assertEqual(state["summary"], arena._json_safe(arena.summarize_results(
                state["games"], bootstrap_samples=100, seed=arena.DEFAULT_SEED)))


class ArenaGameTests(unittest.TestCase):
    def test_max_game_length_is_adjudicated_as_a_draw(self) -> None:
        plan = arena.build_game_plans([START_FEN], games=2, seed=3)[0]
        settings = arena.GameSettings(
            initial_time_s=1.0,
            increment_s=0.0,
            max_plies=2,
            game_wall_time_s=2.0,
            timeout_grace_s=0.1,
        )
        record = arena.play_game(
            plan,
            settings,
            piebot_engine=_FirstLegalEngine(),
            stockfish_engine=_FirstLegalEngine(),
        )
        self.assertEqual(record["termination"], "max_plies")
        self.assertEqual(record["piebot_score"], 0.5)
        self.assertEqual(record["plies"], 2)

    def test_engine_crash_and_hang_are_recorded_as_forfeits(self) -> None:
        plan = arena.build_game_plans([START_FEN], games=2, seed=3)[0]
        normal = _FirstLegalEngine()
        crash = arena.play_game(
            plan,
            arena.GameSettings(1.0, 0.0, 10, 2.0, 0.05),
            piebot_engine=_FirstLegalEngine(error=RuntimeError("engine died")),
            stockfish_engine=normal,
        )
        self.assertEqual(crash["termination"], "piebot_engine_crash")
        self.assertEqual(crash["piebot_score"], 0.0)

        sleeper = _FirstLegalEngine(delay=0.05)
        timeout = arena.play_game(
            plan,
            arena.GameSettings(0.005, 0.0, 10, 2.0, 0.005),
            piebot_engine=sleeper,
            stockfish_engine=_FirstLegalEngine(),
        )
        self.assertEqual(timeout["termination"], "piebot_time_forfeit")
        self.assertEqual(timeout["piebot_score"], 0.0)
        self.assertTrue(sleeper.closed)

        wall_sleeper = _FirstLegalEngine(delay=0.05)
        moving_at_wall_cap = arena.play_game(
            plan,
            arena.GameSettings(1.0, 0.0, 10, 0.005, 0.1),
            piebot_engine=wall_sleeper,
            stockfish_engine=_FirstLegalEngine(),
        )
        self.assertEqual(moving_at_wall_cap["termination"], "piebot_time_forfeit")
        self.assertEqual(moving_at_wall_cap["piebot_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
