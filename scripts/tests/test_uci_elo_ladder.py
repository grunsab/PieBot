import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uci_elo_ladder as ladder


class PooledEstimateTests(unittest.TestCase):
    def test_single_rung_even_score_estimates_the_rung(self) -> None:
        result = ladder.pooled_elo_estimate(
            [{"elo": 2000, "score_points": 50.0, "games": 100}], seed=1
        )
        self.assertFalse(result["degenerate"])
        self.assertAlmostEqual(2000.0, result["estimate"], delta=1.0)
        lo, hi = result["ci_95"]
        self.assertLess(lo, result["estimate"])
        self.assertGreater(hi, result["estimate"])

    def test_two_rungs_pool_toward_the_consistent_strength(self) -> None:
        # 75% against 1500 implies ~1690; 50% against 1700 implies 1700.
        result = ladder.pooled_elo_estimate(
            [
                {"elo": 1500, "score_points": 75.0, "games": 100},
                {"elo": 1700, "score_points": 50.0, "games": 100},
            ],
            seed=1,
        )
        self.assertFalse(result["degenerate"])
        self.assertGreater(result["estimate"], 1650.0)
        self.assertLess(result["estimate"], 1750.0)

    def test_more_games_tighten_the_confidence_interval(self) -> None:
        small = ladder.pooled_elo_estimate(
            [{"elo": 2000, "score_points": 30.0, "games": 60}], seed=3
        )
        large = ladder.pooled_elo_estimate(
            [{"elo": 2000, "score_points": 300.0, "games": 600}], seed=3
        )
        small_width = small["ci_95"][1] - small["ci_95"][0]
        large_width = large["ci_95"][1] - large["ci_95"][0]
        self.assertLess(large_width, small_width)

    def test_perfect_score_is_degenerate_with_no_estimate(self) -> None:
        result = ladder.pooled_elo_estimate(
            [{"elo": 1800, "score_points": 60.0, "games": 60}], seed=1
        )
        self.assertTrue(result["degenerate"])
        self.assertIsNone(result["estimate"])

    def test_zero_score_is_degenerate_with_no_estimate(self) -> None:
        result = ladder.pooled_elo_estimate(
            [{"elo": 2600, "score_points": 0.0, "games": 60}], seed=1
        )
        self.assertTrue(result["degenerate"])
        self.assertIsNone(result["estimate"])


class RungCommandTests(unittest.TestCase):
    def _args(self, **overrides) -> argparse.Namespace:
        base = dict(
            piebot_command="PieBot/target/release/uci",
            piebot_nnue="/models/net.nnue",
            piebot_blend=25,
            stockfish_command="stockfish16",
            rungs=[1320, 1500],
            games=60,
            time_control="60+0.5",
            seed=7,
            out_dir=Path("/tmp/ladder_out"),
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_one_command_per_rung_with_distinct_results_files(self) -> None:
        commands = ladder.build_rung_commands(self._args())
        self.assertEqual(2, len(commands))
        results_paths = set()
        for command, rung in zip(commands, (1320, 1500)):
            joined = " ".join(command)
            self.assertIn("uci_elo_arena.py", joined)
            self.assertIn(f"--stockfish-elo {rung}", joined)
            self.assertIn("--games 60", joined)
            self.assertIn("--time-control 60+0.5", joined)
            self.assertIn("--piebot-blend 25", joined)
            index = command.index("--results")
            results_paths.add(command[index + 1])
        self.assertEqual(2, len(results_paths))

    def test_rung_seeds_differ_so_openings_are_independent(self) -> None:
        commands = ladder.build_rung_commands(self._args())
        seeds = set()
        for command in commands:
            index = command.index("--seed")
            seeds.add(command[index + 1])
        self.assertEqual(2, len(seeds))


if __name__ == "__main__":
    unittest.main()
