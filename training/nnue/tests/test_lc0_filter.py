import json
import random
import unittest

import chess

from training.nnue import lc0_filter


def row(fen, ply, game="g"):
    return json.dumps({"fen": fen, "ply": ply, "game_id": game}).encode() + b"\n"


class InCheckTests(unittest.TestCase):
    def test_matches_python_chess_on_random_playouts(self):
        rng = random.Random(20261002)
        checked = in_check = 0
        for _ in range(60):
            board = chess.Board()
            for _ in range(120):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
                checked += 1
                in_check += board.is_check()
                self.assertEqual(lc0_filter.in_check(board.fen()), board.is_check(), board.fen())
        self.assertGreater(checked, 3000)
        self.assertGreater(in_check, 50)

    def test_every_attacker_kind_and_blocked_rays(self):
        cases = {
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1": True,     # rook
            "4k3/8/8/8/7b/8/8/4K3 w - - 0 1": True,      # bishop
            "4k3/8/8/8/8/3n4/8/4K3 w - - 0 1": True,     # knight
            "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1": True,     # black pawn attacks downward
            "4k3/3P4/8/8/8/8/8/4K3 b - - 0 1": True,     # white pawn attacks upward
            "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1": True,     # queen
            "4k3/8/8/8/4r3/4P3/8/4K3 w - - 0 1": False,  # blocked file
            "4k3/8/8/8/8/8/4p3/4K3 w - - 0 1": False,    # pawn directly ahead does not attack
            "4k3/8/8/8/8/8/8/r3K3 b - - 0 1": False,     # the side NOT to move is the one attacked
        }
        for fen, expected in cases.items():
            self.assertEqual(lc0_filter.in_check(fen), expected, fen)


class KeepRowsTests(unittest.TestCase):
    START = chess.STARTING_FEN

    def game(self, moves, game="g", first_ply=0):
        board = chess.Board()
        rows = [row(board.fen(), first_ply, game)]
        for index, san in enumerate(moves, 1):
            board.push_san(san)
            rows.append(row(board.fen(), first_ply + index, game))
        return rows

    def kept_plies(self, rows, **options):
        return [json.loads(line)["ply"] for line in lc0_filter.keep_rows(rows, **options)]

    def test_no_filter_keeps_every_row_byte_for_byte(self):
        rows = self.game(["e4", "d5", "exd5"])
        self.assertEqual(list(lc0_filter.keep_rows(rows)), rows)

    def test_early_plies_are_skipped(self):
        rows = self.game(["e4", "e5", "Nf3", "Nc6"])
        self.assertEqual(self.kept_plies(rows, skip_early_plies=3), [3, 4])

    def test_in_check_positions_are_skipped(self):
        rows = self.game(["e4", "f5", "Qh5+", "g6"])
        self.assertEqual(self.kept_plies(rows, skip_in_check=True), [0, 1, 2, 4])

    def test_position_before_a_capture_is_skipped(self):
        # After 1.e4 d5 the move played is exd5: ply 2 is the pre-capture position.
        rows = self.game(["e4", "d5", "exd5", "Qxd5"])
        self.assertEqual(self.kept_plies(rows, skip_before_capture=True), [0, 1, 4])

    def test_capture_lookahead_never_crosses_games_or_ply_gaps(self):
        first = self.game(["e4", "d5"], game="a")
        # A different game that happens to follow with fewer pieces must not
        # mark the previous game's last row; nor may a row after a ply gap.
        second = [row("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 3, "b")]
        gap = [row(chess.Board().fen(), 0, "c"), row("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 2, "c")]
        kept = [json.loads(line) for line in
                lc0_filter.keep_rows(first + second + gap, skip_before_capture=True)]
        self.assertEqual([(r["game_id"], r["ply"]) for r in kept],
                         [("a", 0), ("a", 1), ("a", 2), ("b", 3), ("c", 0), ("c", 2)])

    def test_malformed_rows_pass_through_for_the_trainer_to_judge(self):
        rows = [b"not json\n", json.dumps({"fen": 1.5, "ply": 0, "game_id": "g"}).encode() + b"\n"]
        self.assertEqual(list(lc0_filter.keep_rows(rows, skip_in_check=True, skip_before_capture=True)), rows)

    def test_invalid_options_rejected(self):
        with self.assertRaises(ValueError):
            list(lc0_filter.keep_rows([], skip_early_plies=-1))


if __name__ == "__main__":
    unittest.main()
