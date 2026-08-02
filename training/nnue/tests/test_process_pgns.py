import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "process_pgns.py"
_spec = importlib.util.spec_from_file_location("training.nnue.process_pgns", MODULE_PATH)
process_pgns = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = process_pgns
_spec.loader.exec_module(process_pgns)  # type: ignore[attr-defined]


class FakeBoard:
    def __init__(self, fens: list[str], plies: list[int]) -> None:
        self._fens = fens
        self._plies = plies
        self._index = 0

    def fen(self) -> str:
        return self._fens[self._index]

    def ply(self) -> int:
        return self._plies[self._index]

    def push(self, _move: object) -> None:
        self._index += 1


class FakeGame:
    def __init__(self, headers: dict[str, str], board: FakeBoard, move_count: int) -> None:
        self.headers = headers
        self._board = board
        self._moves = [object() for _ in range(move_count)]
        self.board_called = False

    def board(self) -> FakeBoard:
        self.board_called = True
        return self._board

    def mainline_moves(self) -> list[object]:
        return self._moves


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ProcessPgnTests(unittest.TestCase):
    def test_multiple_inputs_share_shard_numbering_without_overwrite(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.pgn"
            second = root / "second.pgn"
            first.write_text("first source", encoding="utf-8")
            second.write_text("second source", encoding="utf-8")
            out_dir = root / "out"
            games = {
                first: [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["first-0", "first-1", "first-end"], [0, 1, 2]),
                        2,
                    )
                ],
                second: [
                    FakeGame(
                        {"Result": "0-1"},
                        FakeBoard(["second-0", "second-1", "second-end"], [0, 1, 2]),
                        2,
                    )
                ],
            }

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=lambda path: games[path]), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                process_pgns.process_paths(
                    [first, second], out_dir, shard_size=2, sample_every=1, max_games=0
                )

            shards = sorted(out_dir.glob("shard_*.jsonl"))
            self.assertEqual([path.name for path in shards], ["shard_000000.jsonl", "shard_000001.jsonl"])
            self.assertEqual([row["fen"] for row in _read_jsonl(shards[0])], ["first-0", "first-1"])
            self.assertEqual([row["fen"] for row in _read_jsonl(shards[1])], ["second-0", "second-1"])

    def test_setup_fen_board_turn_and_absolute_ply_are_preserved(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "setup.pgn"
            path.write_text("setup source", encoding="utf-8")
            out_dir = root / "out"
            start_fen = "8/8/8/8/8/8/4k3/7K b - - 17 42"
            game = FakeGame(
                {"Result": "1/2-1/2", "SetUp": "1", "FEN": start_fen},
                FakeBoard([start_fen, "after"], [83, 84]),
                1,
            )

            with mock.patch.object(process_pgns, "iter_games_from_pgn", return_value=[game]), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                process_pgns.process_paths([path], out_dir, shard_size=10, sample_every=1, max_games=0)

            rows = _read_jsonl(out_dir / "shard_000000.jsonl")
            self.assertTrue(game.board_called)
            self.assertEqual(rows, [{"fen": start_fen, "result": 0, "ply": 83}])

    def test_existing_shards_are_not_truncated(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("game source", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            existing = out_dir / "shard_000004.jsonl"
            existing.write_text('{"sentinel":true}\n', encoding="utf-8")
            game = FakeGame(
                {"Result": "1-0"},
                FakeBoard(["new", "after"], [0, 1]),
                1,
            )

            with mock.patch.object(process_pgns, "iter_games_from_pgn", return_value=[game]), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                process_pgns.process_paths([path], out_dir, shard_size=10, sample_every=1, max_games=0)

            self.assertEqual(existing.read_text(encoding="utf-8"), '{"sentinel":true}\n')
            self.assertEqual(_read_jsonl(out_dir / "shard_000005.jsonl")[0]["fen"], "new")

    def test_rerun_of_same_source_and_options_is_idempotent(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("unchanged source", encoding="utf-8")
            out_dir = root / "out"

            def fresh_games(_path: Path) -> list[FakeGame]:
                return [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["position-0", "position-1", "end"], [0, 1, 2]),
                        2,
                    )
                ]

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                first = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )
                before = {
                    shard.name: shard.read_text(encoding="utf-8")
                    for shard in out_dir.glob("shard_*.jsonl")
                }
                second = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )

            after = {
                shard.name: shard.read_text(encoding="utf-8")
                for shard in out_dir.glob("shard_*.jsonl")
            }
            manifest = json.loads(
                (out_dir / process_pgns.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(first, (1, 2))
            self.assertEqual(second, (0, 0))
            self.assertEqual(after, before)
            self.assertEqual(len(manifest["completed"]), 1)
            self.assertIsNone(manifest["pending"])

    def test_changed_conversion_options_are_new_work(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("same source", encoding="utf-8")
            out_dir = root / "out"

            def fresh_games(_path: Path) -> list[FakeGame]:
                return [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["position-0", "position-1", "end"], [0, 1, 2]),
                        2,
                    )
                ]

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                first = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )
                second = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=2, max_games=0
                )

            manifest = json.loads(
                (out_dir / process_pgns.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            shards = sorted(out_dir.glob("shard_*.jsonl"))
            self.assertEqual(first, (1, 2))
            self.assertEqual(second, (1, 1))
            self.assertEqual(len(shards), 2)
            self.assertEqual(len(manifest["completed"]), 2)

    def test_changed_source_content_is_new_work(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("version one", encoding="utf-8")
            out_dir = root / "out"

            def fresh_games(_path: Path) -> list[FakeGame]:
                return [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["position", "end"], [0, 1]),
                        1,
                    )
                ]

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )
                path.write_text("version two", encoding="utf-8")
                result = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )

            manifest = json.loads(
                (out_dir / process_pgns.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(result, (1, 1))
            self.assertEqual(len(list(out_dir.glob("shard_*.jsonl"))), 2)
            self.assertEqual(len(manifest["completed"]), 2)
            self.assertNotEqual(
                manifest["completed"][0]["source_sha256"],
                manifest["completed"][1]["source_sha256"],
            )

    def test_failed_commit_is_rolled_back_and_not_marked_complete(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("source", encoding="utf-8")
            out_dir = root / "out"

            def fresh_games(_path: Path) -> list[FakeGame]:
                return [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["position", "end"], [0, 1]),
                        1,
                    )
                ]

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it), \
                 mock.patch.object(process_pgns, "_move_staged_shard", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    process_pgns.process_paths(
                        [path], out_dir, shard_size=10, sample_every=1, max_games=0
                    )

            manifest = json.loads(
                (out_dir / process_pgns.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(list(out_dir.glob("shard_*.jsonl")), [])
            self.assertEqual(manifest["completed"], [])
            self.assertIsNone(manifest["pending"])

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it):
                result = process_pgns.process_paths(
                    [path], out_dir, shard_size=10, sample_every=1, max_games=0
                )

            self.assertEqual(result, (1, 1))
            self.assertEqual(len(list(out_dir.glob("shard_*.jsonl"))), 1)

    def test_failed_final_manifest_write_rolls_back_published_shards(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "game.pgn"
            path.write_text("source", encoding="utf-8")
            out_dir = root / "out"
            real_write = process_pgns._write_manifest_atomic
            calls = 0

            def fail_second_write(target_dir: Path, manifest: dict) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("manifest write failed")
                real_write(target_dir, manifest)

            def fresh_games(_path: Path) -> list[FakeGame]:
                return [
                    FakeGame(
                        {"Result": "1-0"},
                        FakeBoard(["position", "end"], [0, 1]),
                        1,
                    )
                ]

            with mock.patch.object(process_pgns, "iter_games_from_pgn", side_effect=fresh_games), \
                 mock.patch.object(process_pgns, "_tqdm", side_effect=lambda it, **_kw: it), \
                 mock.patch.object(process_pgns, "_write_manifest_atomic", side_effect=fail_second_write):
                with self.assertRaisesRegex(OSError, "manifest write failed"):
                    process_pgns.process_paths(
                        [path], out_dir, shard_size=10, sample_every=1, max_games=0
                    )

            manifest = json.loads(
                (out_dir / process_pgns.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(list(out_dir.glob("shard_*.jsonl")), [])
            self.assertEqual(manifest["completed"], [])
            self.assertIsNone(manifest["pending"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
