import json
import tempfile
import unittest
from pathlib import Path

from training.nnue import dataloader


class DataloaderTests(unittest.TestCase):
    def test_read_jsonl_dir_reads_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / 'shard_000000.jsonl'
            with file_path.open('w', encoding='utf-8') as handle:
                handle.write(json.dumps({'fen': '8/8/8/8/8/8/P7/8 w - - 0 0', 'result': 1}) + '\n')
            records = list(dataloader.read_jsonl_dir(tmp))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['result'], 1)

    def test_jsonl_to_training_samples_handles_new_schema(self) -> None:
        record = {
            'fen': '8/8/8/8/8/8/P7/8 w - - 0 0',
            'result_q': 0.75,
            'value_cp': 123.5,
            'teacher_depth': 6,
            'run_id': 'run-42',
            'game_id': 'run-42-game-7',
            'ply': 7,
            'played_move': 'a2a4',
            'target_best_move': 'a2a3',
            'best_move': 'a2a4',
            'policy_top': [{'move': 'a2a3', 'p': 0.9}],
        }
        samples = list(dataloader.jsonl_to_training_samples([record]))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.fen, record['fen'])
        self.assertEqual(sample.result, 1)
        self.assertAlmostEqual(sample.result_q, 0.75)
        self.assertAlmostEqual(sample.value_cp, 123.5)
        self.assertEqual(sample.teacher_depth, 6)
        self.assertEqual(sample.run_id, 'run-42')
        self.assertEqual(sample.game_id, 'run-42-game-7')
        self.assertEqual(sample.ply, 7)
        self.assertEqual(sample.best_move, 'a2a3')
        self.assertEqual(sample.policy_top[0][0], 'a2a3')
        self.assertTrue(sample.outcome_valid)

    def test_teacher_depth_and_ids_are_optional_and_strictly_typed(self) -> None:
        record = {
            'fen': '8/8/8/8/8/8/4K3/7k w - - 0 1',
            'result': 0,
            'teacher_depth': '6',
            'run_id': 42,
            'game_id': None,
        }
        sample = next(dataloader.jsonl_to_training_samples([record]))
        self.assertIsNone(sample.teacher_depth)
        self.assertIsNone(sample.run_id)
        self.assertIsNone(sample.game_id)

    def test_outcome_valid_is_backward_compatible_and_can_be_false(self) -> None:
        records = [
            {'fen': '8/8/8/8/8/8/4K3/7k w - - 0 1', 'result': 0},
            {
                'fen': '8/8/8/8/8/8/4K3/7k w - - 0 1',
                'result': 0,
                'outcome_valid': False,
            },
        ]
        samples = list(dataloader.jsonl_to_training_samples(records))
        self.assertTrue(samples[0].outcome_valid)
        self.assertFalse(samples[1].outcome_valid)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()


class GzippedShardTests(unittest.TestCase):
    """A gzipped shard must be indistinguishable from its plain twin.

    Self-play rows are 374.5 bytes raw and compress 10.35x, so accumulating a
    corpus is only affordable compressed: 1e9 rows is ~36 GB gzipped against
    ~319 GB raw on a 150 GB disk. If the two paths ever diverge, the training
    corpus silently changes meaning depending on how it happened to be stored.
    """

    @staticmethod
    def _rows():
        return [
            {"fen": "rn1qkbnr/ppp2ppp/4p3/3p1b2/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 4",
             "ply": 0, "result": 0, "value_cp": 23.0, "teacher_depth": 5,
             "policy_top": [{"move": "b8c6", "p": 0.0464}]},
            {"fen": "8/8/8/8/8/8/4K3/7k w - - 99 50",
             "ply": 41, "result": -1, "value_cp": -1180.5, "teacher_depth": 7,
             "policy_top": []},
        ]

    def test_gzipped_shard_reads_identically(self) -> None:
        import gzip as _gzip

        rows = self._rows()
        payload = "".join(json.dumps(r) + "\n" for r in rows)
        with tempfile.TemporaryDirectory() as tmp:
            plain_dir = Path(tmp) / "plain"
            gz_dir = Path(tmp) / "gz"
            plain_dir.mkdir()
            gz_dir.mkdir()
            (plain_dir / "shard_000000.jsonl").write_text(payload, encoding="utf-8")
            with _gzip.open(gz_dir / "shard_000000.jsonl.gz", "wt", encoding="utf-8") as fh:
                fh.write(payload)

            from_plain = list(dataloader.read_jsonl_dir(str(plain_dir)))
            from_gz = list(dataloader.read_jsonl_dir(str(gz_dir)))

        self.assertEqual(rows, from_plain)
        self.assertEqual(from_plain, from_gz)

    def test_mixed_directory_keeps_shard_order(self) -> None:
        import gzip as _gzip

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard_000000.jsonl").write_text(
                json.dumps({"fen": "a", "ply": 0}) + "\n", encoding="utf-8"
            )
            with _gzip.open(root / "shard_000001.jsonl.gz", "wt", encoding="utf-8") as fh:
                fh.write(json.dumps({"fen": "b", "ply": 1}) + "\n")
            (root / "shard_000002.jsonl").write_text(
                json.dumps({"fen": "c", "ply": 2}) + "\n", encoding="utf-8"
            )
            # A stray file must be ignored rather than parsed as records.
            (root / "notes.txt").write_text("ignore me", encoding="utf-8")

            got = [r["fen"] for r in dataloader.read_jsonl_dir(str(root))]

        self.assertEqual(["a", "b", "c"], got)
