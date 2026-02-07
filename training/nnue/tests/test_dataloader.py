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
        self.assertEqual(sample.ply, 7)
        self.assertEqual(sample.best_move, 'a2a3')
        self.assertEqual(sample.policy_top[0][0], 'a2a3')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
