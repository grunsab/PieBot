import datetime as dt
from dataclasses import asdict
import gzip
import hashlib
import importlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue.tests.test_lc0_bin import _make_chunk, binmod


def good_record(**kwargs):
    planes = [0] * 104
    planes[0] = binmod.reverse_bits_in_bytes(1 << 12)
    planes[5] = binmod.reverse_bits_in_bytes(1 << 4)
    planes[11] = binmod.reverse_bits_in_bytes(1 << 60)
    return _make_chunk(planes=planes, best_q=.5, result_q=1., **kwargs)


def archive(path, games):
    with tarfile.open(path, 'w') as tf:
        for name, date, payload in games:
            compressed = gzip.compress(payload, mtime=0)
            info = tarfile.TarInfo(name)
            info.mtime = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()
            info.size = len(compressed)
            tf.addfile(info, io.BytesIO(compressed))
    return {'dest': str(path), 'url': 'https://example.com/test91/' + path.name,
            'suite': 'test91/', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size': path.stat().st_size, 'status': 'downloaded'}


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.corpus = importlib.import_module('training.nnue.lc0_corpus')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, entries, **kwargs):
        manifest = self.root / 'raw.json'
        manifest.write_text(json.dumps({'files': entries}))
        return self.corpus.prepare_corpus(manifest, self.root / 'corpus', since='2026-07-07',
             until='2026-09-08T00:00:00Z', min_free_bytes=0, chunk_positions=2,
             workers=kwargs.pop('workers',1),
             validation_fraction=kwargs.pop('validation_fraction', 0), **kwargs)

    def training_rows(self, manifest):
        obj = json.loads(manifest.read_text())
        return [json.loads(line) for chunk in obj['chunks']
                for line in gzip.open(chunk['path'], 'rt')]

    def test_member_dates_real_names_dedup_chunks_and_white_labels(self):
        a = archive(self.root/'a.tar', [
            ('training.old.gz', '2026-07-06T23:59:59', good_record()),
            ('training.1.gz', '2026-07-07T00:00:00', good_record()*3),
            ('training.future.gz', '2026-09-08T00:00:00', good_record())])
        b = archive(self.root/'b.tar', [
            ('path/training.1.gz', '2026-08-01T00:00:00', good_record()*3),
            ('training.2.gz', '2026-09-07T23:59:59', good_record(side_to_move_or_enpassant=1))])
        manifest = self.prepare([a,b])
        rows = self.training_rows(manifest)
        self.assertEqual(4, len(rows))
        self.assertEqual({'training.1.gz','training.2.gz'}, {r['game_id'] for r in rows})
        self.assertEqual(4, len({r['record_id'] for r in rows}))
        self.assertEqual(-.5, rows[-1]['best_q'])
        self.assertEqual(-1., rows[-1]['result_q'])
        self.assertEqual('white', rows[-1]['value_perspective'])
        obj = json.loads(manifest.read_text())
        self.assertTrue(all(c['positions'] <= 2 for c in obj['chunks']))
        self.assertEqual(1, obj['stats']['duplicate_games'])
        self.assertEqual(2, obj['stats']['outside_window_games'])
        self.assertEqual(manifest.read_bytes(), self.prepare([a,b]).read_bytes())

    def test_hash_holdout_is_game_level_and_global_sample_is_fixed(self):
        games = [(f'training.{i}.gz','2026-08-01T00:00:00',good_record()*3) for i in range(40)]
        a = archive(self.root/'a.tar', games[:20])
        b = archive(self.root/'b.tar', games[20:])
        path = self.prepare([a,b],validation_fraction=.5,validation_samples=5)
        obj = json.loads(path.read_text())
        val = [json.loads(line) for line in Path(obj['validation']['path']).read_text().splitlines()]
        self.assertEqual(5, len(val))
        training_games = {r['game_id'] for r in self.training_rows(path)}
        self.assertFalse(training_games & {r['game_id'] for r in val})
        for row in self.training_rows(path):
            self.assertFalse(self.corpus.is_holdout_game(row['run_id'],row['game_id'],20260907,.5))
        self.assertEqual(path.read_bytes(), self.prepare([a,b],validation_fraction=.5,validation_samples=5).read_bytes())

    def test_invalid_records_and_deletion_are_dropped(self):
        raw = bytearray(good_record())
        import struct
        struct.pack_into('<f',raw, 8+1858*4+104*8+8+4, float('nan'))
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',
                     good_record(invariance_info=1<<6)+bytes(raw)+good_record(invariance_info=1<<4))])
        rows = self.training_rows(self.prepare([a]))
        self.assertEqual(1,len(rows))
        self.assertFalse(rows[0]['outcome_valid'])

    def test_wrong_version_fails_and_archive_retry_does_not_duplicate(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        b = archive(self.root/'b.tar', [('training.2.gz','2026-08-01T00:00:00',good_record(version=7))])
        with self.assertRaisesRegex(ValueError, 'version'):
            self.prepare([a,b])
        with self.assertRaisesRegex(ValueError, 'version'):
            self.prepare([a,b])
        import sqlite3
        conn=sqlite3.connect(self.root/'corpus'/'progress.sqlite3')
        self.assertEqual(1, conn.execute('SELECT COUNT(*) FROM archives').fetchone()[0])
        self.assertEqual(1, conn.execute('SELECT COUNT(*) FROM games').fetchone()[0])
        conn.close()

    def test_checksum_mismatch_fails_before_decoding(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        a['sha256']='0'*64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.prepare([a])

    def test_resume_refuses_changed_configuration(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        self.prepare([a])
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.prepare([a],validation_fraction=.5)

    def test_truncated_game_rolls_back_and_staged_retry_is_clean(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record()),
                                      ('training.2.gz','2026-08-01T00:00:00',b'bad')])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.prepare([a])
        import sqlite3
        conn=sqlite3.connect(self.root/'corpus'/'progress.sqlite3')
        self.assertEqual(0,conn.execute('SELECT COUNT(*) FROM games').fetchone()[0])
        conn.close()
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.prepare([a])
        self.assertFalse((self.root/'corpus'/'corpus_manifest.json').exists())

    def test_resume_rejects_changed_training_cache(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        path = self.prepare([a])
        obj = json.loads(path.read_text())
        Path(obj['chunks'][0]['path']).write_bytes(b'bad cache')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.prepare([a])

    def test_reserve_prevents_any_archive_transaction(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        from types import SimpleNamespace
        with mock.patch.object(self.corpus.shutil,'disk_usage',return_value=SimpleNamespace(free=-1)):
            with self.assertRaisesRegex(OSError,'disk reserve'):
                self.prepare([a])

    def test_unsupported_variant_fails(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record(input_format=132))])
        with self.assertRaisesRegex(ValueError,'input format'):
            self.prepare([a])

    def test_value_only_reader_matches_full_decoder_without_policy(self):
        for kwargs in ({}, {'side_to_move_or_enpassant':1},
                       {'input_format':3,'invariance_info':1<<7},
                       {'input_format':5,'invariance_info':1<<4}):
            with self.subTest(kwargs=kwargs):
                payload=good_record(**kwargs)
                actual=list(self.corpus.iter_value_records(io.BytesIO(payload)))[0]
                reference=binmod.parse_v6_record(payload)
                self.assertEqual([],actual.probabilities)
                reference.probabilities=[]
                self.assertEqual(asdict(reference),asdict(actual))
                self.assertEqual(self.corpus._sample(reference),self.corpus._sample(actual))

    def test_parallel_game_conversion_matches_sequential_order_and_split(self):
        a = archive(self.root/'a.tar', [(f'training.{i}.gz','2026-08-01T00:00:00',good_record()*3) for i in range(12)])
        path=self.prepare([a],validation_fraction=.5,validation_samples=5,workers=1)
        expected=self.training_rows(path)
        ref=json.loads(path.read_text())
        val_ref=Path(ref['validation']['path']).read_bytes()
        parallel=self.corpus.prepare_corpus(self.root/'raw.json',self.root/'parallel',
                   since='2026-07-07',until='2026-09-08T00:00:00Z',min_free_bytes=0,
                   chunk_positions=2,workers=2,validation_fraction=.5,validation_samples=5)
        self.assertEqual(expected,self.training_rows(parallel))
        val=json.loads(parallel.read_text())
        self.assertEqual(val_ref,Path(val['validation']['path']).read_bytes())
        self.assertEqual(ref['corpus_id'],val['corpus_id'])

    def test_requested_validation_rejects_empty_holdout(self):
        a = archive(self.root/'a.tar', [('training.1.gz','2026-08-01T00:00:00',good_record())])
        with self.assertRaisesRegex(ValueError,'holdout'):
            self.prepare([a],validation_fraction=.01)


if __name__ == '__main__':
    unittest.main()
