"""Real archive preparation retains exact learning data while releasing raw copies."""
import gzip
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from training.nnue import lc0_archive_receipts as receipts, lc0_corpus as corpus
from training.nnue.tests.test_lc0_corpus import archive, good_record


class CorpusEvictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.output = self.root / 'corpus'
        self.manifest = self.raw / 'manifest.json'
        self.entries = [archive(self.raw / 'a.tar', [
            (f'training.{i}.gz', '2026-08-01T00:00:00', good_record(visits=i) * 3)
            for i in range(12)]), archive(self.raw / 'b.tar', [
            (f'training.{i}.gz', '2026-08-02T00:00:00', good_record(visits=i) * 2)
            for i in range(12, 24)])]
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(dict(schema='piebot-lc0-raw-v1', complete=True,
            failures=[], files=self.entries), indent=2, sort_keys=True) + '\n')

    def prepare(self, output=None, **kwargs):
        return corpus.prepare_corpus(self.manifest, output or self.output,
            since='2026-07-07', until='2026-09-08T00:00:00Z', min_free_bytes=kwargs.pop('min_free_bytes', 0),
            workers=1, chunk_positions=7, validation_fraction=.5, validation_samples=9,
            **kwargs)

    def rows(self, manifest):
        metadata = json.loads(manifest.read_text())
        return [json.loads(line) for chunk in metadata['chunks']
                for line in gzip.open(chunk['path'], 'rt')]

    def test_default_retains_raw_and_opt_in_preserves_exact_committed_corpus(self):
        result = self.prepare()
        before_manifest = result.read_bytes()
        protected = [self.manifest, self.output / 'progress.sqlite3',
                     self.output / 'identity.json', self.output / 'validation.jsonl',
                     *self.output.glob('archives/*/*')]
        before = {p: p.read_bytes() for p in protected}
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))
        self.assertEqual(result, self.prepare(evict_raw=True, raw_root=self.raw))
        self.assertFalse(any(Path(row['dest']).exists() for row in self.entries))
        self.assertEqual(result.read_bytes(), before_manifest)
        self.assertEqual({p: p.read_bytes() for p in protected}, before)
        with mock.patch.object(corpus, '_convert_archive', side_effect=AssertionError('reconverted committed data')):
            self.assertEqual(self.prepare(evict_raw=True, raw_root=self.raw).read_bytes(), before_manifest)

    def test_each_archive_is_evicted_before_next_conversion_with_identical_targets(self):
        reference = self.prepare(self.root / 'reference')
        expected_rows = self.rows(reference)
        expected = json.loads(reference.read_text())
        expected_validation = Path(expected['validation']['path']).read_bytes()
        convert = corpus._convert_archive
        calls = []
        def checked_convert(entry, stage, conn, config):
            if calls:
                self.assertFalse(Path(self.entries[0]['dest']).exists())
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM archives').fetchone()[0], 1)
                self.assertEqual(len(list((self.output / 'raw_evictions').glob('*.json'))), 1)
            self.assertTrue(Path(entry['dest']).exists())
            calls.append(entry['url'])
            return convert(entry, stage, conn, config)
        raw_bytes = self.manifest.read_bytes()
        with mock.patch.object(corpus, '_convert_archive', side_effect=checked_convert):
            result = self.prepare(evict_raw=True, raw_root=self.raw)
        actual = json.loads(result.read_text())
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.rows(result), expected_rows)
        self.assertEqual(Path(actual['validation']['path']).read_bytes(), expected_validation)
        self.assertEqual(actual['corpus_id'], expected['corpus_id'])
        self.assertEqual(actual['stats'], expected['stats'])
        self.assertEqual(self.manifest.read_bytes(), raw_bytes)
        self.assertFalse(any(Path(row['dest']).exists() for row in self.entries))

    def test_crash_after_archive_commit_resumes_without_reconversion(self):
        with mock.patch.object(receipts.ArchiveReceiptStore, 'evict', side_effect=RuntimeError('after commit')):
            with self.assertRaisesRegex(RuntimeError, 'after commit'):
                self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))
        with closing(sqlite3.connect(self.output / 'progress.sqlite3')) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM archives').fetchone()[0], 1)
        convert = corpus._convert_archive
        with mock.patch.object(corpus, '_convert_archive', wraps=convert) as calls:
            self.prepare(evict_raw=True, raw_root=self.raw)
            self.assertEqual(calls.call_count, 1)
        self.assertFalse(any(Path(row['dest']).exists() for row in self.entries))

    def test_empty_archive_is_committed_and_evicted_without_losing_other_games(self):
        empty = archive(self.raw / 'empty.tar', [])
        self.entries.insert(0, empty)
        self.write_manifest()
        result = self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertEqual(json.loads(result.read_text())['stats']['empty_archives'], 1)
        self.assertEqual(len(list((self.output / 'raw_evictions').glob('*.json'))), 3)
        self.assertFalse(Path(empty['dest']).exists())

    def test_derived_directory_is_durable_before_database_commit(self):
        events = []
        connect, replace, sync = sqlite3.connect, corpus.os.replace, receipts._fsync_directory
        class Connection:
            def __init__(self, connection):
                self.connection = connection
            def __getattr__(self, name):
                return getattr(self.connection, name)
            def commit(self):
                count = self.connection.execute('SELECT COUNT(*) FROM archives').fetchone()[0]
                events.append(('commit', count))
                return self.connection.commit()
        def moved(source, target):
            if Path(source).is_dir():
                events.append(('rename', str(source), str(target)))
            return replace(source, target)
        def synced(path):
            events.append(('sync', str(path)))
            return sync(path)
        with mock.patch.object(corpus.sqlite3, 'connect', side_effect=lambda *a, **k: Connection(connect(*a, **k))), \
                mock.patch.object(corpus.os, 'replace', side_effect=moved), \
                mock.patch.object(receipts, '_fsync_directory', side_effect=synced):
            self.prepare(evict_raw=True, raw_root=self.raw)
        first_rename = next(i for i, event in enumerate(events) if event[0] == 'rename')
        stage = events[first_rename][1]
        self.assertIn(('sync', stage), events[:first_rename])
        first_commit = events.index(('commit', 1))
        self.assertLess(first_rename, first_commit)
        self.assertIn(('sync', str(self.output / 'archives')), events[first_rename:first_commit])

    def test_failed_directory_publication_sync_rolls_back_and_resumes_cleanly(self):
        sync = receipts._fsync_directory
        def fail_parent(path):
            if Path(path) == self.output / 'archives':
                raise OSError('directory publication sync failed')
            return sync(path)
        with mock.patch.object(receipts, '_fsync_directory', side_effect=fail_parent):
            with self.assertRaises((OSError, receipts.ReceiptError)):
                self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))
        with closing(sqlite3.connect(self.output / 'progress.sqlite3')) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM archives').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM games').fetchone()[0], 0)
        self.assertFalse(list(self.output.glob('raw_evictions/*.json')))
        self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertFalse(any(Path(row['dest']).exists() for row in self.entries))

    def test_post_identity_crash_keeps_durable_lock_and_fetch_resume_works(self):
        from training.nnue import fetch_lc0_bins as fetch
        writer, sync = corpus.write_json_atomic, receipts._fsync_directory
        synced = []
        def record_sync(path):
            synced.append(Path(path))
            return sync(path)
        def interrupted_write(path, value):
            if Path(path) == self.output / 'identity.json':
                self.assertIn(self.output.parent, synced)
                with (self.output / 'prepare.lock').open('rb') as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer(path, value)
                raise RuntimeError('after identity publication')
            return writer(path, value)
        with mock.patch.object(receipts, '_fsync_directory', side_effect=record_sync), \
                mock.patch.object(corpus, 'write_json_atomic', side_effect=interrupted_write):
            with self.assertRaisesRegex(RuntimeError, 'after identity publication'):
                self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertTrue((self.output / 'identity.json').exists())
        self.assertFalse((self.output / 'progress.sqlite3').exists())
        before = self.manifest.read_bytes()
        with mock.patch.object(fetch, 'download_curl') as download:
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0,
                eviction_corpus=self.output, raw_root=self.raw), 0)
            download.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), before)
        self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertFalse(any(Path(row['dest']).exists() for row in self.entries))

    def test_failed_conversion_preserves_its_raw_and_cannot_publish_receipt(self):
        bad = archive(self.raw / 'bad.tar', [('training.bad.gz', '2026-08-01T00:00:00',
                                           good_record(version=7))])
        self.entries.insert(0, bad)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'version'):
            self.prepare(evict_raw=True, raw_root=self.raw)
        self.assertTrue(Path(bad['dest']).exists())
        self.assertFalse(list(self.output.glob('raw_evictions/*.json')))

    def test_eviction_requires_explicit_separate_raw_root_before_conversion(self):
        for kwargs in ({}, {'raw_root': self.root}, {'raw_root': self.output}):
            with self.subTest(kwargs=kwargs), mock.patch.object(corpus, '_convert_archive') as convert:
                with self.assertRaises((ValueError, receipts.ReceiptError)):
                    self.prepare(evict_raw=True, **kwargs)
                convert.assert_not_called()
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))

    def test_capacity_ceiling_stops_preparation_with_raw_preserved(self):
        with mock.patch.object(corpus.shutil, 'disk_usage',
                return_value=SimpleNamespace(total=600_000_000_000, free=150_000_000_000)), \
                mock.patch.object(corpus, '_convert_archive') as convert:
            with self.assertRaisesRegex(OSError, 'disk reserve'):
                self.prepare(evict_raw=True, raw_root=self.raw, capacity_bytes=478_000_000_000,
                             min_free_bytes=50 * 1024 ** 3)
            convert.assert_not_called()
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))

    def test_compressed_chunk_writer_checks_capacity_on_chunk_boundaries(self):
        usage = [SimpleNamespace(total=600_000_000_000, free=200_000_000_000)]
        writer = corpus._Chunks(self.root, 'train', 1, 50 * 1024 ** 3,
                                capacity_bytes=478_000_000_000)
        with mock.patch.object(corpus.shutil, 'disk_usage', side_effect=lambda _: usage[0]):
            try:
                writer.write({'record_id': 'first'})
                usage[0] = SimpleNamespace(total=600_000_000_000, free=150_000_000_000)
                with self.assertRaisesRegex(OSError, 'disk reserve'):
                    writer.write({'record_id': 'second'})
            finally:
                writer.close()
        self.assertEqual(len(writer.entries), 1)

    def test_real_conversion_rechecks_ceiling_before_committing_or_evicting_archive(self):
        usage = [SimpleNamespace(total=600_000_000_000, free=200_000_000_000)]
        write = corpus._Chunks.write
        written = []

        def consume_capacity(writer, sample):
            result = write(writer, sample)
            written.append(sample)
            usage[0] = SimpleNamespace(total=600_000_000_000, free=150_000_000_000)
            return result

        with mock.patch.object(corpus.shutil, 'disk_usage', side_effect=lambda _: usage[0]), \
                mock.patch.object(corpus._Chunks, 'write', autospec=True, side_effect=consume_capacity):
            with self.assertRaisesRegex(OSError, 'disk reserve'):
                self.prepare(evict_raw=True, raw_root=self.raw,
                             capacity_bytes=478_000_000_000, min_free_bytes=50 * 1024 ** 3)
        self.assertTrue(written, 'the ceiling must be crossed during real conversion')
        self.assertTrue(all(Path(row['dest']).exists() for row in self.entries))
        self.assertFalse(list(self.output.glob('raw_evictions/*.json')))
        with closing(sqlite3.connect(self.output / 'progress.sqlite3')) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM archives').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM games').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
