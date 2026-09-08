"""Raw eviction requires a durable, independently rechecked corpus commit."""
from contextlib import closing
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from training.nnue import lc0_archive_receipts as receipts


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(data):
    return digest(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())


class ArchiveReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw_root = self.root / 'raw'
        self.corpus = self.root / 'corpus'
        self.raw = self.raw_root / 'test91' / 'source.tar'
        self.raw.parent.mkdir(parents=True)
        self.corpus.mkdir()
        (self.corpus / 'prepare.lock').touch()
        self.raw.write_bytes(b'verified raw archive fixture')
        self.entry = dict(url='https://example.test/source.tar', dest=str(self.raw),
                          size=self.raw.stat().st_size, sha256=digest(self.raw.read_bytes()),
                          suite='test91', status='downloaded')
        self.manifest = self.raw_root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(schema='piebot-lc0-raw-v1', complete=True,
                                                failures=[], files=[self.entry])))
        self.config = dict(schema='piebot-lc0-corpus-v1', since='2026-07-07T00:00:00+00:00',
                           until='2026-09-07T00:00:00+00:00', seed=7, chunk_positions=700000,
                           sources=[{k: self.entry[k] for k in ('url', 'sha256', 'size', 'suite')}])
        self.corpus_id = identity(self.config)
        (self.corpus / 'identity.json').write_text(json.dumps(dict(self.config, corpus_id=self.corpus_id)))
        self.key = identity([self.entry['url'], self.entry['sha256']])
        self.directory = self.corpus / 'archives' / ('000000_' + self.key[:16])
        self.directory.mkdir(parents=True)
        self.train = self.directory / 'train_000000.jsonl.gz'
        self.holdout = self.directory / 'holdout_000000.jsonl.gz'
        for path in (self.train, self.holdout):
            path.write_bytes(gzip.compress(b'{"record_id":"fixture"}\n', mtime=0))
        self.info = dict(chunks=[self.chunk(self.train)], holdout=[self.chunk(self.holdout)],
                         stats=dict(training_positions=1, holdout_positions=1),
                         collection_dates=['2026-08-01'])
        self.db = self.corpus / 'progress.sqlite3'
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.executescript('CREATE TABLE games(game_key TEXT PRIMARY KEY); '
                'CREATE TABLE source_aliases(source_key TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL); '
                'CREATE TABLE archives(archive_key TEXT PRIMARY KEY, directory TEXT, metadata TEXT);')
            conn.execute('INSERT INTO games VALUES(?)', ('game',))
            conn.execute('INSERT INTO source_aliases VALUES(?,?)', ('alias', 'game'))
        self.write_commit()
        self.receipt_path = self.corpus / 'raw_evictions' / (self.key + '.json')

    def chunk(self, path):
        return dict(name=path.name, positions=1, sha256=digest(path.read_bytes()))

    def write_commit(self):
        (self.directory / 'archive.json').write_text(json.dumps(self.info))
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('INSERT OR REPLACE INTO archives VALUES(?,?,?)',
                         (self.key, str(self.directory), json.dumps(self.info, sort_keys=True)))

    def store(self, **kwargs):
        return receipts.ArchiveReceiptStore(self.manifest, self.corpus,
                                           raw_root=kwargs.get('raw_root', self.raw_root))

    def test_verified_eviction_preserves_manifest_database_and_derived_bytes(self):
        protected = [self.manifest, self.db, self.train, self.holdout,
                     self.directory / 'archive.json', self.corpus / 'identity.json']
        before = {p: p.read_bytes() for p in protected}
        with self.store() as store:
            result = store.evict(self.key)
            self.assertTrue(result['raw_deleted'])
            self.assertEqual(result['warnings'], [])
            verified = store.verify(self.key)
            self.assertFalse(verified['raw_present'])
            self.assertEqual(verified['receipt']['corpus_id'], self.corpus_id)
            self.assertEqual(verified['receipt']['raw_manifest_sha256'], digest(before[self.manifest]))
            self.assertEqual(verified['receipt']['raw']['sha256'], self.entry['sha256'])
            self.assertEqual(len(verified['receipt']['derived_files']), 2)
        self.assertFalse(self.raw.exists())
        self.assertTrue(self.receipt_path.is_file())
        self.assertEqual({p: p.read_bytes() for p in protected}, before)

    def test_resume_is_idempotent_and_does_not_open_deleted_raw(self):
        with self.store() as store:
            store.evict(self.key)
        before = self.receipt_path.read_bytes()
        with self.store() as store:
            self.assertFalse(store.evict(self.key)['raw_deleted'])
            self.assertFalse(store.verify(self.key)['raw_present'])
        self.assertEqual(before, self.receipt_path.read_bytes())

    def test_uncommitted_archive_row_cannot_authorize_deletion(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('DELETE FROM archives')
        writer = sqlite3.connect(self.db)
        self.addCleanup(writer.close)
        writer.execute('INSERT INTO archives VALUES(?,?,?)',
                       (self.key, str(self.directory), json.dumps(self.info)))
        with self.store() as store, self.assertRaises(receipts.ReceiptError):
            store.evict(self.key)
        self.assertTrue(self.raw.is_file())
        self.assertFalse(self.receipt_path.exists())

    def test_missing_raw_without_receipt_is_not_retroactively_certified(self):
        self.raw.unlink()
        with self.store() as store, self.assertRaises(receipts.ReceiptError):
            store.evict(self.key)
        self.assertFalse(self.receipt_path.exists())

    def test_missing_or_corrupt_derived_files_fail_closed(self):
        for path in (self.train, self.holdout, self.directory / 'archive.json'):
            original = path.read_bytes()
            for replacement in (None, b'corrupt'):
                with self.subTest(path=path.name, replacement=replacement):
                    if replacement is None:
                        path.unlink()
                    else:
                        path.write_bytes(replacement)
                    with self.store() as store, self.assertRaises(receipts.ReceiptError):
                        store.evict(self.key)
                    self.assertTrue(self.raw.exists())
                    self.assertFalse(self.receipt_path.exists())
                    path.write_bytes(original)

    def test_receipt_is_rechecked_against_current_manifest_identity_row_and_chunks(self):
        with self.store() as store:
            store.evict(self.key)
        cases = [self.manifest, self.corpus / 'identity.json', self.train,
                 self.holdout, self.directory / 'archive.json', self.receipt_path]
        for path in cases:
            before = path.read_bytes()
            with self.subTest(path=path.name):
                path.write_bytes(before + b'changed')
                with self.store() as store, self.assertRaises(receipts.ReceiptError):
                    store.verify(self.key)
                path.write_bytes(before)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('DELETE FROM archives')
        with self.store() as store, self.assertRaises(receipts.ReceiptError):
            store.verify(self.key)

    def test_changed_raw_checksum_is_retained(self):
        self.raw.write_bytes(b'wrong archive bytes')
        with self.store() as store, self.assertRaises(receipts.ReceiptError):
            store.evict(self.key)
        self.assertTrue(self.raw.exists())
        self.assertFalse(self.receipt_path.exists())

    def test_regular_files_and_ancestors_cannot_be_symlinks(self):
        for path in (self.raw, self.train, self.corpus / 'identity.json', self.db):
            with self.subTest(path=path.name):
                target = self.root / 'target'
                path.rename(target)
                path.symlink_to(target)
                with self.assertRaises(receipts.ReceiptError):
                    with self.store() as store:
                        store.evict(self.key)
                self.assertTrue(target.is_file())
                path.unlink()
                target.rename(path)
        alias = self.root / 'raw_alias'
        alias.symlink_to(self.raw_root, target_is_directory=True)
        with self.assertRaises(receipts.ReceiptError):
            with self.store(raw_root=alias):
                self.fail('symlink root was accepted')

    def test_inventory_path_escape_and_part_file_are_never_deleted(self):
        for destination in (self.root / 'outside.tar', self.raw.parent / 'source.tar.part'):
            with self.subTest(destination=destination):
                destination.write_bytes(self.raw.read_bytes())
                changed = dict(self.entry, dest=str(destination))
                self.manifest.write_text(json.dumps(dict(schema='piebot-lc0-raw-v1', complete=True,
                                                        failures=[], files=[changed])))
                with self.store() as store, self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
                self.assertTrue(destination.exists())
                self.assertTrue(self.raw.exists())

    def test_archive_and_chunk_paths_must_belong_to_committed_namespace(self):
        original = dict(self.info['chunks'][0])
        for name in ('../train_000000.jsonl.gz', '/tmp/train_000000.jsonl.gz', 'checkpoint.json'):
            with self.subTest(name=name):
                self.info['chunks'][0] = dict(original, name=name)
                self.write_commit()
                with self.store() as store, self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
                self.assertTrue(self.raw.exists())

    def test_exclusive_existing_prepare_lock_prevents_two_owners(self):
        with self.store():
            with self.assertRaises(receipts.ReceiptError):
                with self.store():
                    self.fail('second lock owner accepted')

    def test_methods_require_context_lock(self):
        store = self.store()
        with self.assertRaises(receipts.ReceiptError):
            store.evict(self.key)

    def test_fsync_or_receipt_publication_failure_keeps_raw(self):
        for operation in ('_sync_committed', '_publish_receipt'):
            with self.subTest(operation=operation), self.store() as store:
                with mock.patch.object(store, operation, side_effect=OSError('injected durability fault')):
                    with self.assertRaises(receipts.ReceiptError):
                        store.evict(self.key)
                self.assertTrue(self.raw.exists())
                self.assertFalse(self.receipt_path.exists())

    def test_receipt_and_derived_durability_precede_unlink(self):
        with self.store() as store:
            events = []
            sync, publish, unlink = store._sync_committed, store._publish_receipt, store._unlink_raw
            def record_sync(*args):
                events.append('sync'); return sync(*args)
            def record_publish(*args):
                events.append('publish'); return publish(*args)
            def record_unlink(*args):
                events.append('unlink')
                self.assertTrue(self.receipt_path.is_file())
                self.assertTrue(store.verify(self.key)['raw_present'])
                return unlink(*args)
            with mock.patch.object(store, '_sync_committed', side_effect=record_sync), \
                 mock.patch.object(store, '_publish_receipt', side_effect=record_publish), \
                 mock.patch.object(store, '_unlink_raw', side_effect=record_unlink):
                store.evict(self.key)
            self.assertEqual(events, ['sync', 'publish', 'unlink'])

    def test_receipt_survives_failed_unlink_and_can_resume(self):
        with self.store() as store:
            with mock.patch.object(store, '_unlink_raw', side_effect=OSError('unlink refused')):
                with self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
            self.assertTrue(self.raw.exists())
            before = self.receipt_path.read_bytes()
            self.assertTrue(store.verify(self.key)['raw_present'])
            self.assertTrue(store.evict(self.key)['raw_deleted'])
            self.assertEqual(before, self.receipt_path.read_bytes())

    def test_raw_replacement_after_receipt_publication_is_not_deleted(self):
        with self.store() as store:
            publish = store._publish_receipt
            def replace_raw(*args):
                result = publish(*args)
                replacement = self.raw.with_name('replacement')
                replacement.write_bytes(self.raw.read_bytes())
                os.replace(replacement, self.raw)
                return result
            with mock.patch.object(store, '_publish_receipt', side_effect=replace_raw):
                with self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
        self.assertTrue(self.raw.exists())

    def test_derived_drift_after_publication_prevents_unlink_and_resume(self):
        with self.store() as store:
            publish = store._publish_receipt
            def change_chunk(*args):
                publish(*args)
                self.train.write_bytes(b'changed after publication')
            with mock.patch.object(store, '_publish_receipt', side_effect=change_chunk):
                with self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
            self.assertTrue(self.raw.is_file())
            self.assertTrue(self.receipt_path.is_file())
            with self.assertRaises(receipts.ReceiptError):
                store.verify(self.key)
            with self.assertRaises(receipts.ReceiptError):
                store.evict(self.key)

    def test_short_writes_publish_complete_receipt(self):
        write = os.write
        def short_write(fd, data):
            return write(fd, data[:7])
        with self.store() as store, mock.patch.object(receipts.os, 'write', side_effect=short_write):
            result = store.evict(self.key)
            self.assertTrue(result['raw_deleted'])
            self.assertEqual(json.loads(self.receipt_path.read_bytes()), result['receipt'])
            self.assertEqual(list(self.receipt_path.parent.iterdir()), [self.receipt_path])

    def test_interrupted_receipt_write_preserves_raw_without_publishing(self):
        write, calls = os.write, []
        def broken_write(fd, data):
            calls.append(None)
            if len(calls) == 1:
                return write(fd, data[:10])
            raise OSError('injected short write then disk failure')
        with self.store() as store, mock.patch.object(receipts.os, 'write', side_effect=broken_write):
            with self.assertRaises(receipts.ReceiptError):
                store.evict(self.key)
        self.assertTrue(self.raw.is_file())
        self.assertFalse(self.receipt_path.exists())
        self.assertEqual(list(self.receipt_path.parent.iterdir()), [])

    def test_publication_directory_fsync_failure_keeps_raw_and_resumes(self):
        fsync = os.fsync
        def fail_receipt_directory(fd):
            if self.receipt_path.exists():
                info, parent = os.fstat(fd), self.receipt_path.parent.stat()
                if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
                    raise OSError('injected receipt directory fsync failure')
            return fsync(fd)
        with self.store() as store:
            with mock.patch.object(receipts.os, 'fsync', side_effect=fail_receipt_directory):
                with self.assertRaises(receipts.ReceiptError):
                    store.evict(self.key)
            self.assertTrue(self.raw.is_file())
            self.assertTrue(store.verify(self.key)['raw_present'])
            self.assertTrue(store.evict(self.key)['raw_deleted'])

    def test_post_unlink_directory_fsync_failure_reports_actual_deletion(self):
        fsync_directory = receipts._fsync_directory
        def fail_after_unlink(path):
            if path == self.raw.parent and not self.raw.exists():
                raise OSError('injected raw directory fsync failure')
            return fsync_directory(path)
        with self.store() as store:
            with mock.patch.object(receipts, '_fsync_directory', side_effect=fail_after_unlink):
                result = store.evict(self.key)
            self.assertTrue(result['raw_deleted'])
            self.assertEqual(len(result['warnings']), 1)
            self.assertFalse(store.verify(self.key)['raw_present'])
            self.assertFalse(store.evict(self.key)['raw_deleted'])

    def test_wal_database_is_rejected_without_deletion(self):
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
        with self.store() as store, self.assertRaises(receipts.ReceiptError):
            store.evict(self.key)
        self.assertTrue(self.raw.is_file())
        self.assertFalse(self.receipt_path.exists())

    def test_empty_committed_archive_can_be_evicted_with_metadata_preserved(self):
        self.train.unlink(); self.holdout.unlink()
        self.info.update(chunks=[], holdout=[], stats=dict(training_positions=0, holdout_positions=0))
        self.write_commit()
        with self.store() as store:
            self.assertTrue(store.evict(self.key)['raw_deleted'])
            self.assertEqual(store.verify(self.key)['receipt']['derived_files'], [])


if __name__ == '__main__':
    unittest.main()
