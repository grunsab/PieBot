"""Frozen acquisition resumes from verified corpus receipts without redownloads."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from training.nnue import fetch_lc0_bins as fetch
from training.nnue import lc0_corpus
from training.nnue.lc0_archive_receipts import ArchiveReceiptStore
from training.nnue.tests.test_lc0_corpus import archive, good_record


class FetchEvictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw_root = self.root / 'raw'
        self.raw_root.mkdir()
        self.raw = self.raw_root / 'source.tar'
        self.entry = archive(self.raw, [('training.1.gz', '2026-08-01T00:00:00', good_record())])
        self.raw_bytes = self.raw.read_bytes()
        self.manifest = self.raw_root / 'manifest.json'
        self.snapshot = dict(schema='piebot-lc0-raw-v1', complete=True, failures=[],
                             since='2026-07-07T00:00:00+00:00', until='2026-09-07T00:00:00+00:00',
                             files=[self.entry])
        # Intentionally differs from write_json_atomic's formatting.
        self.manifest.write_text(json.dumps(self.snapshot, sort_keys=True, indent=1) + '\n\n')
        self.corpus = self.root / 'corpus'
        self.key = hashlib.sha256(json.dumps([self.entry['url'], self.entry['sha256']],
                                            separators=(',', ':')).encode()).hexdigest()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return lc0_corpus.prepare_corpus(self.manifest, self.corpus, since='2026-07-07',
                until='2026-09-07T00:00:00+00:00', validation_fraction=0,
                min_free_bytes=0, workers=1, chunk_positions=2)

    def evict(self):
        self.prepare()
        with ArchiveReceiptStore(self.manifest, self.corpus, raw_root=self.raw_root) as store:
            result = store.evict(self.key)
        self.receipt = Path(result['receipt_path'])
        return result

    def fetch(self):
        return fetch.download_snapshot(self.manifest, min_free_bytes=0,
                                       eviction_corpus=self.corpus, raw_root=self.raw_root)

    def assert_refused_unchanged(self):
        before = self.manifest.read_bytes()
        with mock.patch.object(fetch, 'download_curl') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.fetch(), 1)
            download.assert_not_called()
            write.assert_not_called()
        self.assertEqual(before, self.manifest.read_bytes())

    def test_default_mode_still_redownloads_missing_raw(self):
        self.evict()
        def download(job, **_kwargs):
            job.dest.write_bytes(self.raw_bytes)
            return dict(sha256=self.entry['sha256'])
        with mock.patch.object(fetch, 'download_curl', side_effect=download) as called, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0), 0)
        called.assert_called_once()
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_opt_in_initial_download_before_corpus_initialization_is_ordinary(self):
        self.raw.unlink()
        self.corpus.mkdir()
        def download(job, **_kwargs):
            job.dest.write_bytes(self.raw_bytes)
            return dict(sha256=self.entry['sha256'])
        with mock.patch.object(fetch, 'download_curl', side_effect=download) as called, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.fetch(), 0)
        called.assert_called_once()
        self.assertTrue(self.raw.is_file())
        self.assertFalse((self.corpus / 'prepare.lock').exists())

    def test_evicted_archive_is_verified_without_manifest_rewrite_or_download(self):
        self.evict()
        before = self.manifest.read_bytes()
        with mock.patch.object(fetch, 'download_curl') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write:
            self.assertEqual(self.fetch(), 0)
            download.assert_not_called()
            write.assert_not_called()
        self.assertFalse(self.raw.exists())
        self.assertEqual(before, self.manifest.read_bytes())

    def test_initialized_corpus_missing_raw_without_receipt_fails_closed(self):
        self.prepare()
        self.raw.unlink()
        self.assert_refused_unchanged()

    def test_missing_dedup_database_cannot_trigger_redownload(self):
        self.evict()
        (self.corpus / 'progress.sqlite3').unlink()
        self.assert_refused_unchanged()

    def test_corrupt_receipt_is_checked_even_when_raw_is_present(self):
        self.evict()
        self.receipt.write_bytes(b'corrupt proof')
        for present in (False, True):
            with self.subTest(present=present):
                if present:
                    self.raw.write_bytes(self.raw_bytes)
                self.assert_refused_unchanged()
                self.assertEqual(self.raw.exists(), present)

    def test_corrupt_derived_chunk_refuses_even_if_raw_is_present(self):
        self.evict()
        self.raw.write_bytes(self.raw_bytes)
        chunk = next((self.corpus / 'archives').glob('*/train_*.jsonl.gz'))
        chunk.write_bytes(b'corrupt derived bytes')
        self.assert_refused_unchanged()
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_present_raw_checksum_is_checked_and_failure_preserves_manifest(self):
        self.prepare()
        data = bytearray(self.raw_bytes)
        data[-1] ^= 1
        self.raw.write_bytes(data)
        self.assert_refused_unchanged()
        self.assertEqual(self.raw.read_bytes(), data)

    def test_verified_present_archive_and_receipt_are_never_deleted_by_fetch(self):
        self.evict()
        self.raw.write_bytes(self.raw_bytes)
        before = self.manifest.read_bytes()
        with mock.patch.object(fetch, 'download_curl') as download:
            self.assertEqual(self.fetch(), 0)
            download.assert_not_called()
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_prepared_present_archive_without_receipt_preserves_frozen_bytes(self):
        self.prepare()
        before = self.manifest.read_bytes()
        with mock.patch.object(fetch, 'download_curl') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write:
            self.assertEqual(self.fetch(), 0)
            download.assert_not_called()
            write.assert_not_called()
        self.assertTrue(self.raw.is_file())
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_present_symlink_without_receipt_fails_closed(self):
        self.prepare()
        outside = self.root / 'outside.tar'
        self.raw.rename(outside)
        self.raw.symlink_to(outside)
        self.assert_refused_unchanged()
        self.assertEqual(outside.read_bytes(), self.raw_bytes)

    def test_aliased_manifest_is_rejected_before_any_receipt_exists(self):
        self.prepare()
        outside = self.root / 'provenance.json'
        self.manifest.rename(outside)
        self.manifest.symlink_to(outside)
        self.assert_refused_unchanged()

    def test_main_aliased_manifest_is_rejected_before_reading_or_rewriting(self):
        self.prepare()
        outside = self.root / 'provenance.json'
        self.manifest.rename(outside)
        self.manifest.symlink_to(outside)
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--since', '2026-07-07',
                '--until', '2026-09-07T00:00:00+00:00', '--eviction-corpus', str(self.corpus)]
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'discover_snapshot') as discover, \
             mock.patch.object(fetch, 'download_snapshot') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fetch.main(), 1)
            discover.assert_not_called()
            download.assert_not_called()
            write.assert_not_called()

    def test_fetch_respects_exclusive_preparation_lock(self):
        self.evict()
        with ArchiveReceiptStore(self.manifest, self.corpus, raw_root=self.raw_root):
            self.assert_refused_unchanged()

    def test_opt_in_requires_explicit_raw_root(self):
        with self.assertRaisesRegex(ValueError, 'raw.root'):
            fetch.download_snapshot(self.manifest, eviction_corpus=self.corpus)

    def test_main_receipt_resume_preserves_original_manifest_bytes(self):
        self.evict()
        before = self.manifest.read_bytes()
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--since', '2026-07-07',
                '--until', '2026-09-07T00:00:00+00:00', '--eviction-corpus', str(self.corpus),
                '--min-free-gib', '0']
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'download_curl') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write:
            self.assertEqual(fetch.main(), 0)
            download.assert_not_called()
            write.assert_not_called()
        self.assertEqual(before, self.manifest.read_bytes())

    def test_eviction_flag_is_snapshot_only(self):
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--eviction-corpus', str(self.corpus)]
        with mock.patch.object(fetch.sys, 'argv', argv), \
             self.assertRaisesRegex(ValueError, 'snapshot'):
            fetch.main()

    def test_main_missing_manifest_with_corpus_evidence_cannot_discover_or_write(self):
        self.evict()
        self.manifest.unlink()
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--since', '2026-07-07',
                '--until', '2026-09-07T00:00:00+00:00', '--eviction-corpus', str(self.corpus)]
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'discover_snapshot') as discover, \
             mock.patch.object(fetch, 'download_curl') as download, \
             mock.patch.object(fetch, 'write_json_atomic') as write, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fetch.main(), 1)
            discover.assert_not_called()
            download.assert_not_called()
            write.assert_not_called()
        self.assertFalse(self.manifest.exists())

    def test_download_capacity_counts_existing_usage_and_keeps_reserve(self):
        before = self.manifest.read_bytes()
        reserve, already_used = 100, 100_000_000_000
        def download(job, **_kwargs):
            job.dest.write_bytes(self.raw_bytes)
            return dict(sha256=self.entry['sha256'])
        for extra, expected in ((-1, 1), (0, 0)):
            with self.subTest(extra=extra):
                self.manifest.write_bytes(before)
                self.raw.unlink(missing_ok=True)
                capacity = already_used + len(self.raw_bytes) + reserve + extra
                with mock.patch.object(fetch.shutil, 'disk_usage',
                         return_value=SimpleNamespace(total=600_000_000_000, free=500_000_000_000)), \
                     mock.patch.object(fetch, 'download_curl', side_effect=download) as called, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=reserve,
                                                           capacity_bytes=capacity), expected)
                    self.assertEqual(called.call_count, 1 - expected)

    def test_invalid_api_capacity_fails_before_manifest_write_even_if_cached(self):
        before = self.manifest.read_bytes()
        for capacity in (-1, 0, True, 1.5, '478000000000'):
            with self.subTest(capacity=capacity), \
                 mock.patch.object(fetch, 'write_json_atomic') as write, \
                 self.assertRaises(ValueError):
                fetch.download_snapshot(self.manifest, capacity_bytes=capacity)
            write.assert_not_called()
        self.assertEqual(before, self.manifest.read_bytes())

    def test_invalid_cli_capacity_fails_before_any_manifest_write(self):
        for capacity in ('-1', 'nan', 'inf', '0.0000000001'):
            argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--since', '2026-07-07',
                    '--until', '2026-09-07T00:00:00+00:00', '--disk-capacity-gb', capacity]
            with self.subTest(capacity=capacity), mock.patch.object(fetch.sys, 'argv', argv), \
                 mock.patch.object(fetch, 'write_json_atomic') as write, \
                 self.assertRaises(ValueError):
                fetch.main()
            write.assert_not_called()

    def test_main_wires_decimal_capacity_and_explicit_raw_root(self):
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--since', '2026-07-07',
                '--until', '2026-09-07T00:00:00+00:00', '--disk-capacity-gb', '478',
                '--eviction-corpus', str(self.corpus)]
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'download_snapshot', return_value=0) as download:
            self.assertEqual(fetch.main(), 0)
        self.assertEqual(download.call_args.kwargs['capacity_bytes'], 478_000_000_000)
        self.assertEqual(download.call_args.kwargs['raw_root'], self.raw_root)
        self.assertEqual(download.call_args.kwargs['eviction_corpus'], self.corpus)

    def test_positive_capacity_flag_cannot_be_silently_ignored_in_legacy_mode(self):
        argv = ['fetch_lc0_bins', '--out', str(self.raw_root), '--disk-capacity-gb', '478']
        with mock.patch.object(fetch.sys, 'argv', argv), \
             self.assertRaisesRegex(ValueError, 'snapshot'):
            fetch.main()


if __name__ == '__main__':
    unittest.main()
