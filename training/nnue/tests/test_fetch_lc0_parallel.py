"""Parallel snapshot downloads preserve progress and share disk admission."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from training.nnue import fetch_lc0_bins as fetch


class ParallelSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.manifest = self.root / 'manifest.json'

    def inventory(self, count=3, size=8):
        self.rows = [dict(name=f'{i}.tar', dest=str(self.root / f'{i}.tar'),
                          url=f'https://example.invalid/{i}.tar', size=size, status='queued')
                     for i in range(count)]
        fetch.write_json_atomic(self.manifest, {'files': self.rows, 'complete': False, 'failures': []})

    def publish(self, job):
        row = next(x for x in self.rows if x['url'] == job.url)
        payload = bytes([int(job.dest.stem)]) * row['size']
        job.dest.write_bytes(payload)
        return {'sha256': hashlib.sha256(payload).hexdigest()}

    def test_out_of_order_completion_is_persisted_before_slow_first_download(self):
        self.inventory(5)
        first_started, release_first, second_saved = (threading.Event() for _ in range(3))
        active = peak = 0
        lock = threading.Lock()
        errors = []
        owner = []
        result = []
        original = fetch.write_json_atomic

        def download(job, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if job.dest.stem == '0':
                    first_started.set()
                    if not release_first.wait(5):
                        raise RuntimeError('test first-download release timed out')
                else:
                    if not first_started.wait(5):
                        raise RuntimeError('test first download never started')
                return self.publish(job)
            finally:
                with lock:
                    active -= 1

        def save(path, value):
            self.assertEqual(threading.get_ident(), owner[0])
            original(path, value)
            if value['files'][1]['status'] == 'downloaded':
                second_saved.set()

        def run():
            owner.append(threading.get_ident())
            try:
                result.append(fetch.download_snapshot(self.manifest, min_free_bytes=0, concurrency=2))
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(fetch, 'download_curl', side_effect=download), \
             mock.patch.object(fetch, 'write_json_atomic', side_effect=save):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(second_saved.wait(5), str(errors))
                saved = json.loads(self.manifest.read_text())
                self.assertEqual(saved['files'][0]['status'], 'queued')
                self.assertFalse(saved['complete'])
            finally:
                release_first.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, [0])
        self.assertEqual(peak, 2)
        saved = json.loads(self.manifest.read_text())
        self.assertTrue(saved['complete'])
        self.assertEqual([r['name'] for r in saved['files']], [r['name'] for r in self.rows])
        self.assertTrue(all(r['sha256'] == fetch.sha256_file(Path(r['dest'])) for r in saved['files']))

    def test_failure_drains_existing_jobs_and_resume_fetches_only_unfinished(self):
        self.inventory()
        second_started, failure_saved = threading.Event(), threading.Event()
        started = []
        original = fetch.write_json_atomic

        def download(job, **kwargs):
            started.append(job.dest.stem)
            if job.dest.stem == '0':
                self.assertTrue(second_started.wait(5))
                raise OSError('network lost')
            self.assertEqual(job.dest.stem, '1')
            second_started.set()
            self.assertTrue(failure_saved.wait(5))
            return self.publish(job)

        def save(path, value):
            original(path, value)
            if value['failures']:
                failure_saved.set()

        with mock.patch.object(fetch, 'download_curl', side_effect=download), \
             mock.patch.object(fetch, 'write_json_atomic', side_effect=save):
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0, concurrency=2), 1)
        self.assertEqual(set(started), {'0', '1'})
        saved = json.loads(self.manifest.read_text())
        self.assertFalse(saved['complete'])
        self.assertIn('error', saved['files'][0]['status'])
        self.assertEqual(saved['files'][1]['status'], 'downloaded')
        self.assertEqual(saved['files'][2]['status'], 'queued')
        with mock.patch.object(fetch, 'download_curl', side_effect=lambda job, **_: self.publish(job)) as dl:
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0, concurrency=2), 0)
        self.assertEqual({c.args[0].dest.stem for c in dl.call_args_list}, {'0', '2'})

    def test_inflight_sizes_are_reserved_together_and_low_space_waits_for_completion(self):
        self.inventory(3, size=60)
        pending = []
        case = self

        class Pool:
            def __init__(self, max_workers):
                case.assertEqual(max_workers, 2)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def submit(self, fn, *args, **kwargs):
                future = cf.Future()
                pending.append((future, fn, args, kwargs))
                return future

        def wait(futures, **kwargs):
            # 100 available bytes cannot admit two 60-byte downloads together.
            self.assertEqual(len(futures), 1)
            future, fn, args, options = pending.pop(0)
            future.set_result(fn(*args, **options))
            return {future}, set()

        with mock.patch.object(fetch.cf, 'ThreadPoolExecutor', Pool), \
             mock.patch.object(fetch.cf, 'wait', side_effect=wait), \
             mock.patch.object(fetch, 'available_bytes', return_value=100) as available, \
             mock.patch.object(fetch, 'download_curl', side_effect=lambda job, **_: self.publish(job)):
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0, concurrency=2,
                                                    capacity_bytes=478_000_000_000), 0)
        self.assertTrue(all(c.kwargs['capacity_bytes'] == 478_000_000_000 for c in available.call_args_list))

    def test_insufficient_space_stops_without_downloading(self):
        self.inventory(2, size=60)
        with mock.patch.object(fetch, 'available_bytes', return_value=59), \
             mock.patch.object(fetch, 'download_curl') as dl:
            self.assertEqual(fetch.download_snapshot(self.manifest, min_free_bytes=0, concurrency=4), 1)
        dl.assert_not_called()
        self.assertFalse(json.loads(self.manifest.read_text())['complete'])

    def test_invalid_concurrency_and_destination_collisions_fail_before_writing(self):
        self.inventory()
        for value in (0, -1, 17, 1.5, True):
            with self.subTest(concurrency=value), mock.patch.object(fetch, 'write_json_atomic') as write:
                with self.assertRaisesRegex(ValueError, 'concurrency'):
                    fetch.download_snapshot(self.manifest, concurrency=value)
                write.assert_not_called()
        for collision in ('0.tar', '0.tar.part', 'manifest.json', 'manifest.json.part'):
            self.inventory()
            payload = json.loads(self.manifest.read_text())
            payload['files'][1]['dest'] = str(self.root / collision)
            fetch.write_json_atomic(self.manifest, payload)
            with self.subTest(collision=collision), mock.patch.object(fetch, 'write_json_atomic') as write:
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    fetch.download_snapshot(self.manifest, concurrency=2)
                write.assert_not_called()

    def test_snapshot_cli_forwards_existing_concurrency_option(self):
        self.inventory(1)
        payload = json.loads(self.manifest.read_text())
        payload.update(schema='piebot-lc0-raw-v1', since='2026-07-07T00:00:00+00:00',
                       until='2026-09-07T19:39:28+00:00')
        fetch.write_json_atomic(self.manifest, payload)
        argv = ['fetch', '--out', str(self.root), '--manifest', str(self.manifest),
                '--since', '2026-07-07', '--until', '2026-09-07', '--concurrency', '3']
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'download_snapshot', return_value=0) as download:
            self.assertEqual(fetch.main(), 0)
        self.assertEqual(download.call_args.kwargs['concurrency'], 3)

    def test_cli_rejects_output_collision_before_rewriting_manifest(self):
        self.inventory(1)
        payload = json.loads(self.manifest.read_text())
        payload.update(schema='piebot-lc0-raw-v1', since='2026-07-07T00:00:00+00:00',
                       until='2026-09-07T19:39:28+00:00')
        payload['files'][0]['dest'] = str(self.manifest) + '.part'
        fetch.write_json_atomic(self.manifest, payload)
        argv = ['fetch', '--out', str(self.root), '--manifest', str(self.manifest),
                '--since', '2026-07-07', '--until', '2026-09-07', '--concurrency', '4']
        with mock.patch.object(fetch.sys, 'argv', argv), \
             mock.patch.object(fetch, 'write_json_atomic') as write:
            with self.assertRaisesRegex(ValueError, 'overlap'):
                fetch.main()
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
