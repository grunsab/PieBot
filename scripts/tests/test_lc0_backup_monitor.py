"""Backup cadence and transport failures must never publish false progress."""
import hashlib
import importlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeProcess:
    def __init__(self, stdout=b'', stderr=b'', returncode=0):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class BackupMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def module(self):
        return importlib.import_module('scripts.lc0_backup_monitor')

    def config(self, **changes):
        values = dict(pool=self.root / 'pool', state_path=self.root / 'monitor.json',
                      helper_sha256='a' * 64)
        values.update(changes)
        return self.module().BackupConfig(**values)

    def ready(self, key='b', status='running', chunks=1):
        return dict(ready=chunks > 0, reason='ready', status=status, completed_chunks=chunks,
                    state_sha256='c' * 64, model_identity_key=key * 64,
                    hashes_verified=False, deadline_at=1000.)

    def receipt(self, key='b', status='running'):
        self.root.joinpath('pool/blobs').mkdir(parents=True, exist_ok=True)
        payload = dict(schema='piebot-lc0-model-snapshot-v1', capabilities={'strict_campaign_resume': False},
                       **self.ready(key, status))
        raw = json.dumps(payload).encode()
        digest = hashlib.sha256(raw).hexdigest()
        manifest = self.root / 'pool/blobs' / digest
        manifest.write_bytes(raw)
        return dict(snapshot_id=digest, manifest_path=str(manifest),
                    receipt_path=str(self.root / 'pool/snapshots' / (digest + '.json')),
                    catalog_path=str(self.root / 'pool/catalog.json'))

    def test_missing_state_and_zero_chunks_wait_without_touching_journal(self):
        m = self.module()
        for result in (dict(ready=False, reason='state-missing'), self.ready(chunks=0)):
            transfer = mock.Mock()
            self.assertEqual(m.run_once(self.config(), probe_fn=lambda _: result,
                                       transfer_fn=transfer, now=lambda: 10.), 'waiting')
            transfer.assert_not_called()
            self.assertFalse(self.config().state_path.exists())

    def test_corpus_root_reaches_the_capture_command_and_the_journal_identity_only_when_set(self):
        m = self.module()
        plain = self.config()
        self.assertNotIn('--corpus-root', shlex.split(m.ssh_command(plain, probe=False)[-1]))
        self.assertNotIn('corpus_root', m._identity(plain))
        reused = self.config(corpus_root='/workspace/older_lineage')
        capture = shlex.split(m.ssh_command(reused, probe=False)[-1])
        self.assertEqual(capture[capture.index('--corpus-root') + 1], '/workspace/older_lineage')
        self.assertNotIn('--corpus-root', shlex.split(m.ssh_command(reused, probe=True)[-1]))
        self.assertEqual(m._identity(reused)['corpus_root'], '/workspace/older_lineage')
        with self.assertRaises(m.BackupError):
            m.ssh_command(self.config(corpus_root='relative/path'), probe=False)

    def test_daily_changed_key_and_final_capture_use_actual_captured_identity(self):
        m = self.module()
        config = self.config()
        transfer = mock.Mock(return_value=self.receipt('d'))
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('b'),
                                   transfer_fn=transfer, now=lambda: 10.), 'captured')
        state = json.loads(config.state_path.read_text())
        self.assertEqual(state['last_model_identity_key'], 'd' * 64)
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('e'),
                                   transfer_fn=transfer, now=lambda: 11.), 'not-due')
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('d'),
                                   transfer_fn=transfer, now=lambda: 90000.), 'not-due')
        transfer.return_value = self.receipt('e')
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('e'),
                                   transfer_fn=transfer, now=lambda: 90000.), 'captured')
        transfer.return_value = self.receipt('e', 'complete')
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('e', 'complete'),
                                   transfer_fn=transfer, now=lambda: 90001.), 'final-captured')
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('e', 'complete'),
                                   transfer_fn=transfer, now=lambda: 90002.), 'complete')
        self.assertEqual(transfer.call_count, 3)

    def test_expired_clock_alone_does_not_mark_a_running_capture_final(self):
        m = self.module()
        config = self.config()
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready(),
                                   transfer_fn=lambda _: self.receipt(), now=lambda: 2000.), 'captured')
        self.assertFalse(json.loads(config.state_path.read_text())['final_snapshot'])

    def test_captured_status_controls_finality_when_probe_changes(self):
        m = self.module()
        config = self.config()
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready(status='complete'),
                                   transfer_fn=lambda _: self.receipt(), now=lambda: 10.), 'captured')
        self.assertFalse(json.loads(config.state_path.read_text())['final_snapshot'])
        self.assertEqual(m.run_once(config, probe_fn=lambda _: self.ready('d'),
                                   transfer_fn=lambda _: self.receipt('d', 'complete'),
                                   now=lambda: 90000.), 'final-captured')

    def test_real_verified_wrapper_executes_dataclass_helper_and_rejects_changed_bytes(self):
        m = self.module()
        helper = self.root / "helper '$(touch SHOULD_NOT_EXIST).py"
        helper.write_text('from __future__ import annotations\n'
                          'from dataclasses import dataclass\n'
                          'import json, sys\n'
                          '@dataclass\nclass Example:\n    value: str\n'
                          'print(json.dumps({"argv": sys.argv[1:], "value": Example("ok").value}))\n')
        digest = hashlib.sha256(helper.read_bytes()).hexdigest()
        remote = shlex.split(m.ssh_command(self.config(helper=str(helper), helper_sha256=digest), probe=True)[-1])
        remote[0] = sys.executable
        result = subprocess.run(remote, capture_output=True, check=True, timeout=3)
        output = json.loads(result.stdout)
        self.assertEqual(output['value'], 'ok')
        self.assertEqual(output['argv'], ['--campaign-root', '/workspace/piebot_lc0_20260907', '--probe'])
        helper.write_text('raise AssertionError("changed code must never execute")\n')
        failed = subprocess.run(remote, capture_output=True, timeout=3)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn(b'SHA-256 mismatch', failed.stderr)
        self.assertNotIn(b'AssertionError', failed.stderr)
        self.assertFalse((self.root / 'SHOULD_NOT_EXIST').exists())

    def test_singleton_lock_excludes_a_second_monitor_before_probe(self):
        m = self.module()
        config = self.config()
        with m._monitor_lock(config), mock.patch.object(m, 'run_once') as once:
            with self.assertRaisesRegex(m.BackupError, 'lock'):
                m.run(config, once=True)
            once.assert_not_called()

    def test_transfer_failure_preserves_previous_journal_bytes(self):
        m = self.module()
        config = self.config()
        m.run_once(config, probe_fn=lambda _: self.ready(), transfer_fn=lambda _: self.receipt(), now=lambda: 1.)
        original = config.state_path.read_bytes()
        with self.assertRaises(m.BackupError):
            m.run_once(config, probe_fn=lambda _: self.ready('d'),
                       transfer_fn=mock.Mock(side_effect=m.BackupError('partial stream')),
                       now=lambda: 90000.)
        self.assertEqual(config.state_path.read_bytes(), original)

    def test_ssh_argv_quotes_remote_arguments_and_verifies_exact_helper_bytes(self):
        m = self.module()
        helper = "/workspace/helper dir/x'$(touch bad).py"
        config = self.config(helper=helper, campaign_root="/workspace/campaign ' quoted")
        command = m.ssh_command(config, probe=True)
        self.assertEqual(command[:2], ['ssh', '-T'])
        self.assertIn('root@104.8.120.185', command)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[:2], ['python3', '-c'])
        self.assertIn('sha256', remote[2])
        self.assertIn('compile(', remote[2])
        self.assertEqual(remote[3:5], [helper, 'a' * 64])
        self.assertIn("/workspace/campaign ' quoted", remote)
        self.assertIn('--probe', remote)

    def test_probe_nonzero_malformed_or_oversized_stdout_refuses(self):
        m = self.module()
        cases = [FakeProcess(b'{}', b'helper mismatch', 2), FakeProcess(b'not-json'),
                 FakeProcess(b'x' * (m.MAX_PROBE_BYTES + 1))]
        for process in cases:
            with self.subTest(process=process), self.assertRaises(m.BackupError):
                m.probe(self.config(), popen=lambda *a, **kw: process)

    def test_probe_waits_for_valid_exit_and_drains_bounded_stderr(self):
        m = self.module()
        process = FakeProcess(json.dumps(self.ready()).encode(), b'x' * 200000)
        calls = []
        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process
        self.assertEqual(m.probe(self.config(), popen=popen)['completed_chunks'], 1)
        self.assertNotIn('shell', calls[0][1])
        self.assertTrue(process.stderr.closed)

    def test_receiver_requires_successful_ssh_exit_before_returning_success(self):
        m = self.module()
        process = FakeProcess(b'valid archive', b'exporter final check failed', 1)
        def receiver(stream, pool, *, confirm_source_success, **kwargs):
            self.assertEqual(stream.read(), b'valid archive')
            if confirm_source_success() is not True:
                raise RuntimeError('sender failed')
            return self.receipt()
        with self.assertRaises(m.BackupError):
            m.transfer(self.config(), receiver=receiver, popen=lambda *a, **kw: process)

    def test_receiver_failure_kills_process_and_does_not_confirm_success(self):
        m = self.module()
        process = FakeProcess(b'partial archive')
        with self.assertRaises(m.BackupError):
            m.transfer(self.config(), receiver=mock.Mock(side_effect=RuntimeError('truncated TAR')),
                       popen=lambda *a, **kw: process)
        self.assertTrue(process.killed)
        self.assertTrue(process.stdout.closed)

    def test_transfer_rejects_receiver_that_skips_source_confirmation(self):
        m = self.module()
        process = FakeProcess(b'archive')
        with self.assertRaises(m.BackupError):
            m.transfer(self.config(), receiver=lambda *a, **kw: self.receipt(),
                       popen=lambda *a, **kw: process)

    def test_standalone_monitor_uses_staged_sibling_even_if_repo_package_is_importable(self):
        m = self.module()
        calls = []
        def receiver(name):
            def receive(*_args, confirm_source_success, **_kwargs):
                calls.append(name)
                self.assertIs(confirm_source_success(), True)
                return self.receipt()
            return types.SimpleNamespace(receive_snapshot=receive)
        modules = {'scripts.lc0_snapshot_store': receiver('repository-package'),
                   'lc0_snapshot_store': receiver('staged-sibling')}
        with mock.patch.dict(sys.modules, modules):
            for package, expected in [('scripts', 'repository-package'), (None, 'staged-sibling')]:
                with mock.patch.object(m, '__package__', package):
                    m.transfer(self.config(), popen=lambda *a, **kw: FakeProcess())
                self.assertEqual(calls[-1], expected)

    def test_transfer_timeout_kills_and_reaps_blocked_transport(self):
        m = self.module()
        processes = []
        def local_process(*_args, **_kwargs):
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)
            return process
        def blocked_receiver(stream, *_args, **_kwargs):
            stream.read()
            raise RuntimeError('stream ended')
        started = time.monotonic()
        with self.assertRaisesRegex(m.BackupError, 'timed out'):
            m.transfer(self.config(transfer_timeout=.05), receiver=blocked_receiver, popen=local_process)
        self.assertLess(time.monotonic() - started, 2.)
        self.assertIsNotNone(processes[0].poll())

    def test_timeout_kills_local_descendants_holding_transport_pipes(self):
        m = self.module()
        processes = []
        def parent_with_child(*_args, **_kwargs):
            code = ('import subprocess, sys, time; '
                    'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"]); '
                    'time.sleep(3)')
            process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            processes.append(process)
            return process
        def blocked_receiver(stream, *_args, **_kwargs):
            stream.read()
            raise RuntimeError('stream ended')
        started = time.monotonic()
        with self.assertRaisesRegex(m.BackupError, 'timed out'):
            m.transfer(self.config(transfer_timeout=.15), receiver=blocked_receiver, popen=parent_with_child)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsNotNone(processes[0].poll())

    def test_stop_event_cancels_an_active_transfer_and_reaps_ssh(self):
        m = self.module()
        stop = threading.Event()
        timer = threading.Timer(.05, stop.set)
        processes = []
        def local_process(*_args, **_kwargs):
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            processes.append(process)
            return process
        def blocked_receiver(stream, *_args, **_kwargs):
            stream.read()
            raise RuntimeError('stream ended')
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaisesRegex(m.BackupError, 'cancelled'):
            m.transfer(self.config(), receiver=blocked_receiver, popen=local_process, stop=stop)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsNotNone(processes[0].poll())

    def test_successful_leader_exit_cleans_descendant_holding_only_stderr(self):
        m = self.module()
        payload = json.dumps(self.ready())
        def parent_with_stderr_child(*_args, **_kwargs):
            code = ('import subprocess, sys; '
                    'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"], '
                    'stdout=subprocess.DEVNULL); '
                    f'print({payload!r})')
            return subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True)
        started = time.monotonic()
        self.assertTrue(m.probe(self.config(probe_timeout=.2), popen=parent_with_stderr_child)['ready'])
        self.assertLess(time.monotonic() - started, 1.)

    def test_valid_captured_header_above_one_mib_uses_protocol_limit(self):
        m = self.module()
        receipt = self.receipt()
        original = Path(receipt['manifest_path'])
        header = json.loads(original.read_text())
        header['protocol_sized_metadata'] = 'x' * (2 * 1024 * 1024)
        raw = json.dumps(header).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = original.with_name(digest)
        path.write_bytes(raw)
        receipt.update(snapshot_id=digest, manifest_path=str(path))
        self.assertEqual(m._captured_summary(receipt, self.config())['model_identity_key'], 'b' * 64)

    def test_continuous_failures_retry_with_bounded_interruptible_sleep(self):
        m = self.module()
        stop = threading.Event()
        calls = []
        def wait(seconds):
            calls.append(seconds)
            stop.set()
            return True
        with mock.patch.object(m, 'run_once', side_effect=m.BackupError('offline')):
            self.assertEqual(m.run(self.config(), stop=stop, wait=wait), 0)
        self.assertEqual(calls, [60.])
        self.assertFalse(self.config().state_path.exists())

    def test_once_missing_state_returns_and_once_failure_reports_error(self):
        m = self.module()
        with mock.patch.object(m, 'run_once', return_value='waiting') as once:
            self.assertEqual(m.run(self.config(), once=True), 0)
            self.assertEqual(once.call_count, 1)
        with mock.patch.object(m, 'run_once', side_effect=m.BackupError('offline')):
            self.assertEqual(m.run(self.config(), once=True), 1)


if __name__ == '__main__':
    unittest.main()
