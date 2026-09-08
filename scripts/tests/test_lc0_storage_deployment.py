"""Opt-in raw eviction and a decimal capacity ceiling reach every LC0 stage."""
import configparser
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from training.nnue import lc0_corpus, lc0_deploy

ROOT = Path(__file__).resolve().parents[2]


class Lc0StorageDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root / 'lc0'
        self.args = ['--repo', str(self.root), '--out-root', str(self.output),
                     '--selfplay-root', str(self.root / 'selfplay')]

    def deploy(self, *args, free=500_000_000_000, total=600_000_000_000):
        torch = mock.Mock()
        torch.cuda.is_available.return_value = True
        torch.ones.return_value.sum.return_value.item.return_value = 1
        locks = SimpleNamespace(_single_instance_lock=lambda _path: contextlib.nullcontext())
        prepared = self.output / 'data/corpus/corpus_manifest.json'
        with mock.patch.object(lc0_deploy, 'validate_bootstrap'), \
             mock.patch.object(lc0_deploy.subprocess, 'check_output', return_value='a' * 40), \
             mock.patch.object(lc0_deploy.subprocess, 'run') as run, \
             mock.patch.object(lc0_deploy.os, 'access', return_value=True), \
             mock.patch.object(lc0_deploy.shutil, 'which', return_value='/usr/bin/curl'), \
             mock.patch.object(lc0_deploy.shutil, 'disk_usage',
                               return_value=SimpleNamespace(total=total, free=free)), \
             mock.patch.dict(sys.modules, {'torch': torch, 'training.nnue.autopilot': locks}), \
             mock.patch.object(lc0_corpus, 'prepare_corpus', return_value=prepared) as prepare, \
             mock.patch.object(lc0_deploy, 'wait_for_training') as train, \
             contextlib.redirect_stdout(io.StringIO()):
            result = lc0_deploy.main(self.args + list(args))
        if '--preflight-only' in args:
            return result, None, prepare.call_args, train.call_args
        fetch = next(call.args[0] for call in run.call_args_list
                     if 'training.nnue.fetch_lc0_bins' in call.args[0])
        return result, fetch, prepare.call_args, train.call_args.args[0]

    def test_default_retains_raw_and_forwards_disabled_capacity_to_every_stage(self):
        result, fetch, prepare, train = self.deploy()
        self.assertEqual(result, 0)
        self.assertNotIn('--eviction-corpus', fetch)
        self.assertFalse(prepare.kwargs.get('evict_raw', False))
        self.assertIsNone(prepare.kwargs.get('capacity_bytes'))
        self.assertEqual(float(fetch[fetch.index('--disk-capacity-gb') + 1]), 0)
        self.assertEqual(float(train[train.index('--disk-capacity-gb') + 1]), 0)
        self.assertEqual(float(train[train.index('--hours') + 1]), 720)

    def test_opt_in_forwards_receipt_root_capacity_and_explicit_raw_root(self):
        result, fetch, prepare, train = self.deploy('--evict-raw', '--disk-capacity-gb', '478')
        self.assertEqual(result, 0)
        self.assertEqual(fetch[fetch.index('--eviction-corpus') + 1], str(self.output / 'data/corpus'))
        self.assertEqual(float(fetch[fetch.index('--disk-capacity-gb') + 1]), 478)
        self.assertEqual(float(train[train.index('--disk-capacity-gb') + 1]), 478)
        self.assertTrue(prepare.kwargs['evict_raw'])
        self.assertEqual(prepare.kwargs['raw_root'], self.output / 'data/raw')
        self.assertEqual(prepare.kwargs['capacity_bytes'], 478_000_000_000)
        self.assertEqual((self.output / 'source_git_commit').read_text(), 'a' * 40 + '\n')

    def test_explicit_no_eviction_overrides_enabled_flag_without_disabling_capacity(self):
        _, fetch, prepare, train = self.deploy('--evict-raw', '--no-evict-raw', '--disk-capacity-gb', '478')
        self.assertNotIn('--eviction-corpus', fetch)
        self.assertFalse(prepare.kwargs.get('evict_raw', False))
        self.assertEqual(prepare.kwargs['capacity_bytes'], 478_000_000_000)
        self.assertEqual(float(train[train.index('--disk-capacity-gb') + 1]), 478)

    def test_invalid_capacity_is_rejected_before_bootstrap_or_pin(self):
        for value in ('-1', 'nan', 'inf', '0.0000000001'):
            with self.subTest(value=value), \
                 mock.patch.object(lc0_deploy, 'validate_bootstrap') as bootstrap, \
                 mock.patch.object(lc0_deploy, 'pin_source') as pin, \
                 self.assertRaises(ValueError):
                lc0_deploy.main(self.args + ['--disk-capacity-gb', value])
            bootstrap.assert_not_called()
            pin.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_capacity_preflight_uses_rented_ceiling_before_pinning(self):
        # Host reports500GB free; only8GB remain below478GB after470GB used.
        with self.assertRaisesRegex(ValueError, 'disk reserve'):
            self.deploy('--disk-capacity-gb', '478', free=500_000_000_000, total=970_000_000_000)
        self.assertFalse((self.output / 'source_git_commit').exists())

    def test_default_capacity_keeps_filesystem_free_space_behavior(self):
        result, _, prepare, _ = self.deploy(free=500_000_000_000, total=970_000_000_000)
        self.assertEqual(result, 0)
        self.assertIsNone(prepare.kwargs.get('capacity_bytes'))

    def test_exhausted_ceiling_with_zero_reserve_refuses_preflight_before_output(self):
        with self.assertRaisesRegex(ValueError, 'disk reserve'):
            self.deploy('--disk-capacity-gb', '478', '--min-free-gib', '0',
                        '--preflight-only', free=122_000_000_000, total=600_000_000_000)
        self.assertFalse(self.output.exists())


class Lc0StorageLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.capture = self.root / 'argv.txt'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        python = self.bin / 'python3'
        python.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "$LC0_TEST_CAPTURE"\n')
        python.chmod(0o700)

    def launch(self, *, overrides=None, args=()):
        env = dict(os.environ)
        for key in ('HOURS', 'EVICT_RAW', 'DISK_CAPACITY_GB'):
            env.pop(key, None)
        env.update(REPO_ROOT=str(self.root), OUT_ROOT=str(self.root / 'output'),
                   PATH=str(self.bin) + ':' + os.environ.get('PATH', ''), LC0_TEST_CAPTURE=str(self.capture))
        env.update(overrides or {})
        result = subprocess.run(['/bin/bash', str(ROOT / 'scripts/run_vast_lc0.sh'), *args],
                                env=env, capture_output=True, text=True)
        argv = self.capture.read_text().splitlines() if self.capture.exists() else []
        return result, argv

    def test_shell_default_is_explicit_no_eviction_no_capacity_and_720_hours(self):
        result, argv = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--no-evict-raw', argv)
        self.assertNotIn('--evict-raw', argv)
        self.assertEqual(argv[argv.index('--disk-capacity-gb') + 1], '0')
        self.assertEqual(argv[argv.index('--hours') + 1], '720')

    def test_shell_enables_eviction_and_preserves_trailing_overrides(self):
        result, argv = self.launch(overrides={'EVICT_RAW': '1', 'DISK_CAPACITY_GB': '478'},
                                   args=('--no-evict-raw',))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--evict-raw', argv)
        self.assertEqual(argv[argv.index('--disk-capacity-gb') + 1], '478')
        self.assertEqual(argv[-1], '--no-evict-raw')

    def test_shell_rejects_invalid_eviction_values_before_starting_python(self):
        for value in ('yes', '-1', ''):
            with self.subTest(value=value):
                result, argv = self.launch(overrides={'EVICT_RAW': value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('EVICT_RAW', result.stderr)
                self.assertEqual(argv, [])

    def test_prospective_supervisor_config_explicitly_selects_478gb_and_eviction(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read(ROOT / 'deploy/vast/piebot_lc0.conf')
        env = config['program:piebot_lc0']['environment']
        self.assertIn('EVICT_RAW="1"', env)
        self.assertIn('DISK_CAPACITY_GB="478"', env)
        self.assertIn('HOURS="720"', env)
        self.assertNotIn('GATE_INTERVAL_HOURS', env)
        self.assertNotIn('GATE_ON_PASS_END', env)


if __name__ == '__main__':
    unittest.main()
