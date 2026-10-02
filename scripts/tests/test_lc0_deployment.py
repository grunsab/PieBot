"""Deployment and rollback contracts for the separate LCZero campaign."""
import configparser
import hashlib
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import lc0_deploy

ROOT = Path(__file__).resolve().parents[2]


class Lc0DeploymentTests(unittest.TestCase):
    def test_deployer_parser_defaults_to_30_days_and_keeps_explicit_overrides(self):
        original = lc0_deploy.argparse.ArgumentParser.parse_args

        class Parsed(Exception):
            pass

        for argv, expected in (([], 720.0), (['--hours', '48'], 48.0)):
            with self.subTest(argv=argv):
                observed = []

                def capture(parser, args):
                    observed.append(original(parser, args))
                    raise Parsed

                with mock.patch.object(lc0_deploy.argparse.ArgumentParser, 'parse_args', autospec=True, side_effect=capture):
                    with self.assertRaises(Parsed):
                        lc0_deploy.main(argv)
                self.assertEqual(observed[0].hours, expected)

    def test_launcher_defaults_to_30_days_and_forwards_hours(self):
        launcher = (ROOT / 'scripts/run_vast_lc0.sh').read_text()
        self.assertIn('HOURS="${HOURS:-720}"', launcher)
        self.assertIn('--hours "$HOURS"', launcher)

    def test_parent_waits_for_training_child_on_group_shutdown(self):
        received = []
        previous = signal.signal(signal.SIGTERM, lambda *_: received.append('term'))
        try:
            def child_run(*args, **kwargs):
                os.kill(os.getpid(), signal.SIGTERM)
                self.assertEqual(received, [])
                return 'child checkpoint committed'
            with mock.patch.object(lc0_deploy.subprocess, 'run', side_effect=child_run):
                self.assertEqual(lc0_deploy.wait_for_training(['trainer']),
                                 'child checkpoint committed')
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertEqual(received, ['term'])
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_invalid_bootstrap_does_not_create_source_pin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / 'checkpoint.json'
            checkpoint.write_text(json.dumps({'format': 'wrong', 'hidden_dim': 1024}))
            quant = root / 'quant.nnue'
            quant.write_bytes(b'PIENNQ02' + bytes(64))
            with self.assertRaises(ValueError):
                lc0_deploy.validate_bootstrap(
                    checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    quant, hashlib.sha256(quant.read_bytes()).hexdigest())
            self.assertFalse((root / 'source_git_commit').exists())

    def test_bootstrap_checksum_is_required_even_with_valid_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / 'checkpoint.json'
            checkpoint.write_text('{}')
            quant = root / 'quant.nnue'
            quant.write_bytes(b'PIENNQ02')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                lc0_deploy.validate_bootstrap(checkpoint, '0' * 64, quant, '0' * 64)

    def test_source_pin_refuses_a_different_revision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lc0_deploy.pin_source(root, 'a' * 40)
            lc0_deploy.pin_source(root, 'a' * 40)
            with self.assertRaises(ValueError):
                lc0_deploy.pin_source(root, 'b' * 40)
            self.assertEqual((root / 'source_git_commit').read_text().strip(), 'a' * 40)

    def test_output_root_cannot_overwrite_selfplay_or_its_parent(self):
        protected = Path('/workspace/piebot_campaign_v8')
        for output in (protected, protected / 'lc0', protected.parent):
            with self.assertRaises(ValueError):
                lc0_deploy.assert_separate_root(output, protected)
        lc0_deploy.assert_separate_root(Path('/workspace/piebot_lc0_20260907'), protected)

    def test_supervisor_separates_campaign_and_stops_process_group(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read(ROOT / 'deploy/vast/piebot_lc0.conf')
        run = config['program:piebot_lc0']
        self.assertTrue(run.getboolean('stopasgroup'))
        self.assertTrue(run.getboolean('killasgroup'))
        self.assertEqual(run['autorestart'], 'unexpected')
        self.assertIn('HOURS="720"', run['environment'])
        self.assertIn('/workspace/piebot_lc0_repo', run['directory'])

    def parse(self, argv):
        return lc0_deploy.build_parser().parse_args(argv)

    def command(self, argv, corpora=('/out/data/corpus/corpus_manifest.json',)):
        args = self.parse(argv)
        return lc0_deploy.training_command(
            args, python='python3', repo=Path('/repo'), output=Path('/out'),
            corpora=[Path(corpus) for corpus in corpora], checkpoint=Path('/boot/checkpoint.json'), active=Path('/boot/active.nnue'), commit='a' * 40)

    def test_default_training_command_carries_no_new_lineage_options(self):
        command = self.command([])
        self.assertEqual(command[command.index('--corpus-manifest') + 1],
                         '/out/data/corpus/corpus_manifest.json')
        for flag in ('--extra-corpus-manifest', '--lr-gamma', '--learning-rate',
                     '--lr-epoch-positions', '--gate-movetime-ms'):
            self.assertNotIn(flag, command)

    def test_training_command_forwards_schedule_gate_and_keeps_primary_validation_corpus(self):
        command = self.command(['--primary-corpus-manifest', '/old/corpus_manifest.json',
                                '--learning-rate', '0.0004375', '--lr-gamma', '0.995',
                                '--lr-epoch-positions', '100000000', '--gate-movetime-ms', '1000'])
        def value(flag):
            return command[command.index(flag) + 1]
        # The existing corpus stays primary so its fixed validation set, and
        # therefore the loss scale, is unchanged; the new window only adds chunks.
        self.assertEqual(value('--corpus-manifest'), '/old/corpus_manifest.json')
        self.assertEqual(value('--extra-corpus-manifest'), '/out/data/corpus/corpus_manifest.json')
        self.assertEqual(float(value('--learning-rate')), 0.0004375)
        self.assertEqual(float(value('--lr-gamma')), 0.995)
        self.assertEqual(int(value('--lr-epoch-positions')), 100000000)
        self.assertEqual(int(value('--gate-movetime-ms')), 1000)

    def test_explicit_initial_checkpoint_requires_its_checksum(self):
        default = lc0_deploy.resolve_bootstrap(self.parse([]))
        self.assertEqual(default, (Path('/workspace/piebot_campaign_v8/cycles/cycle_000206/train/checkpoint.json'),
                                   lc0_deploy.CHECKPOINT_SHA))
        chosen = lc0_deploy.resolve_bootstrap(self.parse(
            ['--initial-checkpoint', '/boot/c.json', '--initial-checkpoint-sha256', 'b' * 64]))
        self.assertEqual(chosen, (Path('/boot/c.json'), 'b' * 64))
        for argv in (['--initial-checkpoint', '/boot/c.json'],
                     ['--initial-checkpoint-sha256', 'b' * 64],
                     ['--initial-checkpoint', '/boot/c.json', '--initial-checkpoint-sha256', 'xyz']):
            with self.assertRaises(ValueError):
                lc0_deploy.resolve_bootstrap(self.parse(argv))

    def test_each_extra_window_is_acquired_separately_and_trained_as_an_extra_corpus(self):
        # One frozen inventory is downloaded whole before conversion evicts it,
        # so a long span must be split into windows that each fit on disk.
        args = self.parse(['--since', '2026-03-07T00:00:00+00:00', '--until', '2026-04-07T00:00:00+00:00',
                           '--extra-window', '2026-04-07T00:00:00+00:00,2026-05-07T00:00:00+00:00',
                           '--extra-window', '2026-05-07T00:00:00+00:00,2026-06-07T00:00:00+00:00'])
        windows = lc0_deploy.corpus_windows(args, Path('/out'))
        self.assertEqual(windows, [
            ('2026-03-07T00:00:00+00:00', '2026-04-07T00:00:00+00:00',
             Path('/out/data/raw'), Path('/out/data/corpus')),
            ('2026-04-07T00:00:00+00:00', '2026-05-07T00:00:00+00:00',
             Path('/out/data/window_01/raw'), Path('/out/data/window_01/corpus')),
            ('2026-05-07T00:00:00+00:00', '2026-06-07T00:00:00+00:00',
             Path('/out/data/window_02/raw'), Path('/out/data/window_02/corpus')),
        ])
        command = self.command([], corpora=('/out/a.json', '/out/b.json', '/out/c.json'))
        self.assertEqual(command[command.index('--corpus-manifest') + 1], '/out/a.json')
        extras = [command[i + 1] for i, item in enumerate(command) if item == '--extra-corpus-manifest']
        self.assertEqual(extras, ['/out/b.json', '/out/c.json'])
        command = self.command(['--primary-corpus-manifest', '/old.json'], corpora=('/out/a.json', '/out/b.json'))
        self.assertEqual(command[command.index('--corpus-manifest') + 1], '/old.json')
        extras = [command[i + 1] for i, item in enumerate(command) if item == '--extra-corpus-manifest']
        self.assertEqual(extras, ['/out/a.json', '/out/b.json'])
        for bad in ('2026-04-07', 'a,b,c', ','):
            with self.assertRaises(ValueError):
                lc0_deploy.corpus_windows(self.parse(['--extra-window', bad]), Path('/out'))

    def test_filter_and_result_weight_options_are_forwarded_only_when_given(self):
        default = self.command([])
        for flag in ('--teacher-mix', '--skip-early-plies', '--skip-in-check', '--skip-before-capture'):
            self.assertNotIn(flag, default)
        command = self.command(['--teacher-mix', '0.7', '--skip-early-plies', '16',
                                '--skip-in-check', '--skip-before-capture'])
        self.assertEqual(float(command[command.index('--teacher-mix') + 1]), 0.7)
        self.assertEqual(int(command[command.index('--skip-early-plies') + 1]), 16)
        self.assertIn('--skip-in-check', command)
        self.assertIn('--skip-before-capture', command)

    def test_prepare_only_is_off_by_default(self):
        self.assertFalse(self.parse([]).prepare_only)
        self.assertTrue(self.parse(['--prepare-only']).prepare_only)


if __name__ == '__main__':
    unittest.main()
