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
        self.assertIn('HOURS="336"', run['environment'])
        self.assertIn('/workspace/piebot_lc0_repo', run['directory'])


if __name__ == '__main__':
    unittest.main()
