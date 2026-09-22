import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import lc0_anchor_monitor as monitor


class AnchorMonitorTests(unittest.TestCase):
    def test_snapshot_requires_matching_model_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / 'model.nnue'
            model.write_bytes(b'example-model')
            expected = hashlib.sha256(model.read_bytes()).hexdigest()
            copied = monitor.snapshot_model(model, expected, root / 'job')
            model.write_bytes(b'changed')
            self.assertEqual(copied.read_bytes(), b'example-model')
            with self.assertRaises(ValueError):
                monitor.snapshot_model(model, expected, root / 'other-job')

    def test_recent_measured_model_does_not_repeat(self):
        state = {'last_completed_at': 100., 'last_model_sha256': 'same'}
        self.assertFalse(monitor.measurement_due(state, 'same', 100000.))
        self.assertFalse(monitor.measurement_due(state, 'new', 101.))
        self.assertTrue(monitor.measurement_due(state, 'new', 100000.))

    def test_fixed_anchor_protocol_keeps_the_canonical_rungs(self):
        command = monitor.ladder_command(Path('/repo'), Path('/model.nnue'),
                                         Path('/stockfish16'), Path('/job'), 20260907)
        for option, value in (('--rungs', '3000,3190'), ('--games', '100'),
                              ('--time-control', '60+0.5'), ('--piebot-blend', '75')):
            self.assertEqual(command[command.index(option) + 1], value)


if __name__ == '__main__':
    unittest.main()
