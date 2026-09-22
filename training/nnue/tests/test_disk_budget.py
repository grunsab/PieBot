import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training.nnue import disk_budget


class DiskBudgetTests(unittest.TestCase):
    def test_default_matches_actual_filesystem_free_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            usage = shutil.disk_usage(path)
            with mock.patch.object(disk_budget.shutil, 'disk_usage', return_value=usage) as probe:
                self.assertEqual(disk_budget.available_bytes(path), usage.free)
            probe.assert_called_once_with(path)

    def test_default_requires_only_the_existing_free_field(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage', return_value=SimpleNamespace(free=123)):
            self.assertEqual(disk_budget.available_bytes('/workspace'), 123)

    def test_capacity_clamps_larger_overlay_free_space(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage', return_value=SimpleNamespace(
                total=2_000_000_000_000, free=1_700_000_000_000)):
            self.assertEqual(disk_budget.available_bytes('/workspace', capacity_bytes=478_000_000_000),
                             178_000_000_000)

    def test_filesystem_free_space_remains_the_tighter_limit(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage', return_value=SimpleNamespace(total=100, free=10)):
            self.assertEqual(disk_budget.available_bytes('/workspace', capacity_bytes=478), 10)

    def test_capacity_already_used_or_exceeded_returns_zero(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage', return_value=SimpleNamespace(total=1000, free=400)):
            for capacity in (600, 478):
                with self.subTest(capacity=capacity):
                    self.assertEqual(disk_budget.available_bytes('/workspace', capacity_bytes=capacity), 0)

    def test_invalid_capacities_fail_before_filesystem_access(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage') as probe:
            for value in (True, False, 0, -1, 478.0, '478', float('nan'), float('inf')):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        disk_budget.available_bytes('/workspace', capacity_bytes=value)
            probe.assert_not_called()

    def test_each_call_probes_again_and_propagates_filesystem_errors(self):
        with mock.patch.object(disk_budget.shutil, 'disk_usage', side_effect=[
                SimpleNamespace(total=1000, free=600), SimpleNamespace(total=1000, free=500),
                FileNotFoundError('missing')]) as probe:
            self.assertEqual(disk_budget.available_bytes('/workspace', capacity_bytes=478), 78)
            self.assertEqual(disk_budget.available_bytes('/workspace', capacity_bytes=478), 0)
            with self.assertRaises(FileNotFoundError):
                disk_budget.available_bytes('/workspace', capacity_bytes=478)
            self.assertEqual(probe.call_count, 3)

    def test_decimal_gigabytes_and_disabled_zero(self):
        self.assertEqual(disk_budget.decimal_gb_bytes(478), 478_000_000_000)
        self.assertEqual(disk_budget.decimal_gb_bytes(478.25), 478_250_000_000)
        self.assertEqual(disk_budget.decimal_gb_bytes(0.000000001), 1)
        self.assertIsNone(disk_budget.decimal_gb_bytes(0))
        self.assertIsNone(disk_budget.decimal_gb_bytes(0.0))

    def test_decimal_gigabytes_reject_invalid_inputs(self):
        for value in (True, False, None, '478', -1, float('inf'), -float('inf'), float('nan'), 0.0000000001):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    disk_budget.decimal_gb_bytes(value)


if __name__ == '__main__':
    unittest.main()
