"""The archive pool is configurable through the existing supervised launcher."""
import unittest
from unittest import mock
from training.nnue import lc0_deploy
from scripts.tests import test_lc0_storage_deployment as storage


class ParallelDeploymentTests(unittest.TestCase):
    setUp = storage.Lc0StorageDeploymentTests.setUp
    deploy = storage.Lc0StorageDeploymentTests.deploy

    def test_pool_defaults_to_four_and_explicit_override_reaches_fetch(self):
        for args, workers in (((), 4), (('--download-concurrency', '3'), 3)):
            with self.subTest(workers=workers):
                _, fetch, _, train = self.deploy(*args)
                self.assertEqual(int(fetch[fetch.index('--concurrency') + 1]), workers)
                self.assertNotIn('--concurrency', train)
                self.assertEqual(float(train[train.index('--hours') + 1]), 720)

    def test_bad_pool_size_is_rejected_before_bootstrap_or_source_pin(self):
        for value in ('0', '-1', '17'):
            with self.subTest(value=value), \
                 mock.patch.object(lc0_deploy, 'validate_bootstrap') as bootstrap, \
                 mock.patch.object(lc0_deploy, 'pin_source') as pin:
                with self.assertRaisesRegex(ValueError, 'concurrency'):
                    lc0_deploy.main(self.args + ['--download-concurrency', value])
                bootstrap.assert_not_called()
                pin.assert_not_called()


class ParallelLauncherTests(unittest.TestCase):
    setUp = storage.Lc0StorageLauncherTests.setUp
    launch = storage.Lc0StorageLauncherTests.launch

    def test_shell_forwards_download_pool_and_explicit_cli_override(self):
        result, argv = self.launch(overrides={'DOWNLOAD_CONCURRENCY': '3'},
                                   args=('--download-concurrency', '2'))
        self.assertEqual(result.returncode, 0, result.stderr)
        values = [argv[i + 1] for i, value in enumerate(argv) if value == '--download-concurrency']
        self.assertEqual(values, ['3', '2'])


if __name__ == '__main__':
    unittest.main()
