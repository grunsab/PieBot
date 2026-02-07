import unittest

from training.nnue import autopilot


class AutopilotTests(unittest.TestCase):
    def test_zen5_9755_profile_has_expected_relabel_defaults(self) -> None:
        profile = autopilot.zen5_9755_7d_profile()
        self.assertEqual(9, profile["teacher_relabel_depth"])
        self.assertEqual(8, profile["teacher_relabel_every"])
        self.assertGreaterEqual(profile["teacher_relabel_threads"], 32)
        self.assertGreaterEqual(profile["teacher_relabel_hash_mb"], 2048)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

