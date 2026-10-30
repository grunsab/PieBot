import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import w2048_watch as w  # noqa: E402

SHAPE = dict(buckets=16, hidden=8, l1=4, l2=8)


def write_ckpt(root: Path, n: int, floats: np.ndarray | None, complete: bool = True) -> Path:
    d = root / f"w2048-{n}"
    d.mkdir(parents=True)
    if floats is not None:
        # bullet pads quantised.bin with a few garbage bytes; the check ignores them.
        (d / "quantised.bin").write_bytes(floats.astype("<f4").tobytes() + b"\xff" * 12)
    if complete:
        (d / "raw.bin").write_bytes(b"x")
        (d / "optimiser_state").mkdir()
    return d


class ShapeTests(unittest.TestCase):
    def test_pairwise_float_count(self):
        nf = 16 * 768
        h, l1, l2 = 8, 4, 8
        want = nf * h + h + 8 * l1 * h + 8 * l1 + 8 * l2 * l1 + 8 * l2 + 8 * l2 + 8
        self.assertEqual(w.expected_floats(**SHAPE), want)


class ThreatShapeTests(unittest.TestCase):
    def test_threats_add_one_row_per_threat_feature_after_the_stack(self):
        self.assertEqual(w.expected_floats(**SHAPE, threats=True), w.expected_floats(**SHAPE) + 59808 * 8)

    def test_a_threats_checkpoint_is_checked_with_its_threat_rows(self):
        tmp = Path(tempfile.mkdtemp())
        n = w.expected_floats(**SHAPE, threats=True)
        d = write_ckpt(tmp, 20, np.full(n, 0.25))
        self.assertEqual(w.check_checkpoint(d, **SHAPE, threats=True), (True, "ok"))
        # Read as a plain net, the extra rows make the file the wrong size.
        self.assertFalse(w.check_checkpoint(d, **SHAPE)[0])

    def test_run_config_defaults_to_the_plain_w2048_run(self):
        cfg = w.RunConfig()
        self.assertFalse(cfg.threats)
        self.assertEqual(cfg.nets, Path("/workspace/w2048_nets"))
        self.assertIn("w2048-200_vs_pw1500_200g.json", str(cfg.gate_result("w2048", 200)))
        t = w.RunConfig(threats=True, nets=Path("/x/nets"), gates=Path("/x/gates"), incumbent_label="w2048best")
        self.assertEqual(t.gate_result("w2048t", 400), Path("/x/gates/w2048t-400_vs_w2048best_200g.json"))


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.n = w.expected_floats(**SHAPE)

    def test_sane_weights_pass(self):
        d = write_ckpt(self.tmp, 20, np.full(self.n, 0.5))
        self.assertEqual(w.check_checkpoint(d, **SHAPE), (True, "ok"))

    def test_nan_and_huge_values_fail(self):
        bad = np.full(self.n, 0.5)
        bad[3] = np.nan
        ok, why = w.check_checkpoint(write_ckpt(self.tmp, 20, bad), **SHAPE)
        self.assertFalse(ok)
        self.assertIn("non-finite", why)
        bad = np.full(self.n, 0.5)
        bad[-1] = 3e30   # an exponent bit flip
        ok, why = w.check_checkpoint(write_ckpt(self.tmp, 40, bad), **SHAPE)
        self.assertFalse(ok)
        self.assertIn("magnitude", why)

    def test_short_file_fails(self):
        ok, why = w.check_checkpoint(write_ckpt(self.tmp, 20, np.zeros(10)), **SHAPE)
        self.assertFalse(ok)
        self.assertIn("size", why)

    def test_latest_good_skips_bad_and_incomplete(self):
        good = np.full(self.n, 0.1)
        bad = good.copy()
        bad[0] = np.inf
        write_ckpt(self.tmp, 20, good)
        write_ckpt(self.tmp, 40, good)
        write_ckpt(self.tmp, 60, bad)
        write_ckpt(self.tmp, 80, good, complete=False)
        (self.tmp / "w2048-100").mkdir()      # being written
        self.assertEqual(w.latest_good(self.tmp, "w2048", **SHAPE), self.tmp / "w2048-40")
        self.assertIsNone(w.latest_good(self.tmp / "nothing", "w2048", **SHAPE))


class ScheduleTests(unittest.TestCase):
    def test_gates_every_200_and_nets_kept(self):
        self.assertEqual([n for n in range(20, 1001, 20) if w.gate_due(n)], [200, 400, 600, 800, 1000])

    def test_checkpoint_numbers_sorted_numerically(self):
        tmp = Path(tempfile.mkdtemp())
        for n in (100, 20, 1000, 200):
            (tmp / f"w2048-{n}").mkdir()
        (tmp / "other-5").mkdir()
        self.assertEqual([n for n, _ in w.checkpoints(tmp, "w2048")], [20, 100, 200, 1000])


class ManifestTests(unittest.TestCase):
    def test_done_set_reads_manifest(self):
        tmp = Path(tempfile.mkdtemp())
        m = tmp / "manifest.jsonl"
        m.write_text(json.dumps({"n": 20, "ok": True}) + "\n" + json.dumps({"n": 40, "ok": False}) + "\n")
        self.assertEqual(w.done_set(m), {20, 40})
        self.assertEqual(w.done_set(tmp / "missing.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
