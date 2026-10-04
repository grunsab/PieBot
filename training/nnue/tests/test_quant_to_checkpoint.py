import json
import random
import tempfile
import unittest
from pathlib import Path

from training.nnue import quant_to_checkpoint, run_pipeline, train_stub

OBJECTIVE = {"schema": "nnue-objective-v1", "target_mode": "lc0-q-outcome"}


def float_checkpoint(input_dim=6, hidden_dim=4, seed=1):
    rng = random.Random(seed)
    return {
        "format": "piebot-halfkp-dp-screlu-v1-torch", "arch": "v2",
        "input_dim": input_dim, "hidden_dim": hidden_dim, "quant_qa": 255, "quant_qb": 64,
        "wdl_scale_cp": 400.0,
        "w1": [rng.uniform(-1.5, 1.5) for _ in range(input_dim * hidden_dim)],
        "b1": [rng.uniform(-1.0, 1.0) for _ in range(hidden_dim)],
        "w2": [rng.uniform(-1.9, 1.9) for _ in range(2 * hidden_dim)],
        "b2": rng.uniform(-0.5, 0.5),
    }


class QuantToCheckpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def export(self, checkpoint, name):
        path = self.root / name
        run_pipeline._export_v2_checkpoint(checkpoint, quant_path=path)
        return path

    def test_recovered_checkpoint_exports_to_the_identical_playing_net(self):
        original = self.export(float_checkpoint(), "original.nnue")
        recovered = quant_to_checkpoint.checkpoint_from_quant(original, objective=OBJECTIVE)
        again = self.export(recovered, "again.nnue")
        self.assertEqual(again.read_bytes(), original.read_bytes())

    def test_recovered_checkpoint_carries_the_metadata_the_trainer_requires(self):
        original = self.export(float_checkpoint(), "original.nnue")
        recovered = quant_to_checkpoint.checkpoint_from_quant(original, objective=OBJECTIVE)
        self.assertEqual(recovered["format"], "piebot-halfkp-dp-screlu-v1-torch")
        self.assertEqual(recovered["arch"], "v2")
        self.assertEqual(recovered["feature_set"], "halfkp-dp-screlu-v1")
        self.assertEqual(recovered["target_schema"], train_stub.TARGET_SCHEMA)
        self.assertEqual(recovered["objective"], OBJECTIVE)
        self.assertEqual((recovered["input_dim"], recovered["hidden_dim"]), (6, 4))
        self.assertEqual((recovered["quant_qa"], recovered["quant_qb"]), (255, 64))
        self.assertEqual(len(recovered["w1"]), 24)
        self.assertEqual(len(recovered["w2"]), 8)
        self.assertEqual(recovered["recovered_from_quant"]["sha256"],
                         quant_to_checkpoint.sha256(original))

    def test_weights_keep_the_checkpoint_layout_not_the_file_layout(self):
        # The file stores w1 feature-major; checkpoints store it row-major [hidden][input].
        checkpoint = float_checkpoint()
        checkpoint["w1"] = [0.0] * 24
        checkpoint["w1"][1 * 6 + 5] = 1.0   # hidden unit 1, input feature 5
        recovered = quant_to_checkpoint.checkpoint_from_quant(
            self.export(checkpoint, "one.nnue"), objective=OBJECTIVE)
        self.assertEqual([i for i, v in enumerate(recovered["w1"]) if v], [1 * 6 + 5])
        self.assertAlmostEqual(recovered["w1"][11], 1.0, places=9)

    def test_truncated_or_foreign_files_are_rejected(self):
        original = self.export(float_checkpoint(), "original.nnue")
        data = original.read_bytes()
        for name, payload in (("magic.nnue", b"PIENNQ01" + data[8:]), ("short.nnue", data[:-3]),
                              ("long.nnue", data + b"\0")):
            path = self.root / name
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                quant_to_checkpoint.checkpoint_from_quant(path, objective=OBJECTIVE)

    def test_command_line_writes_json_and_reports_its_checksum(self):
        original = self.export(float_checkpoint(), "original.nnue")
        out = self.root / "recovered.json"
        code = quant_to_checkpoint.main(["--quant", str(original), "--out", str(out),
                                         "--objective-json", json.dumps(OBJECTIVE)])
        self.assertEqual(code, 0)
        again = self.export(json.loads(out.read_text()), "again.nnue")
        self.assertEqual(again.read_bytes(), original.read_bytes())
        with self.assertRaises(ValueError):
            quant_to_checkpoint.main(["--quant", str(original), "--out", str(out),
                                      "--objective-json", json.dumps(OBJECTIVE)])


if __name__ == "__main__":
    unittest.main()
