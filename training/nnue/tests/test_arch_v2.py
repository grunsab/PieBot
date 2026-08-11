"""Arch-v2 trainer/exporter tests (dual-perspective SCReLU, PIENNQ02)."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from training.nnue import features_v2
from training.nnue.run_pipeline import _export_v2_checkpoint

try:
    import torch  # noqa: F401

    _TORCH = True
except Exception:
    _TORCH = False

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "4k3/8/8/8/8/7r/4P3/4K3 b - - 0 1"


class FeaturesV2Tests(unittest.TestCase):
    def test_dimensions_and_bounds(self) -> None:
        self.assertEqual(features_v2.PER_PERSPECTIVE_DIM, 40_960)
        white_p, black_p, stm_white = features_v2.active_indices(START)
        self.assertTrue(stm_white)
        self.assertEqual(len(white_p), 30)
        self.assertEqual(len(black_p), 30)
        for idx in white_p + black_p:
            self.assertTrue(0 <= idx < 40_960)

    def test_stm_ordering_flips_for_black(self) -> None:
        white_p, black_p, stm_white = features_v2.active_indices(BLACK_TO_MOVE)
        self.assertFalse(stm_white)
        stm, opp = features_v2.stm_ordered(BLACK_TO_MOVE)
        self.assertEqual(stm, black_p)
        self.assertEqual(opp, white_p)

    def test_color_symmetry_of_perspective_relative_indices(self) -> None:
        # A color-mirrored position must produce mirrored perspective indices:
        # white's view of the original == black's view of the mirror.
        original = "4k3/8/8/8/8/7r/4P3/4K3 w - - 0 1"
        mirrored = "4k3/4p3/7R/8/8/8/8/4K3 b - - 0 1"
        ow, ob, _ = features_v2.active_indices(original)
        mw, mb, _ = features_v2.active_indices(mirrored)
        self.assertEqual(sorted(ow), sorted(mb))
        self.assertEqual(sorted(ob), sorted(mw))


class ExportV2Tests(unittest.TestCase):
    def _tiny_checkpoint(self, hidden: int = 2) -> dict:
        input_dim = features_v2.PER_PERSPECTIVE_DIM
        return {
            "arch": "v2",
            "input_dim": input_dim,
            "hidden_dim": hidden,
            "quant_qa": 255,
            "quant_qb": 64,
            "wdl_scale_cp": 400.0,
            "w1": [0.001] * (hidden * input_dim),
            "b1": [0.5] * hidden,
            "w2": [1.0, -1.0] * hidden,
            "b2": 0.25,
        }

    def test_export_writes_piennq02_with_correct_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant = Path(tmp) / "v2.nnue"
            meta = _export_v2_checkpoint(self._tiny_checkpoint(), quant_path=quant)
            self.assertEqual(meta["quant_format"], "PIENNQ02")
            raw = quant.read_bytes()
            self.assertEqual(raw[:8], b"PIENNQ02")
            version, input_dim, hidden, output_dim = struct.unpack("<IIII", raw[8:24])
            qa, qb, scale = struct.unpack("<iii", raw[24:36])
            self.assertEqual(
                (input_dim, hidden, output_dim, qa, qb, scale),
                (40_960, 2, 1, 255, 64, 400),
            )
            expected_len = 36 + 2 * input_dim * hidden + 2 * hidden + 2 * hidden + 4
            self.assertEqual(len(raw), expected_len)

    def test_checkpoint_dimensions_accepts_v2_double_width_head(self) -> None:
        # 2026-08-08 cycle-1 incident: the legacy dimension validator required
        # len(w2) == hidden and crash-looped the first v6 training cycle.
        from training.nnue.run_pipeline import _checkpoint_dimensions

        ckpt = self._tiny_checkpoint(hidden=2)
        self.assertEqual(_checkpoint_dimensions(ckpt), (40_960, 2, 1))
        bad = dict(ckpt)
        bad["w2"] = [1.0, -1.0]  # v1-width head on a v2 checkpoint
        with self.assertRaises(ValueError):
            _checkpoint_dimensions(bad)

    def test_v2_export_passes_arch_aware_artifact_validation(self) -> None:
        # 2026-08-08 cycle-1 incident #2: the export stage unconditionally
        # validated and finalized the dense artifact, which the v2 path never
        # writes, crashing on nnue_dense.nnue.tmp. The arch-aware validator
        # must accept a dense-free v2 export and reject a corrupted one.
        from training.nnue.run_pipeline import _validate_export_artifacts

        ckpt = self._tiny_checkpoint(hidden=2)
        with tempfile.TemporaryDirectory() as tmp:
            quant = Path(tmp) / "v2.nnue"
            _export_v2_checkpoint(ckpt, quant_path=quant)
            dims, files = _validate_export_artifacts(
                ckpt, dense_path=None, quant_path=quant
            )
            self.assertEqual(dims, (40_960, 2, 1))
            self.assertEqual(files, [quant])
            quant.write_bytes(quant.read_bytes()[:-2])  # truncate
            with self.assertRaises(ValueError):
                _validate_export_artifacts(ckpt, dense_path=None, quant_path=quant)

    def test_export_rejects_size_mismatches(self) -> None:
        bad = self._tiny_checkpoint()
        bad["w2"] = [1.0]  # must be 2 * hidden
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _export_v2_checkpoint(bad, quant_path=Path(tmp) / "v2.nnue")


@unittest.skipUnless(_TORCH, "torch not installed")
class TrainV2Tests(unittest.TestCase):
    def test_tiny_v2_training_produces_v2_checkpoint(self) -> None:
        from training.nnue import train_torch

        fens = [START, BLACK_TO_MOVE,
                "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN5/PPP2PPP/R2Q1RK1 w - - 0 12"]
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for g in range(20):
                for p, fen in enumerate(fens):
                    rows.append({
                        "fen": fen, "result": (g + p) % 3 - 1,
                        "outcome_valid": True,
                        "value_cp": 50.0 * ((g + p) % 5 - 2),
                        "teacher_depth": 6, "run_id": "t",
                        "game_id": f"g{g}", "ply": p,
                    })
            src = Path(tmp) / "data"
            src.mkdir()
            (src / "shard_000000.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            out = Path(tmp) / "out"
            metrics = train_torch.train_model(
                jsonl_dir=src, out_dir=out, arch="v2", hidden_dim=4,
                epochs=1, batch_size=16, max_samples=60, loss_kind="wdl",
                device="cpu", teacher_mix=0.8, min_teacher_depth=5,
            )
            self.assertTrue(metrics)
            # 2026-08-08 incident #3: metrics stamped the v1 feature-set
            # constant, poisoning the state's training_model_identity and
            # arming a lineage-validator refusal on the next restart.
            saved_metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(
                saved_metrics["feature_set"], features_v2.FEATURE_SET_V2
            )
            ckpt = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(ckpt["format"], "piebot-halfkp-dp-screlu-v1-torch")
            self.assertEqual(ckpt["arch"], "v2")
            self.assertEqual(ckpt["feature_set"], features_v2.FEATURE_SET_V2)
            self.assertEqual(ckpt["input_dim"], features_v2.PER_PERSPECTIVE_DIM)
            self.assertEqual(len(ckpt["w2"]), 2 * ckpt["hidden_dim"])
            # Output weights stay inside the int8-at-QB envelope.
            limit = 127.0 / float(ckpt["quant_qb"])
            self.assertTrue(all(abs(v) <= limit + 1e-6 for v in ckpt["w2"]))
            quant = Path(tmp) / "trained.nnue"
            meta = _export_v2_checkpoint(ckpt, quant_path=quant)
            self.assertEqual(meta["quant_format"], "PIENNQ02")


@unittest.skipUnless(_TORCH, "torch is not installed")
class ScreluFreshInitTests(unittest.TestCase):
    """The SCReLU head is clamp(h, 0, 1)**2, whose derivative is exactly zero
    outside (0, 1). A HalfKP bag sums ~30 active features, so the accumulator's
    spread is sqrt(30) times the per-weight scale: leaving EmbeddingBag at
    PyTorch's default N(0, 1) puts half the units at 0 and most of the rest
    saturated at 1, and a unit with zero derivative learns nothing. A fresh
    net must instead start with most of its width inside the live range.
    """

    HIDDEN = 1024
    ACTIVE_FEATURES = 30

    def _fresh_preactivations(self) -> "torch.Tensor":
        import torch

        from training.nnue.train_torch import TorchNnueV2

        torch.manual_seed(20260811)
        model = TorchNnueV2(
            input_dim=features_v2.PER_PERSPECTIVE_DIM,
            hidden_dim=self.HIDDEN,
            wdl_scale=400.0,
        )
        batch = 64
        # One bag per sample, each summing ACTIVE_FEATURES distinct features,
        # which is the shape a real HalfKP perspective produces.
        flat = torch.randint(
            0,
            features_v2.PER_PERSPECTIVE_DIM,
            (batch * self.ACTIVE_FEATURES,),
            dtype=torch.long,
        )
        offsets = torch.arange(0, batch * self.ACTIVE_FEATURES, self.ACTIVE_FEATURES)
        with torch.no_grad():
            return model.embed(flat, offsets) + model.b1

    def test_most_hidden_units_start_inside_the_screlu_live_range(self) -> None:
        pre = self._fresh_preactivations()
        live = ((pre > 0.0) & (pre < 1.0)).float().mean().item()
        self.assertGreater(
            live,
            0.5,
            "only {:.1%} of hidden units start inside SCReLU's differentiable "
            "range (0, 1); the rest carry zero gradient, so the net trains at a "
            "fraction of the width it pays full evaluation cost for".format(live),
        )

    def test_fresh_accumulator_is_not_wildly_overscaled(self) -> None:
        pre = self._fresh_preactivations()
        std = pre.std().item()
        self.assertLess(
            std,
            1.5,
            "fresh accumulator std is {:.2f}; SCReLU's whole live range is one "
            "unit wide, so this saturates the nonlinearity".format(std),
        )


if __name__ == "__main__":
    unittest.main()
