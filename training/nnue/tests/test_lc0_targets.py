import gzip
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import train_stub, train_torch
from training.nnue.dataloader import jsonl_to_training_samples


FEN = "k7/8/8/8/8/8/8/K7 w - - 0 1"
TARGET_ARGS = dict(loss_kind="wdl", target_cp=250.0, teacher_mix=0.8,
                   max_teacher_cp=1500.0, outcome_decay=1.0,
                   min_teacher_depth=0, wdl_scale_cp=400.0)


def record(**values):
    return next(jsonl_to_training_samples([dict(
        fen=FEN, result=1, result_q=1.0, best_q=0.5, **values
    )]))


def make_corpus_smoke_fixture(root):
    """Small official-format archive exercising the actual importer boundary."""
    from training.nnue import lc0_corpus
    from training.nnue.tests.test_lc0_corpus import archive, good_record

    games = []
    for i in range(20):
        # Distinct searches of the same small positions are separate game
        # payloads, while duplicated payloads must be removed by the importer.
        payload = (good_record(visits=i) + good_record(side_to_move_or_enpassant=1, visits=i)
                   + good_record(invariance_info=1 << 4, visits=i)
                   + good_record(invariance_info=1 << 6, visits=i))
        games.append((f"training.{i}.gz", "2026-08-01T00:00:00", payload))
    entry = archive(root / "games.tar", games)
    manifest = root / "raw.json"
    manifest.write_text(json.dumps({"files": [entry]}))
    corpus_path = lc0_corpus.prepare_corpus(
        manifest, root / "corpus", since="2026-07-07", until="2026-09-08T00:00:00Z",
        validation_fraction=0.5, validation_samples=12, chunk_positions=100,
        min_free_bytes=0, workers=1,
    )
    corpus = json.loads(corpus_path.read_text())
    training = root / "training.jsonl"
    with training.open("wt") as dest:
        for chunk in corpus["chunks"]:
            with gzip.open(chunk["path"], "rt") as source:
                dest.write(source.read())
    return training, Path(corpus["validation"]["path"]), corpus


class Lc0TargetTests(unittest.TestCase):
    def test_q_mixture_uses_direct_probabilities_not_cp_sigmoid(self):
        sample = record()
        self.assertEqual(sample.best_q, 0.5)
        cp, p = train_stub._targets_for_record(
            sample, target_mode="lc0-q-outcome", **TARGET_ARGS)
        self.assertAlmostEqual(p, 0.8)
        self.assertAlmostEqual(cp, 400.0 * math.log(4.0))

    def test_max_length_outcome_uses_q_alone(self):
        sample = record(outcome_valid=False)
        _, p = train_stub._targets_for_record(
            sample, target_mode="lc0-q-outcome", **TARGET_ARGS)
        self.assertEqual(p, 0.75)

    def test_invalid_lc0_labels_fail_instead_of_being_clamped(self):
        for key, value in [("best_q", None), ("best_q", float("nan")),
                           ("best_q", float("inf")), ("best_q", 1.01),
                           ("best_q", "bad"), ("result_q", -1.01),
                           ("result_q", float("nan")), ("result_q", "bad")]:
            row = dict(fen=FEN, result=1, result_q=1.0, best_q=0.5)
            row[key] = value
            sample = next(jsonl_to_training_samples([row]))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                train_stub._targets_for_record(
                    sample, target_mode="lc0-q-outcome", **TARGET_ARGS)

    def test_target_identity_changes_only_for_opt_in(self):
        arguments = dict(**TARGET_ARGS, huber_delta_cp=100.0)
        old = train_stub.objective_metadata(**arguments)
        explicit_old = train_stub.objective_metadata(target_mode="selfplay", **arguments)
        self.assertEqual(old, explicit_old)
        self.assertNotIn("target_mode", old)
        new = train_stub.objective_metadata(target_mode="lc0-q-outcome", **arguments)
        self.assertEqual(new["target_mode"], "lc0-q-outcome")
        self.assertEqual(new["teacher_mix"], 0.8)
        self.assertNotEqual(old, new)


class Lc0FeatureConstructionTests(unittest.TestCase):
    def test_skipping_legacy_indices_preserves_all_rows_targets_and_provenance(self):
        rows = [
            dict(fen=FEN, best_q=0.5, result_q=1.0, outcome_valid=False),
            dict(fen=FEN.replace(" w ", " b "), best_q=-0.5, result_q=-1.0),
            dict(fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", best_q=0.0,
                 result_q=0.0),
        ]
        for index, row in enumerate(rows):
            row.update(run_id="fixture", game_id="game", ply=index)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rows.jsonl"
            source.write_text("\n" + "\n\n".join(map(json.dumps, rows)) + "\n")
            original = list(train_stub.iterate_lc0_samples(source, len(rows)))
            with mock.patch.object(train_stub, "_active_halfkp_indices",
                                   side_effect=AssertionError("unused indices built")):
                optimized = list(train_stub.iterate_lc0_samples(
                    source, len(rows), include_legacy_features=False))
            self.assertEqual(len(optimized), len(rows))
            self.assertEqual([record for _, record in original],
                             [record for _, record in optimized])
            self.assertEqual(original[0][0], [])  # Bare kings are still a row.
            self.assertTrue(original[2][0])
            for (_, before), (_, after) in zip(original, optimized):
                self.assertEqual(
                    train_stub._targets_for_record(before, target_mode="lc0-q-outcome", **TARGET_ARGS),
                    train_stub._targets_for_record(after, target_mode="lc0-q-outcome", **TARGET_ARGS))
            with self.assertRaisesRegex(ValueError, "max_samples"):
                list(train_stub.iterate_lc0_samples(
                    source, len(rows) - 1, include_legacy_features=False))

    def test_skipping_legacy_indices_preserves_fen_and_target_validation(self):
        invalid = [dict(fen="k7/8/8/8/8/8/8/K6 w - - 0 1"),
                   dict(fen="k7/8/8/8/8/8/8/K8 w - - 0 1"),
                   dict(fen="k7/8/8/8/8/8/8/8K w - - 0 1"),
                   dict(fen="k7/8/8/8/8/8/K7 w - - 0 1"),
                   dict(best_q=float("nan")), dict(best_q=1.1),
                   dict(result_q=-1.1)]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rows.jsonl"
            for changes in invalid:
                source.write_text(json.dumps(dict(fen=FEN, best_q=0.5,
                                                  result_q=1.0) | changes) + "\n")
                for include in (True, False):
                    with self.subTest(changes=changes, include=include), self.assertRaises(ValueError):
                        list(train_stub.iterate_lc0_samples(
                            source, 1, include_legacy_features=include))


@unittest.skipUnless(train_torch.torch_available(), "torch is not installed")
class Lc0TrainerTests(unittest.TestCase):
    def setUp(self):
        train_torch.torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data.jsonl"
        self.validation = self.root / "validation.jsonl"
        rows = [dict(fen=FEN, result_q=1.0, best_q=0.5, outcome_valid=False),
                dict(fen=FEN.replace(" w ", " b "), result_q=1.0, best_q=0.5)]
        for index, row in enumerate(rows):
            row.update(run_id="fixture", game_id="train", ply=index)
        self.data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        validation_row = dict(rows[1], game_id="validation")
        self.validation.write_text(json.dumps(validation_row) + "\n")

    def train(self, **overrides):
        arguments = dict(jsonl_dir=self.data, validation_jsonl_dir=self.validation,
                         max_samples=2, max_validation_samples=1, hidden_dim=2,
                         arch="v2", epochs=1, batch_size=2, val_split=0.0,
                         learning_rate=0.001, target_mode="lc0-q-outcome",
                         checkpoint_selection="latest", out_dir=self.root / "out",
                         device="cpu", teacher_sample_fraction=0.0, **TARGET_ARGS)
        arguments.update(overrides)
        return train_torch.train_model(**arguments)

    def test_training_and_fixed_validation_flip_black_targets_exactly_once(self):
        captured = []
        original = train_torch._objective_loss

        def capture(pred_cp, target_cp, target_wdl, **kwargs):
            captured.extend(round(float(v), 5) for v in target_wdl.tolist())
            return original(pred_cp, target_cp, target_wdl, **kwargs)

        with mock.patch.object(train_torch, "_objective_loss", side_effect=capture):
            metrics = self.train()
        self.assertEqual(metrics["train_samples"], 2)
        self.assertEqual(metrics["reference_val_samples"], 1)
        self.assertEqual(set(captured), {0.75, 0.2})

    def test_lc0_chunk_limit_rejects_silent_subsampling(self):
        with self.assertRaisesRegex(ValueError, "max_samples"):
            self.train(max_samples=1)

    def test_v2_lc0_does_not_build_legacy_indices_for_training_or_holdout(self):
        with mock.patch.object(train_stub, "_active_halfkp_indices",
                               side_effect=AssertionError("unused indices built")):
            metrics = self.train()
        self.assertEqual(metrics["train_samples"], 2)
        self.assertEqual(metrics["reference_val_samples"], 1)

    def test_latest_reuses_final_epoch_evaluations_with_exact_artifact_parity(self):
        from training.nnue.run_pipeline import _export_v2_checkpoint

        # The reference best mode selects the same final weights, and retains
        # the original explicit selected-checkpoint evaluation path.
        rows = [json.loads(line) for line in self.data.read_text().splitlines()]
        for index, row in enumerate(rows):
            row["fen"] = "4k3/8/8/8/8/8/4P3/4K3 " + ("w" if index == 0 else "b") + " - - 0 1"
            row["game_id"] = f"train-{index}"
        self.data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        original_eval = train_torch._eval_split
        self.train(out_dir=self.root / "parent")
        resume = dict(initial_checkpoint=self.root / "parent" / "checkpoint.json",
                      initial_optimizer_state=self.root / "parent" / "optimizer.pt")
        cases = ((1, False, 0.0, True), (1, True, 0.0, True),
                 (2, True, 0.0, True), (1, True, 0.5, True), (1, True, 0.0, False))
        for index, (epochs, warm, split, holdout) in enumerate(cases):
            with self.subTest(epochs=epochs, warm=warm, split=split, holdout=holdout):
                options = dict(epochs=epochs, val_split=split,
                               validation_jsonl_dir=self.validation if holdout else None,
                               **(resume if warm else {}))
                latest_dir = self.root / f"latest-{index}"
                best_dir = self.root / f"best-{index}"
                with mock.patch.object(train_stub, "is_better_checkpoint", return_value=True), \
                     mock.patch.object(train_torch, "_eval_split", wraps=original_eval) as evaluate:
                    latest = self.train(out_dir=latest_dir, **options)
                splits = 1 + int(split > 0.0) + int(holdout)
                # Evaluate each split per epoch and for the incoming warm
                # start, without an extra pass over the unchanged final model.
                self.assertEqual(evaluate.call_count, splits * (epochs + int(warm)))
                with mock.patch.object(train_stub, "is_better_checkpoint", return_value=True), \
                     mock.patch.object(train_torch, "_eval_split", wraps=original_eval) as evaluate:
                    reference = self.train(out_dir=best_dir, checkpoint_selection="best", **options)
                self.assertEqual(evaluate.call_count,
                                 splits * (epochs + int(warm)) + 1 + int(holdout))
                fields = [key for key in latest if key.startswith(("selected_val_", "selected_reference_",
                          "initial_train_", "initial_reference_")) or key.endswith("_history")]
                for field in fields:
                    self.assertEqual(latest[field], reference[field], field)
                checkpoints = [json.loads((path / "checkpoint.json").read_text())
                               for path in (latest_dir, best_dir)]
                for field in ("w1", "b1", "w2", "b2", "objective"):
                    self.assertEqual(checkpoints[0][field], checkpoints[1][field], field)
                states = [train_torch._torch_load(path / "optimizer.pt")
                          for path in (latest_dir, best_dir)]
                self.assertEqual(states[0]["model_parameters_sha256"], states[1]["model_parameters_sha256"])
                self.assertEqual(states[0]["state_dict"]["param_groups"], states[1]["state_dict"]["param_groups"])
                for parameter, values in states[0]["state_dict"]["state"].items():
                    for name, value in values.items():
                        self.assertTrue(train_torch.torch.equal(
                            value, states[1]["state_dict"]["state"][parameter][name]))
                for checkpoint, path in zip(checkpoints, (latest_dir, best_dir)):
                    _export_v2_checkpoint(checkpoint, quant_path=path / "model.nnue")
                self.assertEqual((latest_dir / "model.nnue").read_bytes(),
                                 (best_dir / "model.nnue").read_bytes())

    def test_v1_and_selfplay_keep_legacy_features_and_explicit_selected_evaluation(self):
        for arch, target_mode in (("v1", "lc0-q-outcome"), ("v2", "selfplay")):
            with self.subTest(arch=arch, target_mode=target_mode), \
                 mock.patch.object(train_stub, "_active_halfkp_indices",
                                   wraps=train_stub._active_halfkp_indices) as features, \
                 mock.patch.object(train_torch, "_eval_split", wraps=train_torch._eval_split) as evaluate:
                self.train(arch=arch, target_mode=target_mode)
            self.assertGreater(features.call_count, 0)
            self.assertEqual(evaluate.call_count, 4)

    def test_lc0_rejects_auxiliary_loss_or_wrong_probability_scale(self):
        for override in [dict(cp_loss_weight=1.0), dict(wdl_scale_cp=250.0),
                         dict(loss_kind="mse"), dict(min_teacher_depth=6)]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.train(**override)

    def test_latest_weights_and_bound_adam_survive_rejected_validation(self):
        first = self.train()
        parent_path = self.root / "out" / "checkpoint.json"
        optimizer_path = self.root / "out" / "optimizer.pt"
        parent = json.loads(parent_path.read_text())
        with mock.patch.object(train_stub, "is_better_checkpoint", return_value=False):
            resumed = self.train(initial_checkpoint=parent_path,
                                 initial_optimizer_state=optimizer_path,
                                 out_dir=self.root / "resumed")
        child = json.loads((self.root / "resumed" / "checkpoint.json").read_text())
        self.assertEqual(resumed["best_epoch"], 0)
        self.assertEqual(resumed["selected_epoch"], 1)
        self.assertEqual(resumed["checkpoint_selection"], "latest")
        self.assertTrue(resumed["optimizer_state_restored"])
        self.assertNotEqual(parent["b2"], child["b2"])
        self.assertIsNotNone(resumed["initial_reference_val_loss"])
        old_opt = train_torch.torch.load(optimizer_path, weights_only=False)
        new_opt = train_torch.torch.load(self.root / "resumed" / "optimizer.pt", weights_only=False)
        old_steps = [float(v["step"]) for v in old_opt["state_dict"]["state"].values()]
        new_steps = [float(v["step"]) for v in new_opt["state_dict"]["state"].values()]
        self.assertTrue(all(n == o + 1 for o, n in zip(old_steps, new_steps)))
        # A further restore verifies the optimizer SHA and parameter binding.
        self.train(initial_checkpoint=self.root / "resumed" / "checkpoint.json",
                   initial_optimizer_state=self.root / "resumed" / "optimizer.pt",
                   out_dir=self.root / "third")

    def test_new_objective_needs_weights_only_and_initializes_fresh_adam(self):
        self.train(target_mode="selfplay", checkpoint_selection="best")
        checkpoint = self.root / "out" / "checkpoint.json"
        optimizer = self.root / "out" / "optimizer.pt"
        with self.assertRaisesRegex(ValueError, "objective"):
            self.train(initial_checkpoint=checkpoint, out_dir=self.root / "strict")
        with self.assertRaisesRegex(ValueError, "forbids"):
            self.train(initial_checkpoint=checkpoint, initial_checkpoint_weights_only=True,
                       initial_optimizer_state=optimizer, out_dir=self.root / "bad")
        metrics = self.train(initial_checkpoint=checkpoint,
                             initial_checkpoint_weights_only=True,
                             out_dir=self.root / "new")
        self.assertFalse(metrics["optimizer_state_restored"])
        self.assertTrue(metrics["initialized_from"]["weights_only_objective_transition"])

    def test_official_binary_archive_through_v2_training_and_quant_export(self):
        from training.nnue.run_pipeline import _export_v2_checkpoint, _validate_export_artifacts

        training, validation, corpus = make_corpus_smoke_fixture(self.root)
        rows = [json.loads(line) for line in training.read_text().splitlines()]
        self.assertGreater(len(rows), 0)
        self.assertEqual(corpus["stats"]["rejected_records"], 20)
        black = next(row for row in rows if row["fen"].split()[1] == "b")
        self.assertEqual(black["best_q"], -0.5)
        self.assertEqual(black["result_q"], -1.0)
        self.assertTrue(any(not row["outcome_valid"] for row in rows))
        seen_probabilities = []
        original = train_torch._objective_loss

        def observe(pred_cp, target_cp, target_wdl, **kwargs):
            seen_probabilities.extend(round(float(v), 5) for v in target_wdl.tolist())
            return original(pred_cp, target_cp, target_wdl, **kwargs)

        with mock.patch.object(train_torch, "_objective_loss", side_effect=observe):
            metrics = self.train(jsonl_dir=training, validation_jsonl_dir=validation,
                                 max_samples=100, max_validation_samples=12,
                                 hidden_dim=8, batch_size=16)
        # Black labels go stm -> white at import -> stm at the v2 trainer.
        self.assertEqual(set(seen_probabilities), {0.8, 0.75})
        self.assertEqual(metrics["train_samples"], len(rows))
        self.assertEqual(metrics["train_records_with_teacher_value"], len(rows))
        self.assertEqual(metrics["validation_records_with_teacher_value"], 12)
        self.assertTrue(metrics["teacher_sampling_satisfied"])
        self.assertEqual(metrics["sampling_schema"], "complete-lc0-chunk-v1")
        self.assertEqual(metrics["checkpoint_selection_schema"], "latest-complete-epoch-v1")
        checkpoint = json.loads((self.root / "out" / "checkpoint.json").read_text())
        quant = self.root / "trained.nnue"
        _export_v2_checkpoint(checkpoint, quant_path=quant)
        dimensions, files = _validate_export_artifacts(checkpoint, dense_path=None, quant_path=quant)
        self.assertEqual(dimensions, (40960, 8, 1))
        self.assertEqual(files, [quant])
        raw = quant.read_bytes()
        self.assertEqual(raw[:8], b"PIENNQ02")
        self.assertEqual(struct.unpack("<iii", raw[24:36]), (255, 64, 400))


if __name__ == "__main__":
    unittest.main()
