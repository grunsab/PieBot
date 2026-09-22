import gzip
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from training.nnue import lc0_autopilot


class LC0CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.chunks = []
        for i in range(3):
            path = self.corpus / f"chunk-{i}.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps({"game_id": str(i)}) + "\n")
            self.chunks.append({"path": str(path), "sha256": self.sha(path), "positions": 1})
        val = self.corpus / "validation.jsonl"
        val.write_text('{"game_id":"holdout"}\n')
        self.manifest = self.corpus / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema": "piebot-lc0-corpus-v1", "complete": True,
            "corpus_id": "frozen-corpus", "raw_manifest_sha256": "a" * 64,
            "since": "2026-07-07T00:00:00Z", "until": "2026-09-07T00:00:00Z",
            "seed": 42, "chunks": self.chunks,
            "validation": {"path": str(val), "sha256": self.sha(val), "positions": 1},
        }))
        self.initial = self.root / "selfplay" / "checkpoint.json"
        self.initial.parent.mkdir()
        self.initial.write_text(json.dumps({"arch": "v2", "hidden_dim": 1024, "input_dim": 40960}))
        self.incumbent = self.initial.parent / "accepted.nnue"
        self.incumbent.write_bytes(b"PIENNQ02" + b"incumbent")
        self.out = self.root / "lc0"
        self.args = lc0_autopilot._parse_args([
            "--corpus-manifest", str(self.manifest), "--out-root", str(self.out),
            "--initial-checkpoint", str(self.initial), "--initial-active-model", str(self.incumbent),
            "--source-commit", "a" * 40, "--max-chunks", "1", "--gate-games", "0",
            "--disk-reserve-gib", "0", "--device", "cpu",
        ])
        self.calls = []

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def trainer(self, **kwargs):
        self.calls.append(kwargs)
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.json").write_text(json.dumps({
            "arch": "v2", "hidden_dim": 1024, "input_dim": 40960,
            "chunk": len(self.calls),
        }))
        (out / "optimizer.pt").write_bytes(b"new-adam")
        metrics = {"initial_reference_val_loss": 0.7, "selected_reference_val_loss": 0.6 + len(self.calls) * 0.001}
        (out / "metrics.json").write_text(json.dumps(metrics))
        return metrics

    def export(self, checkpoint, *, quant_path):
        quant_path.write_bytes(b"PIENNQ02" + str(checkpoint.get("chunk", 0)).encode())
        return {"quant_format": "PIENNQ02"}

    def run_campaign(self, *, trainer=None, now=None):
        with mock.patch.object(lc0_autopilot, "_source_identity", return_value={"commit": "a" * 40, "digest": "code"}), \
             mock.patch.object(lc0_autopilot, "_train", side_effect=trainer or self.trainer), \
             mock.patch.object(lc0_autopilot.run_pipeline, "_export_v2_checkpoint", side_effect=self.export):
            return lc0_autopilot.run(self.args, **({"now": now} if now else {}))

    def state(self):
        return json.loads((self.out / "lc0_state.json").read_text())

    def test_first_chunk_new_optimizer_next_chunk_restores_latest(self):
        self.run_campaign()
        first = self.calls[-1]
        self.assertTrue(first["initial_checkpoint_weights_only"])
        self.assertIsNone(first["initial_optimizer_state"])
        self.assertEqual(first["checkpoint_selection"], "latest")
        self.assertEqual(first["target_mode"], "lc0-q-outcome")
        self.assertEqual(first["max_samples"], 700000)
        self.assertEqual(first["teacher_mix"], 0.8)
        self.assertEqual(first["cp_loss_weight"], 0)
        saved = self.state()["training_checkpoint_path"]
        self.run_campaign()
        second = self.calls[-1]
        self.assertFalse(second["initial_checkpoint_weights_only"])
        self.assertEqual(str(second["initial_checkpoint"]), saved)
        self.assertEqual(second["initial_optimizer_state"].name, "optimizer.pt")
        self.assertEqual(self.state()["completed_chunks"], 2)
        self.assertEqual(self.state()["active_model_path"], str(self.incumbent.resolve()))
        self.assertEqual(self.state()["baseline_validation"]["loss"], 0.7)

    def test_pass_visits_every_chunk_once_before_repeat_and_retains_best(self):
        self.args.max_chunks = 4
        self.run_campaign()
        ids = [item["chunk_index"] for item in self.state()["history"]]
        self.assertEqual(sorted(ids[:3]), [0, 1, 2])
        self.assertEqual(self.state()["pass_number"], 1)
        self.assertTrue(Path(self.state()["best_checkpoint_path"]).exists())
        self.assertTrue(self.initial.exists())
        self.assertTrue(self.incumbent.exists())
        self.assertEqual(len(list((self.out / "chunks").glob("*/train/checkpoint.json"))), 3)

    def test_unfinished_chunk_replayed_after_crash(self):
        with self.assertRaisesRegex(RuntimeError, "power loss"):
            self.run_campaign(trainer=mock.Mock(side_effect=RuntimeError("power loss")))
        pending = self.state()["in_progress"]["chunk_index"]
        self.assertEqual(self.state()["completed_chunks"], 0)
        self.run_campaign()
        self.assertEqual(self.state()["history"][0]["chunk_index"], pending)
        self.assertTrue(self.calls[0]["initial_checkpoint_weights_only"])

    def test_changed_corpus_manifest_or_bootstrap_refuses_resume(self):
        self.run_campaign()
        self.initial.write_text(self.initial.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign()

    def test_corrupt_chunk_and_incomplete_corpus_rejected(self):
        Path(self.chunks[0]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            lc0_autopilot.load_corpus(self.manifest)
        manifest = json.loads(self.manifest.read_text())
        manifest["complete"] = False
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "complete"):
            lc0_autopilot.load_corpus(self.manifest)

    def test_validation_copy_is_fixed_and_not_training_directory(self):
        self.run_campaign()
        args = self.calls[0]
        self.assertNotEqual(args["validation_jsonl_dir"], args["jsonl_dir"])
        files = list(args["validation_jsonl_dir"].glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_text(), '{"game_id":"holdout"}\n')

    def test_deadline_stops_before_next_chunk_and_stays_finished_on_restart(self):
        self.args.max_chunks = 10
        clock = [100.0]
        def expire(**kwargs):
            result = self.trainer(**kwargs)
            clock[0] = 100.0 + self.args.hours * 3600 + 1
            return result
        self.run_campaign(trainer=expire, now=lambda: clock[0])
        count = len(self.calls)
        self.assertLessEqual(count, 1)
        self.assertEqual(self.state()["status"], "complete")
        self.run_campaign(now=lambda: clock[0] + 1)
        self.assertEqual(len(self.calls), count)

    def test_committed_checkpoint_corruption_refuses_resume(self):
        self.run_campaign()
        Path(self.state()["training_optimizer_path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_campaign()

    def test_starting_weights_remain_validation_best_when_first_chunk_regresses(self):
        def regress(**kwargs):
            result = self.trainer(**kwargs)
            result["selected_reference_val_loss"] = 0.8
            return result
        self.run_campaign(trainer=regress)
        self.assertEqual(self.state()["best_validation_loss"], 0.7)
        self.assertEqual(self.state()["best_checkpoint_path"], str(self.initial.resolve()))
        self.assertNotEqual(self.state()["training_checkpoint_path"], str(self.initial.resolve()))

    def test_gate_failure_does_not_retrain_completed_chunk(self):
        self.args.max_chunks = 3
        self.args.gate_games = 400
        with mock.patch.object(lc0_autopilot.autopilot, "_run_confirmed_gate_attempt", side_effect=RuntimeError("gate failed")):
            with self.assertRaisesRegex(RuntimeError, "gate failed"):
                self.run_campaign()
        self.assertEqual(self.state()["completed_chunks"], 3)
        self.assertTrue(self.state()["evaluation_pending"])
        self.args.max_chunks = 1
        with mock.patch.object(lc0_autopilot.autopilot, "_run_confirmed_gate_attempt", return_value={"accepted": False}) as gate:
            self.run_campaign()
        self.assertEqual(gate.call_count, 1)
        self.assertEqual(self.state()["completed_chunks"], 4)

    def test_complete_chunk_manifest_recovers_after_state_write_failure(self):
        original = lc0_autopilot.autopilot._atomic_write_json
        failed = [False]
        def crash(path, obj):
            if path.name == "lc0_state.json" and obj.get("completed_chunks") == 1 and not failed[0]:
                failed[0] = True
                raise OSError("state disk failure")
            return original(path, obj)
        with mock.patch.object(lc0_autopilot.autopilot, "_atomic_write_json", side_effect=crash):
            with self.assertRaisesRegex(OSError, "state disk failure"):
                self.run_campaign()
        self.run_campaign(trainer=mock.Mock(side_effect=AssertionError("must reuse completed chunk")))
        self.assertEqual(self.state()["completed_chunks"], 1)
        self.assertEqual(len(self.calls), 1)

    def test_disabled_gate_never_promotes(self):
        self.args.max_chunks = 3
        self.run_campaign()
        self.assertEqual(self.state()["last_gate"]["reason"], "gate-disabled-promotion-ineligible")
        self.assertFalse(self.state()["accepted_models"])

    def test_gate_reuses_confirmed_same_search_policy(self):
        self.args.max_chunks = 3
        self.args.gate_games = 400
        accepted = {"accepted": True, "reason": "confirmation-accepted"}
        with mock.patch.object(lc0_autopilot.autopilot, "_run_confirmed_gate_attempt", return_value=accepted) as gate:
            self.run_campaign()
        kw = gate.call_args.kwargs
        self.assertEqual((kw["screen_games"], kw["confirmation_games"]), (400, 1000))
        self.assertEqual((kw["base_blend_percent"], kw["candidate_blend_percent"]), (75, 75))
        self.assertTrue(kw["paired_openings"])
        self.assertEqual(kw["threads"], 1)
        self.assertEqual(kw["confidence_level"], 0.95)
        self.assertEqual(kw["incremental_pst_policy"], "strict-superiority")
        self.assertNotEqual(self.state()["active_model_path"], str(self.incumbent))
        self.assertTrue(Path(self.state()["active_model_path"]).exists())

    def test_single_instance_lock(self):
        self.out.mkdir()
        with lc0_autopilot.autopilot._single_instance_lock(self.out / "lc0.lock"):
            with self.assertRaises((BlockingIOError, RuntimeError)):
                self.run_campaign()

    def test_disk_pressure_evicts_only_reproducible_training_cache(self):
        cache = self.out / "cache" / "training"
        cache.mkdir(parents=True)
        (cache / "train.jsonl").write_text("reproducible")
        saved = self.out / "accepted" / "saved.nnue"
        saved.parent.mkdir()
        saved.write_bytes(b"protected")
        with mock.patch.object(lc0_autopilot.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                lc0_autopilot._check_disk(self.out, 50)
        self.assertFalse(cache.exists())
        self.assertEqual(saved.read_bytes(), b"protected")
        self.assertTrue(self.initial.exists())

    def test_default_parallel_gate_uses_eight_single_thread_games(self):
        self.assertEqual(self.args.gate_parallel_games, 8)

    def test_default_30_day_budget_sets_one_immutable_deadline(self):
        self.assertEqual(self.args.hours, 720.0)
        self.run_campaign(now=lambda: 100.0)
        deadline = 100.0 + 720 * 3600
        self.assertEqual(self.state()['deadline_at'], deadline)
        self.run_campaign(now=lambda: 200.0)
        self.assertEqual(self.state()['deadline_at'], deadline)
        self.args.hours = 336.0
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_campaign(now=lambda: 300.0)

    def test_capacity_is_decimal_disabled_by_default_and_immutable_on_resume(self):
        self.assertEqual(self.args.disk_capacity_gb, 0)
        self.args.disk_capacity_gb = 478
        with mock.patch.object(lc0_autopilot.shutil, 'disk_usage',
                               return_value=SimpleNamespace(total=400_000_000_000, free=200_000_000_000)):
            self.run_campaign(now=lambda: 100.)
            self.assertEqual(self.state()['identity']['disk_capacity_bytes'], 478_000_000_000)
            self.args.disk_capacity_gb = 513
            with self.assertRaisesRegex(ValueError, 'identity'):
                self.run_campaign(now=lambda: 101.)

    def test_disk_ceiling_stops_before_training_even_with_filesystem_space(self):
        self.args.disk_capacity_gb = 478
        self.args.disk_reserve_gib = 50
        with mock.patch.object(lc0_autopilot.shutil, 'disk_usage',
                               return_value=SimpleNamespace(total=600_000_000_000, free=150_000_000_000)):
            with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                self.run_campaign()
        self.assertEqual(self.calls, [])
        self.assertTrue(self.initial.exists())
        self.assertTrue(self.incumbent.exists())

    def test_disk_ceiling_rechecked_after_chunk_expansion(self):
        self.args.disk_capacity_gb = 478
        self.args.disk_reserve_gib = 50
        usage = [SimpleNamespace(total=600_000_000_000, free=200_000_000_000)]
        expand = lc0_autopilot._expand_chunk
        def use_working_space(*args, **kwargs):
            result = expand(*args, **kwargs)
            usage[0] = SimpleNamespace(total=600_000_000_000, free=150_000_000_000)
            return result
        with mock.patch.object(lc0_autopilot.shutil, 'disk_usage', side_effect=lambda _: usage[0]), \
                mock.patch.object(lc0_autopilot, '_expand_chunk', side_effect=use_working_space):
            with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                self.run_campaign()
        self.assertEqual(self.calls, [])

    def test_invalid_capacity_rejected_before_state_creation(self):
        for amount in (-1, float('nan'), float('inf')):
            with self.subTest(amount=amount):
                self.args.disk_capacity_gb = amount
                with self.assertRaisesRegex(ValueError, 'disk capacity'):
                    self.run_campaign()
                self.assertFalse((self.out / 'lc0_state.json').exists())


if __name__ == "__main__":
    unittest.main()
