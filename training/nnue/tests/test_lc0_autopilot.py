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

    def test_learning_rate_decays_with_positions_trained_and_is_recorded(self):
        self.args.learning_rate = 0.001
        self.args.lr_gamma = 0.5
        self.args.lr_epoch_positions = 1
        self.args.max_chunks = 2
        self.run_campaign()
        # A restart must continue the schedule from committed history, not reset it.
        self.args.max_chunks = 1
        self.run_campaign()
        rates = [call["learning_rate"] for call in self.calls]
        self.assertEqual(len(rates), 3)
        for observed, expected in zip(rates, (0.001, 0.0005, 0.00025)):
            self.assertAlmostEqual(observed, expected, places=12)
        recorded = [item["learning_rate"] for item in self.state()["history"]]
        self.assertEqual(recorded, rates)

    def test_default_schedule_is_constant_and_keeps_the_founding_identity_shape(self):
        self.assertEqual(self.args.lr_gamma, 1.0)
        self.assertEqual(self.args.gate_movetime_ms, 150)
        self.args.max_chunks = 2
        self.run_campaign()
        self.assertEqual([call["learning_rate"] for call in self.calls], [0.001, 0.001])
        identity = self.state()["identity"]
        for key in ("lr_schedule", "gate_movetime_ms", "extra_corpora"):
            self.assertNotIn(key, identity)

    def test_changed_schedule_or_gate_time_refuses_resume(self):
        self.args.lr_gamma = 0.995
        self.run_campaign()
        self.assertEqual(self.state()["identity"]["lr_schedule"],
                         {"gamma": 0.995, "epoch_positions": 100_000_000})
        self.args.lr_gamma = 0.99
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign()
        self.args.lr_gamma = 0.995
        self.args.gate_movetime_ms = 1000
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign()

    def test_invalid_schedule_rejected_before_state_creation(self):
        for gamma, epoch in ((0.0, 100), (1.5, 100), (float("nan"), 100), (0.9, 0)):
            self.args.lr_gamma, self.args.lr_epoch_positions = gamma, epoch
            with self.assertRaisesRegex(ValueError, "schedule"):
                self.run_campaign()
        self.assertFalse((self.out / "lc0_state.json").exists())

    def test_gate_movetime_is_configurable(self):
        self.args.max_chunks = 3
        self.args.gate_games = 400
        self.args.gate_movetime_ms = 1000
        rejected = {"accepted": False, "reason": "screen-rejected"}
        with mock.patch.object(lc0_autopilot.autopilot, "_run_confirmed_gate_attempt", return_value=rejected) as gate:
            self.run_campaign()
        self.assertEqual(gate.call_args.kwargs["movetime_ms"], 1000)
        self.assertEqual(self.state()["identity"]["gate_movetime_ms"], 1000)

    def extra_corpus(self, *, since="2026-03-07T00:00:00+00:00", until="2026-07-07T00:00:00+00:00"):
        extra = self.root / "extra"
        extra.mkdir()
        chunks = []
        for i in range(2):
            path = extra / f"chunk-{i}.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps({"game_id": f"extra-{i}"}) + "\n")
            chunks.append({"path": str(path), "sha256": self.sha(path), "positions": 1})
        val = extra / "validation.jsonl"
        val.write_text('{"game_id":"extra-holdout"}\n')
        manifest = extra / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "piebot-lc0-corpus-v1", "complete": True, "corpus_id": "extra-corpus",
            "since": since, "until": until, "seed": 42, "chunks": chunks,
            "validation": {"path": str(val), "sha256": self.sha(val), "positions": 1},
        }))
        return manifest

    def test_extra_corpus_joins_traversal_and_primary_validation_stays_fixed(self):
        extra = self.extra_corpus()
        self.args.extra_corpus_manifest = [extra]
        self.args.max_chunks = 5
        self.run_campaign()
        state = self.state()
        self.assertEqual(sorted(item["chunk_index"] for item in state["history"]), [0, 1, 2, 3, 4])
        self.assertEqual(state["pass_number"], 1)
        self.assertEqual(state["identity"]["extra_corpora"],
                         [{"corpus_id": "extra-corpus", "manifest_sha256": self.sha(extra)}])
        primary_validation = json.loads(self.manifest.read_text())["validation"]["sha256"]
        self.assertEqual(state["identity"]["validation_sha256"], primary_validation)
        copied = self.calls[0]["validation_jsonl_dir"] / "validation.jsonl"
        self.assertEqual(self.sha(copied), primary_validation)

    def test_extra_corpus_overlapping_the_primary_window_is_rejected(self):
        # The fixed validation games are held out of the primary window only;
        # an overlapping extra window could train on them.
        self.args.extra_corpus_manifest = [self.extra_corpus(until="2026-07-08T00:00:00+00:00")]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_campaign()
        self.assertFalse((self.out / "lc0_state.json").exists())

    def write_game_corpus(self):
        """Replace chunk 0..2 with one chunk holding a short real game."""
        import chess
        board = chess.Board()
        fens = [board.fen()]
        for san in ("e4", "f5", "Qh5+", "g6", "Qxg6+"):
            board.push_san(san)
            fens.append(board.fen())
        path = self.corpus / "game.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for ply, fen in enumerate(fens):
                handle.write(json.dumps({"game_id": "g", "ply": ply, "fen": fen}) + "\n")
        manifest = json.loads(self.manifest.read_text())
        manifest["chunks"] = [{"path": str(path), "sha256": self.sha(path), "positions": len(fens)}]
        self.manifest.write_text(json.dumps(manifest))

    def test_position_filters_drop_rows_before_training_and_are_recorded(self):
        self.write_game_corpus()
        self.args.skip_early_plies = 1
        self.args.skip_in_check = True
        self.args.skip_before_capture = True
        seen = []
        def trainer(**kwargs):
            seen.append([json.loads(line)["ply"]
                         for line in (kwargs["jsonl_dir"] / "train.jsonl").read_text().splitlines()])
            return self.trainer(**kwargs)
        self.run_campaign(trainer=trainer)
        # ply 0 is early, ply 3 and 5 are in check, ply 4 precedes a capture.
        self.assertEqual(seen, [[1, 2]])
        state = self.state()
        self.assertEqual(state["identity"]["position_filter"],
                         {"skip_early_plies": 1, "skip_in_check": True, "skip_before_capture": True})
        self.assertEqual(state["history"][0]["positions"], 6)
        self.assertEqual(state["history"][0]["positions_trained"], 2)

    def test_learning_rate_schedule_counts_positions_actually_trained(self):
        self.write_game_corpus()
        self.args.skip_early_plies = 4
        self.args.lr_gamma = 0.5
        self.args.lr_epoch_positions = 2
        self.args.max_chunks = 2
        self.run_campaign()
        # Two of six rows survive the filter, so the second chunk is one epoch on.
        self.assertEqual([call["learning_rate"] for call in self.calls], [0.001, 0.0005])

    def test_unfiltered_lineage_keeps_identity_and_teacher_mix_is_a_founding_option(self):
        self.run_campaign()
        identity = self.state()["identity"]
        self.assertNotIn("position_filter", identity)
        self.assertEqual(identity["teacher_mix"], 0.8)
        self.assertEqual(self.calls[0]["teacher_mix"], 0.8)
        self.assertNotIn("positions_trained", self.state()["history"][0])
        self.args.teacher_mix = 0.7
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign()

    def test_teacher_mix_is_forwarded_and_validated(self):
        self.args.teacher_mix = 0.7
        self.run_campaign()
        self.assertEqual(self.calls[0]["teacher_mix"], 0.7)
        self.assertEqual(self.state()["identity"]["teacher_mix"], 0.7)

    def test_invalid_teacher_mix_or_filter_rejected_before_state_creation(self):
        for name, value in (("teacher_mix", 1.5), ("teacher_mix", float("nan")), ("skip_early_plies", -1)):
            args = lc0_autopilot._parse_args([
                "--corpus-manifest", str(self.manifest), "--out-root", str(self.out),
                "--initial-checkpoint", str(self.initial), "--initial-active-model", str(self.incumbent),
                "--source-commit", "a" * 40, "--max-chunks", "1", "--gate-games", "0",
                "--disk-reserve-gib", "0", "--device", "cpu"])
            setattr(args, name, value)
            self.args = args
            with self.assertRaises(ValueError):
                self.run_campaign()
        self.assertFalse((self.out / "lc0_state.json").exists())

    def test_absolute_deadline_overrides_hours_and_is_part_of_the_identity(self):
        # Acquisition time is unknown at launch, so a lineage that must end at a
        # fixed instant names it instead of guessing an hour budget.
        self.args.deadline_utc = "2026-12-07T18:57:28+00:00"
        expected = 1796669848.0
        self.run_campaign(now=lambda: 1791000000.0)
        self.assertEqual(self.state()["deadline_at"], expected)
        self.assertEqual(self.state()["identity"]["deadline_utc"], "2026-12-07T18:57:28+00:00")
        self.run_campaign(now=lambda: 1791000500.0)
        self.assertEqual(self.state()["deadline_at"], expected)
        self.args.deadline_utc = "2026-12-08T18:57:28+00:00"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign(now=lambda: 1791001000.0)

    def test_deadline_must_be_a_future_timestamp_with_a_zone(self):
        for value in ("2026-12-07T18:57:28", "not a time", "2020-01-01T00:00:00+00:00"):
            self.args.deadline_utc = value
            with self.assertRaisesRegex(ValueError, "deadline"):
                self.run_campaign(now=lambda: 1791000000.0)
        self.assertFalse((self.out / "lc0_state.json").exists())

    def test_no_deadline_option_keeps_the_hour_budget_identity(self):
        self.run_campaign(now=lambda: 100.0)
        self.assertNotIn("deadline_utc", self.state()["identity"])
        self.assertEqual(self.state()["deadline_at"], 100.0 + 720 * 3600)

    def test_initial_cursor_continues_another_lineages_traversal(self):
        # A lineage founded to replace a stopped one (same corpus, same seed)
        # starts in the shared chunk order where that one stopped.
        order = lc0_autopilot._order(3, self.args.seed, 0)
        self.args.initial_cursor = 2
        self.args.max_chunks = 2
        self.run_campaign()
        history = self.state()["history"]
        self.assertEqual((history[0]["pass_number"], history[0]["cursor"]), (0, 2))
        self.assertEqual(history[0]["chunk_index"], order[2])
        self.assertEqual((history[1]["pass_number"], history[1]["cursor"]), (1, 0))
        self.assertEqual(self.state()["identity"]["initial_cursor"], 2)
        self.args.initial_cursor = 1
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_campaign()

    def test_default_start_keeps_the_founding_identity_and_bad_cursor_is_rejected(self):
        for bad in (-1, 3):
            self.args.initial_cursor = bad
            with self.assertRaisesRegex(ValueError, "initial cursor"):
                self.run_campaign()
            self.assertFalse((self.out / "lc0_state.json").exists())
        self.args.initial_cursor = 0
        self.run_campaign()
        self.assertNotIn("initial_cursor", self.state()["identity"])
        self.assertEqual(self.state()["history"][0]["cursor"], 0)


if __name__ == "__main__":
    unittest.main()
