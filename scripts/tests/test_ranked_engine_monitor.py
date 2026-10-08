import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import ranked_engine_monitor as monitor


class RankedEngineMonitorTests(unittest.TestCase):
    def test_due_keeps_baseline_then_measures_changed_models_daily(self):
        self.assertTrue(monitor.measurement_due({}, 'initial', 0))
        previous = {'last_completed_at': 100, 'last_model_sha256': 'old'}
        self.assertFalse(monitor.measurement_due(previous, 'new', 101))
        self.assertFalse(monitor.measurement_due(previous, 'old', 100000))
        self.assertTrue(monitor.measurement_due(previous, 'new', 100000))
        self.assertTrue(monitor.measurement_due(previous, 'new', 101, ended=True))

    def test_daily_cadence_counts_from_start_of_previous_match(self):
        previous = {'last_started_at': 1, 'last_completed_at': 100,
                    'last_model_sha256': 'old'}
        self.assertTrue(monitor.measurement_due(previous, 'new', 86401))

    def test_expired_deadline_waits_for_final_chunk_and_measures_its_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            piebot, old, new, book = [root / name for name in ('uci', 'old.nnue', 'new.nnue', 'book')]
            for path in (piebot, old, new, book):
                path.write_bytes(path.name.encode())
            old_sha, new_sha = monitor.sha256(old), monitor.sha256(new)
            engine_root = root / 'engines' / monitor.COHORT[0]
            engine_root.mkdir(parents=True)
            (engine_root / 'manifest.json').write_text('{}')
            campaign = root / 'campaign' / 'training' / 'lc0_state.json'
            campaign.parent.mkdir(parents=True)
            campaign.write_text(json.dumps({'status': 'running', 'deadline_at': 0,
                                            'best_quant_path': str(old), 'best_quant_sha256': old_sha}))
            engine = {'id': monitor.COHORT[0], 'name': 'Carp', 'rating': 3524,
                      'rating_source': 'https://example.test', 'rating_snapshot': '2026-09-05',
                      'executable': str(piebot), 'executable_sha256': monitor.sha256(piebot),
                      'cwd': str(root)}
            measured = []

            def run(command, cwd, log):
                model = Path(command[command.index('--piebot-nnue') + 1])
                measured.append(monitor.sha256(model))
                results = Path(command[command.index('--results') + 1])
                results.write_text(json.dumps({'summary': {'games': 2, 'complete_pairs': 1,
                                                          'measurement_eligible': False}}))

            def publish_final_chunk(_seconds):
                campaign.write_text(json.dumps({'status': 'complete', 'deadline_at': 0,
                                                'best_quant_path': str(new), 'best_quant_sha256': new_sha}))

            with mock.patch.object(monitor, 'verify_engine', return_value=engine), \
                 mock.patch.object(monitor, 'run_arena', side_effect=run), \
                 mock.patch.object(monitor.time, 'sleep', side_effect=publish_final_chunk):
                self.assertEqual(monitor.main([
                    '--piebot-bin', str(piebot), '--engine-root', str(root / 'engines'),
                    '--engines', monitor.COHORT[0], '--campaign-root', str(root / 'campaign'),
                    '--out-root', str(root / 'out'), '--baseline', str(old),
                    '--baseline-sha256', old_sha, '--book', str(book), '--games', '2']), 0)
            self.assertEqual(measured, [old_sha, new_sha])

    def test_snapshot_does_not_overwrite_an_existing_measurement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'candidate.nnue'
            source.write_bytes(b'weights')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            target = monitor.snapshot_model(source, expected, root / 'job')
            source.unlink()
            self.assertEqual(monitor.snapshot_model(source, expected, root / 'job'), target)
            target.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                monitor.snapshot_model(source, expected, root / 'job')

    def test_arena_command_uses_full_strength_and_pins_all_dependencies(self):
        engine = {'id': 'test-1', 'name': 'Test 1', 'rating': 3508, 'rank': 100,
                  'rating_source': 'https://example.test/rankings',
                  'rating_snapshot': '2026-09-05', 'executable': '/engines/test',
                  'executable_sha256': 'a' * 64,
                  'auxiliary_files': [{'path': '/engines/net.nnue', 'sha256': 'b' * 64}],
                  'qualification': {'single_thread_qualified': True}}
        command = monitor.arena_command(Path('/eval'), Path('/piebot'),
                                        Path('/model.nnue'), engine, Path('/book.fen'),
                                        Path('/job/result.json'), games=400, seed=42)
        for key, expected in (('--games', '400'), ('--time-control', '120+1'),
                              ('--piebot-blend', '100'), ('--opponent-sha256', 'a' * 64),
                              ('--opponent-asset', '/engines/net.nnue'),
                              ('--opponent-asset-sha256', 'b' * 64),
                              ('--openings-file', '/book.fen')):
            self.assertEqual(command[command.index(key) + 1], expected)
        self.assertNotIn('--stockfish-elo', command)
        self.assertIn('--opponent-single-thread-qualified', command)

    def test_incomplete_or_ineligible_arena_cannot_support_boundary_claim(self):
        healthy = {'summary': {'games': 400, 'complete_pairs': 200,
                              'measurement_eligible': True, 'score_95_ci': [0.51, 0.61]}}
        report = monitor.cohort_report({key: healthy for key in monitor.COHORT}, 400)
        self.assertTrue(report['local_boundary_supported'])
        failed = {'summary': {**healthy['summary'], 'measurement_eligible': False}}
        self.assertFalse(monitor.cohort_report({'carp': healthy, 'lambergar': failed}, 400)
                         ['local_boundary_supported'])
        incomplete = {'summary': {**healthy['summary'], 'games': 399, 'complete_pairs': 199}}
        self.assertFalse(monitor.cohort_report({'carp': incomplete}, 400)
                         ['local_boundary_supported'])
        self.assertFalse(monitor.cohort_report({}, 400)['local_boundary_supported'])
        self.assertFalse(monitor.cohort_report({'an-unrelated-weaker-engine': healthy}, 400)
                         ['local_boundary_supported'])

    def test_resuming_pending_job_does_not_switch_to_newer_candidate(self):
        previous = {'pending': {'model_sha256': 'old', 'model_source': '/old',
                                'directory': '/job'}}
        candidate = {'best_quant_sha256': 'new', 'best_quant_path': '/new'}
        self.assertEqual(monitor.choose_candidate(previous, candidate, Path('/baseline'), 'base'),
                         (Path('/old'), 'old'))

    def test_unlisted_runtime_files_invalidate_the_installed_opponent(self):
        from scripts import fetch_ranked_engines as fetch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            binary = root / 'engine'
            binary.write_bytes(b'engine')
            binary.chmod(0o755)
            files = fetch.file_inventory(root)
            manifest = {'schema': fetch.MANIFEST_SCHEMA, 'executable_relative': 'engine',
                        'executable': str(binary), 'executable_sha256': monitor.sha256(binary),
                        'cwd': str(root), 'files': files,
                        'files_identity_sha256': fetch.canonical_sha256(files)}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(monitor.verify_engine(root), manifest)
            (root / 'unexpected-network.nnue').write_bytes(b'network')
            with self.assertRaises(ValueError):
                monitor.verify_engine(root)


if __name__ == '__main__':
    unittest.main()
