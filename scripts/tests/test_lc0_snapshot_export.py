"""Model snapshots bind one committed LC0 state despite concurrent retention."""

import hashlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest import mock


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.campaign = self.root / 'lc0'
        self.bootstrap = self.root / 'selfplay'
        self.campaign.mkdir()
        self.bootstrap.mkdir()
        self.initial = self.write(self.bootstrap / 'initial.json', {'initial': True})
        self.incumbent = self.write(self.bootstrap / 'initial.nnue', b'PIENNQ02-initial')
        self.pin = self.write(self.campaign / 'source_git_commit', b'a' * 40 + b'\n')
        self.raw = self.write(self.campaign / 'data/raw/manifest.json', {
            'schema': 'piebot-lc0-raw-v1', 'since': '2026-07-07T00:00:00+00:00',
            'until': '2026-09-07T19:39:28.530592+00:00', 'files': []})
        self.identity = self.write(self.campaign / 'data/corpus/identity.json', {'corpus_id': 'frozen'})
        self.validation = self.write(self.campaign / 'data/corpus/validation.jsonl', b'{"fixture":true}\n')
        self.corpus = self.write(self.campaign / 'data/corpus/corpus_manifest.json', {
            'schema': 'piebot-lc0-corpus-v1', 'complete': True, 'corpus_id': 'frozen',
            'raw_manifest_sha256': self.sha(self.raw),
            'chunks': [{'path': str(self.campaign / 'data/corpus/omitted.jsonl.gz'),
                        'sha256': '1' * 64, 'positions': 1}],
            'validation': {'path': str(self.validation), 'sha256': self.sha(self.validation), 'positions': 1}})
        self.best = self.chunk(0)
        self.latest = self.chunk(1)
        self.state = {'schema': 'piebot-lc0-autopilot-v1',
            'identity': {'source': {'commit': 'a' * 40, 'digest': 'b' * 64},
                'initial_checkpoint': {'path': str(self.initial), 'sha256': self.sha(self.initial)},
                'initial_active_model': {'path': str(self.incumbent), 'sha256': self.sha(self.incumbent)},
                'manifest_sha256': self.sha(self.corpus), 'validation_sha256': self.sha(self.validation),
                'corpus_id': 'frozen', 'arch': 'v2', 'hidden_dim': 1024, 'hours': 336},
            'completed_chunks': 2, 'history': [self.best, self.latest],
            'training_checkpoint_path': str(self.path(self.latest, 'train/checkpoint.json')),
            'training_optimizer_path': str(self.path(self.latest, 'train/optimizer.pt')),
            'best_checkpoint_path': str(self.path(self.best, 'train/checkpoint.json')),
            'best_checkpoint_sha256': self.best['checkpoint_sha256'],
            'best_quant_path': str(self.path(self.best, 'candidate.nnue')),
            'best_quant_sha256': self.best['quant_sha256'],
            'active_model_path': str(self.incumbent), 'active_model_sha256': self.sha(self.incumbent),
            'accepted_models': [], 'last_gate': None, 'in_progress': {'number': 2},
            'started_at': 100, 'deadline_at': 1209700}
        self.state_path = self.write(self.campaign / 'training/lc0_state.json', self.state)
        self.write(self.campaign / 'training/chunks/chunk_00000002/train/checkpoint.json', b'UNCOMMITTED')

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, sort_keys=True).encode() if isinstance(value, dict) else value)
        return path

    def sha(self, path):
        return digest(path.read_bytes())

    def path(self, record, suffix):
        return Path(record['directory']) / suffix

    def chunk(self, number):
        directory = self.campaign / 'training/chunks' / f'chunk_{number:08}'
        adam = self.write(directory / 'train/optimizer.pt', f'Adam {number}'.encode())
        checkpoint = self.write(directory / 'train/checkpoint.json', {
            'weights': number, 'optimizer_state': {'sha256': self.sha(adam)}})
        quant = self.write(directory / 'candidate.nnue', f'PIENNQ02-{number}'.encode())
        self.write(directory / 'train/metrics.json', {'optimizer_state': {'sha256': self.sha(adam)}})
        record = {'number': number, 'directory': str(directory), 'positions': 1,
                  'checkpoint_sha256': self.sha(checkpoint), 'optimizer_sha256': self.sha(adam),
                  'quant_sha256': self.sha(quant), 'finished_at': number + 100}
        self.write(directory / 'complete.json', record)
        return record

    def exporter(self):
        return importlib.import_module('scripts.lc0_snapshot_export')

    def capture(self, **kwargs):
        return self.exporter().capture_plan(campaign_root=self.campaign, bootstrap_root=self.bootstrap,
                                           max_bytes=10_000_000, max_files=100, **kwargs)

    def test_complete_capture_contains_only_committed_recovery_files(self):
        exact_state = self.state_path.read_bytes()
        with self.capture() as plan:
            self.assertFalse(plan.manifest['capabilities']['strict_campaign_resume'])
            self.assertEqual(plan.manifest['state_sha256'], digest(exact_state))
            paths = {source['path'] for entry in plan.manifest['files'] for source in entry['sources']}
            self.assertIn(str(self.initial), paths)
            self.assertIn(str(self.incumbent), paths)
            self.assertIn(str(self.path(self.best, 'train/optimizer.pt')), paths)
            self.assertIn(str(self.path(self.latest, 'train/optimizer.pt')), paths)
            self.assertIn(str(self.corpus), paths)
            self.assertFalse(any('chunk_00000002' in path or path.endswith('.jsonl.gz') for path in paths))
            self.assertFalse((self.campaign / 'training/lc0.lock').exists())
            for entry in plan.manifest['files']:
                output = io.BytesIO()
                plan.copy_file(entry['sources'][0]['path'], output)
                self.assertEqual(digest(output.getvalue()), entry['sha256'])

    def reuse_another_lineages_corpus(self):
        """Move the corpus out of the campaign, as a lineage founded on reused corpora has it."""
        donor = self.root / 'donor'
        donor.mkdir()
        (self.campaign / 'data').rename(donor / 'data')
        validation = donor / 'data/corpus/validation.jsonl'
        manifest = json.loads((donor / 'data/corpus/corpus_manifest.json').read_text())
        manifest['validation']['path'] = str(validation)
        self.corpus = self.write(donor / 'data/corpus/corpus_manifest.json', manifest)
        self.state['identity']['manifest_sha256'] = self.sha(self.corpus)
        self.write(self.state_path, self.state)
        return donor

    def test_corpus_may_live_in_a_separately_approved_root(self):
        donor = self.reuse_another_lineages_corpus()
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture()
        with self.capture(corpus_root=donor) as plan:
            by_role = {source['role']: source['path'] for entry in plan.manifest['files']
                       for source in entry['sources']}
            self.assertEqual(plan.manifest['corpus_root'], str(donor))
        self.assertEqual(by_role['corpus_manifest'], str(donor / 'data/corpus/corpus_manifest.json'))
        self.assertEqual(by_role['raw_manifest'], str(donor / 'data/raw/manifest.json'))
        self.assertEqual(by_role['corpus_identity'], str(donor / 'data/corpus/identity.json'))
        self.assertEqual(by_role['fixed_validation'], str(donor / 'data/corpus/validation.jsonl'))
        self.assertTrue(by_role['latest_checkpoint'].startswith(str(self.campaign)))

    def test_default_capture_names_no_corpus_root_and_cli_forwards_one(self):
        with self.capture() as plan:
            self.assertNotIn('corpus_root', plan.manifest)
        donor = self.reuse_another_lineages_corpus()
        script = Path(self.exporter().__file__)
        result = subprocess.run([sys.executable, str(script), '--campaign-root', str(self.campaign),
                                 '--bootstrap-root', str(self.bootstrap), '--corpus-root', str(donor)],
                                cwd=self.root, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
            self.assertEqual(archive.getnames()[-1], 'complete.json')

    def test_open_descriptors_survive_retention_and_state_advance(self):
        state_a = self.state_path.read_bytes()
        expected = self.path(self.best, 'train/checkpoint.json').read_bytes()
        def retained(plan):
            self.path(self.best, 'train/checkpoint.json').unlink()
            newer = dict(self.state, completed_chunks=3)
            replacement = self.write(self.campaign / 'new-state.json', newer)
            replacement.replace(self.state_path)
        with self.capture(after_open=retained) as plan:
            self.assertEqual(plan.manifest['state_sha256'], digest(state_a))
            output = io.BytesIO()
            plan.copy_file(str(self.path(self.best, 'train/checkpoint.json')), output)
            self.assertEqual(output.getvalue(), expected)

    def test_missing_retained_artifact_after_state_read_aborts_attempt(self):
        def remove(_):
            self.path(self.best, 'train/optimizer.pt').unlink()
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture(after_state_read=remove)

    def test_mutated_pinned_artifact_refuses_instead_of_mixing(self):
        def mutate(_):
            self.path(self.latest, 'train/optimizer.pt').write_bytes(b'wrong!')
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture(after_open=mutate)

    def test_changed_bytes_during_copy_fail_and_close_remains_available(self):
        with self.capture() as plan:
            path = self.path(self.latest, 'candidate.nnue')
            path.write_bytes(b'changed bytes')
            with self.assertRaises(self.exporter().SnapshotError):
                plan.copy_file(str(path), io.BytesIO())

    def test_symlink_escape_and_directory_aliases_refuse(self):
        victim = self.path(self.best, 'train/optimizer.pt')
        original = victim.read_bytes()
        external = self.write(self.root / 'outside.pt', original)
        victim.unlink()
        victim.symlink_to(external)
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture()
        victim.unlink()
        victim.write_bytes(original)
        training = self.campaign / 'training'
        moved = self.root / 'moved-training'
        training.rename(moved)
        training.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture()

    def test_changed_manifest_source_pin_or_completion_binding_refuses(self):
        for path in (self.corpus, self.pin, self.path(self.best, 'complete.json')):
            original = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(original + b'broken')
                with self.assertRaises(self.exporter().SnapshotError):
                    self.capture()
                path.write_bytes(original)

    def test_bound_optimizer_metadata_mismatch_refuses(self):
        metrics = self.path(self.latest, 'train/metrics.json')
        self.write(metrics, {'optimizer_state': {'sha256': '0' * 64}})
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture()

    def test_baseline_best_needs_no_invented_optimizer(self):
        self.state.update(best_checkpoint_path=str(self.initial), best_checkpoint_sha256=self.sha(self.initial),
                          best_quant_path=str(self.incumbent), best_quant_sha256=self.sha(self.incumbent))
        self.write(self.state_path, self.state)
        for suffix in ('train/checkpoint.json', 'train/optimizer.pt', 'candidate.nnue'):
            self.path(self.best, suffix).unlink()
        with self.capture() as plan:
            self.assertFalse(any('chunk_00000000' in source['path']
                                 for row in plan.manifest['files'] for source in row['sources']))

    def test_capture_size_and_file_caps_refuse_before_payload_stream(self):
        exporter = self.exporter()
        for limit in ({'max_bytes': 1, 'max_files': 100}, {'max_bytes': 10_000_000, 'max_files': 1}):
            with self.subTest(limit=limit), self.assertRaises(exporter.SnapshotError):
                exporter.capture_plan(campaign_root=self.campaign, bootstrap_root=self.bootstrap, **limit)

    def test_ustar_stream_contract_deduplicates_and_finishes_with_verified_marker(self):
        archive = io.BytesIO()
        with self.capture() as plan:
            self.exporter().write_tar(plan, archive)
            expected_state = plan.state_bytes
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:') as tar:
            members = tar.getmembers()
            self.assertTrue(all(item.isfile() and not item.pax_headers for item in members))
            self.assertEqual(members[0].name, 'snapshot.json')
            self.assertEqual(members[-1].name, 'complete.json')
            header = tar.extractfile(members[0]).read()
            snapshot = json.loads(header)
            names = [row['member'] for row in snapshot['files']]
            self.assertEqual(names, sorted(set(names)))
            self.assertEqual([item.name for item in members[1:-1]], names)
            sources = [source for row in snapshot['files'] for source in row['sources']]
            self.assertEqual(len({source['role'] for source in sources}), len(sources))
            self.assertTrue(snapshot['capabilities']['model_recovery'])
            self.assertFalse(snapshot['capabilities']['strict_campaign_resume'])
            for row in snapshot['files']:
                payload = tar.extractfile(row['member']).read()
                self.assertEqual(digest(payload), row['sha256'])
                self.assertEqual(len(payload), row['size'])
                if any(source['role'] == 'state' for source in row['sources']):
                    self.assertEqual(payload, expected_state)
            complete = json.loads(tar.extractfile(members[-1]).read())
            self.assertEqual(complete['snapshot_sha256'], digest(header))
            self.assertEqual(complete['blob_count'], len(snapshot['files']))
            self.assertEqual(complete['total_blob_bytes'], sum(row['size'] for row in snapshot['files']))
            self.assertEqual(complete['schema'], 'piebot-lc0-model-snapshot-complete-v1')

    def test_mutated_alias_prevents_completion_marker(self):
        alias = self.write(self.campaign / 'training/accepted/alias.nnue', self.incumbent.read_bytes())
        self.state['active_model_path'] = str(alias)
        self.write(self.state_path, self.state)
        archive = io.BytesIO()
        with self.capture() as plan:
            def mutate(_):
                alias.write_bytes(b'wrong alias bytes')
            with self.assertRaises(self.exporter().SnapshotError):
                self.exporter().write_tar(plan, archive, before_complete=mutate)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:') as tar:
            self.assertNotIn('complete.json', tar.getnames())

    def test_probe_is_small_and_waits_for_first_committed_chunk(self):
        exporter = self.exporter()
        probe = exporter.probe(self.campaign)
        self.assertTrue(probe['ready'])
        key = probe['model_identity_key']
        self.assertLess(len(json.dumps(probe)), 8000)
        self.state['in_progress'] = {'number': 9}
        self.write(self.state_path, self.state)
        self.assertEqual(exporter.probe(self.campaign)['model_identity_key'], key)
        self.state.update(completed_chunks=0, history=[], training_checkpoint_path=None, training_optimizer_path=None)
        self.write(self.state_path, self.state)
        self.assertFalse(exporter.probe(self.campaign)['ready'])
        self.state_path.unlink()
        self.assertFalse(exporter.probe(self.campaign)['ready'])

    def test_default_capture_cap_is_five_gib(self):
        exporter = self.exporter()
        with mock.patch.object(exporter, 'capture_plan') as capture, \
             mock.patch.object(exporter, 'write_tar'):
            self.assertEqual(exporter.main(['--campaign-root', str(self.campaign),
                                            '--bootstrap-root', str(self.bootstrap)]), 0)
            self.assertEqual(capture.call_args.kwargs['max_bytes'], 5 * 1024**3)

    def test_failed_capture_closes_every_retained_descriptor(self):
        retained = []
        def break_file(plan):
            retained.extend(file.handle for file in plan.files.values())
            self.path(self.latest, 'candidate.nnue').write_bytes(b'wrong')
        with self.assertRaises(self.exporter().SnapshotError):
            self.capture(after_open=break_file)
        self.assertTrue(retained)
        self.assertTrue(all(handle.closed for handle in retained))

    def test_standalone_cli_probe_and_tar_have_no_repo_import_dependency(self):
        exporter = self.exporter()
        base = [sys.executable, '-I', exporter.__file__, '--campaign-root', str(self.campaign)]
        status = subprocess.run(base + ['--probe'], cwd=self.root, capture_output=True, check=True)
        self.assertTrue(json.loads(status.stdout)['ready'])
        self.assertEqual(status.stderr, b'')
        result = subprocess.run(base + ['--bootstrap-root', str(self.bootstrap)], cwd=self.root,
                                capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as tar:
            self.assertEqual(tar.getnames()[-1], 'complete.json')
        self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
