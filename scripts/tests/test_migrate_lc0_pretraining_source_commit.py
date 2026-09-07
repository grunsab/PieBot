"""The LC0 migration boundary exists only before the training clock starts."""

import fcntl
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import migrate_vast_source_commit as migration
from scripts.tests.test_migrate_vast_source_commit import _git, _sha256


class Lc0PretrainingMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        _git(self.repo, 'init', '-q')
        _git(self.repo, 'config', 'user.name', 'Tests')
        _git(self.repo, 'config', 'user.email', 'tests@example.invalid')
        source = self.repo / 'trainer.py'
        source.write_text('old\n')
        _git(self.repo, 'add', 'trainer.py')
        _git(self.repo, 'commit', '-qm', 'old')
        self.old = _git(self.repo, 'rev-parse', 'HEAD')
        source.write_text('new\n')
        _git(self.repo, 'commit', '-qam', 'new')
        self.new = _git(self.repo, 'rev-parse', 'HEAD')
        self.out = self.root / 'lc0'
        self.out.mkdir()
        self.lock = self.out / 'launcher.lock'
        self.lock.write_text('stopped-pid\n')
        self.pin = self.out / 'source_git_commit'
        self.pin.write_text(self.old + '\n')
        self.bootstrap = self.root / 'selfplay'
        self.bootstrap.mkdir()
        self.checkpoint = self.bootstrap / 'checkpoint.json'
        self.checkpoint.write_text('{"weights":"unchanged"}\n')
        self.active = self.bootstrap / 'active.nnue'
        self.active.write_bytes(b'PIENNQ02-original')
        self.raw = self.out / 'data/raw/manifest.json'
        self.raw.parent.mkdir(parents=True)
        archive = self.raw.parent / 'test91/a.tar'
        archive.parent.mkdir()
        archive.write_bytes(b'archive fixture')
        self.raw.write_text(json.dumps({
            'schema': 'piebot-lc0-raw-v1', 'since': '2026-07-07T00:00:00+00:00',
            'until': '2026-09-07T19:39:28.530592+00:00', 'complete': False,
            'files': [{'url': 'https://data.lczero.org/test91/a.tar', 'suite': 'test91',
                       'size': archive.stat().st_size, 'sha256': _sha256(archive),
                       'dest': str(archive), 'status': 'downloaded'}], 'failures': []
        }, indent=2) + '\n')

    def migrate(self, **overrides):
        kwargs = dict(repo_root=self.repo, out_root=self.out,
                      expected_old_commit=self.old, expected_new_commit=self.new,
                      bootstrap_root=self.bootstrap, bootstrap_checkpoint=self.checkpoint,
                      expected_bootstrap_sha256=_sha256(self.checkpoint),
                      active_model=self.active, expected_active_sha256=_sha256(self.active),
                      supervisor_stop_verified=True)
        kwargs.update(overrides)
        return migration.migrate_lc0_pretraining_source_commit(**kwargs)

    def add_corpus(self):
        corpus = self.out / 'data/corpus'
        corpus.mkdir()
        raw = json.loads(self.raw.read_text())
        config = {'schema': 'piebot-lc0-corpus-v1', 'since': raw['since'], 'until': raw['until'],
                  'game_identity': 'sha256_decompressed_v6_game_v1',
                  'source_alias_policy': 'immutable_content_v1', 'chunk_positions': 700000,
                  'validation_fraction': .01, 'validation_samples': 100000, 'seed': 20260907,
                  'sources': [{key: row[key] for key in ('url', 'sha256', 'size', 'suite')}
                              for row in raw['files']]}
        corpus_id = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        (corpus / 'identity.json').write_text(json.dumps(dict(config, corpus_id=corpus_id)))
        (corpus / 'prepare.lock').write_text('')
        (corpus / 'progress.sqlite3').write_bytes(b'opaque committed sqlite fixture')
        archive = corpus / 'archives/000000_fixture'
        archive.mkdir(parents=True)
        chunk = archive / 'train_000000.jsonl.gz'
        chunk.write_bytes(b'compressed payload commitment fixture')
        (archive / 'archive.json').write_text(json.dumps({
            'chunks': [{'name': chunk.name, 'positions': 1, 'sha256': _sha256(chunk)}],
            'holdout': [], 'stats': {'training_positions': 1}}))
        validation = corpus / 'validation.jsonl'
        validation.write_text('{"fen":"fixture"}\n')
        (corpus / 'corpus_manifest.json').write_text(json.dumps({
            'schema': 'piebot-lc0-corpus-v1', 'complete': True, 'corpus_id': corpus_id,
            'since': raw['since'], 'until': raw['until'], 'raw_manifest_sha256': _sha256(self.raw),
            'chunks': [{'path': str(chunk), 'positions': 1, 'sha256': _sha256(chunk)}],
            'validation': {'path': str(validation), 'positions': 1, 'sha256': _sha256(validation)}}))
        return corpus

    def test_prepared_audit_preserves_manifest_bootstrap_and_unstarted_clock(self):
        corpus = self.add_corpus()
        original = {path: path.read_bytes() for path in self.out.rglob('*') if path.is_file()}
        original.update({self.checkpoint: self.checkpoint.read_bytes(), self.active: self.active.read_bytes()})
        observed = {}
        def before(audit_path, pin):
            observed.update(json.loads(audit_path.read_text()))
            self.assertEqual(pin.read_text(), self.old + '\n')
        result = self.migrate(before_pin_replace=before)
        self.assertEqual(result.status, 'migrated')
        self.assertEqual(self.pin.read_text(), self.new + '\n')
        self.assertEqual(observed['mode'], 'lc0-pretraining')
        self.assertEqual(observed['lock']['path'], str(self.lock))
        self.assertTrue(observed['supervisor_stop']['externally_verified_by_caller'])
        self.assertFalse(observed['supervisor_stop']['verified_by_tool'])
        self.assertEqual(observed['bootstrap_checkpoint']['sha256'], _sha256(self.checkpoint))
        self.assertEqual(observed['active_model']['sha256'], _sha256(self.active))
        self.assertEqual(observed['raw_manifest']['sha256'], _sha256(self.raw))
        self.assertEqual(observed['raw_manifest']['until'], '2026-09-07T19:39:28.530592+00:00')
        self.assertEqual(observed['training']['required_budget_hours'], 336)
        self.assertFalse(observed['training']['live_budget_verified_by_tool'])
        self.assertFalse(observed['training']['clock_started'])
        self.assertIn('trainer.py', observed['source_delta']['name_status'])
        self.assertEqual(observed['corpus']['files']['identity.json']['sha256'], _sha256(corpus / 'identity.json'))
        self.assertFalse(observed['corpus']['payloads_rehashed'])
        self.assertEqual(len(observed['corpus']['chunk_commitments']), 1)
        for path, content in original.items():
            if path != self.pin:
                self.assertEqual(path.read_bytes(), content, str(path))
        self.assertFalse((self.out / 'training').exists())
        self.assertFalse((self.out / 'autopilot.lock').exists())
        self.assertFalse((self.out / 'autopilot_state.json').exists())

    def test_requires_external_stop_attestation_and_real_launcher_lock(self):
        with self.assertRaisesRegex(migration.MigrationError, 'externally'):
            self.migrate(supervisor_stop_verified=False)
        with self.lock.open('r+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(migration.MigrationError, 'running|held'):
                self.migrate()
        self.lock.unlink()
        with self.assertRaisesRegex(migration.MigrationError, 'lock.*missing'):
            self.migrate()
        self.assertFalse(self.lock.exists())

    def test_acquisition_without_corpus_preserves_queued_archive_and_frozen_cutoff(self):
        raw = json.loads(self.raw.read_text())
        raw['files'].append({'url': 'https://data.lczero.org/test91/pending.tar', 'suite': 'test91',
                             'size': 1000, 'dest': str(self.raw.parent / 'test91/pending.tar'),
                             'status': 'queued'})
        self.raw.write_text(json.dumps(raw))
        frozen = self.raw.read_bytes()
        result = self.migrate()
        audit = json.loads(result.audit_path.read_text())
        self.assertIsNone(audit['corpus'])
        self.assertEqual(audit['raw_manifest']['archive_count'], 2)
        self.assertFalse(audit['raw_manifest']['payloads_rehashed'])
        self.assertEqual(self.raw.read_bytes(), frozen)
        self.assertFalse((self.out / 'data/corpus').exists())

    def test_corpus_manifest_bounds_must_match_frozen_identity(self):
        corpus = self.add_corpus()
        path = corpus / 'corpus_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['until'] = '2026-09-08T00:00:00+00:00'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(migration.MigrationError, 'manifest|identity'):
            self.migrate()

    def test_optional_prepare_or_trainer_lock_is_respected(self):
        self.add_corpus()
        training = self.out / 'training'
        training.mkdir()
        (training / 'lc0.lock').write_text('')
        for path in (self.out / 'data/corpus/prepare.lock', training / 'lc0.lock'):
            with self.subTest(path=path), path.open('r+') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(migration.MigrationError, 'running|held'):
                    self.migrate()

    def test_optional_lock_appearing_after_acquisition_refuses(self):
        validate = migration._validate_repository
        training = self.out / 'training'
        training.mkdir()
        lock = training / 'lc0.lock'
        def create_lock(*args, **kwargs):
            lock.touch()
            return validate(*args, **kwargs)
        with mock.patch.object(migration, '_validate_repository', side_effect=create_lock), \
             self.assertRaisesRegex(migration.MigrationError, 'lock'):
            self.migrate()
        self.assertEqual(self.pin.read_text(), self.old + '\n')

    def test_replaced_launcher_lock_refuses(self):
        def replace_lock(*_):
            replacement = self.out / 'replacement.lock'
            replacement.write_text('different inode')
            replacement.replace(self.lock)
        with self.assertRaisesRegex(migration.MigrationError, 'lock'):
            self.migrate(before_pin_replace=replace_lock)
        self.assertEqual(self.pin.read_text(), self.old + '\n')

    def test_any_training_state_or_artifact_refuses_even_zero_chunks(self):
        for relative in ('training/lc0_state.json', 'training/chunks', 'training/train/checkpoint.json',
                         'training/optimizer.pt', 'training/accepted/old.nnue',
                         'autopilot_state.json', 'lc0_state.json'):
            path = self.out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"completed_chunks":0}')
            with self.subTest(relative=relative), self.assertRaisesRegex(migration.MigrationError, 'training|self-play|artifact'):
                self.migrate()
            path.unlink()
        self.assertEqual(self.pin.read_text(), self.old + '\n')

    def test_bootstrap_hash_root_and_symlink_validation(self):
        for changes in (dict(expected_bootstrap_sha256='0' * 64),
                        dict(expected_active_sha256='0' * 64),
                        dict(bootstrap_root=self.out), dict(bootstrap_root=self.root / 'unrelated')):
            with self.subTest(changes=changes), self.assertRaises(migration.MigrationError):
                self.migrate(**changes)
        linked = self.bootstrap / 'linked.json'
        linked.symlink_to(self.checkpoint)
        with self.assertRaisesRegex(migration.MigrationError, 'symlink'):
            self.migrate(bootstrap_checkpoint=linked)

    def test_intermediate_directory_aliases_refuse(self):
        actual = self.bootstrap / 'actual'
        actual.mkdir()
        checkpoint = actual / 'checkpoint.json'
        checkpoint.write_bytes(self.checkpoint.read_bytes())
        alias = self.bootstrap / 'alias'
        alias.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(migration.MigrationError, 'symlink'):
            self.migrate(bootstrap_checkpoint=alias / checkpoint.name)
        data = self.out / 'data'
        moved = self.root / 'external-data'
        data.rename(moved)
        data.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(migration.MigrationError, 'symlink|outside'):
            self.migrate()
        self.assertEqual(self.pin.read_text(), self.old + '\n')

    def test_audit_removed_corrupted_or_replaced_in_hook_refuses(self):
        audit_path = migration.audit_path_for(self.out, self.old, self.new)
        for action in ('remove', 'corrupt', 'replace'):
            def tamper(path, _):
                if action == 'remove':
                    path.unlink()
                elif action == 'corrupt':
                    path.write_text('invalid JSON')
                else:
                    audit = json.loads(path.read_text())
                    audit['prepared_at_utc'] = '2000-01-01T00:00:00Z'
                    path.write_text(json.dumps(audit))
            with self.subTest(action=action), self.assertRaises(migration.MigrationError):
                self.migrate(before_pin_replace=tamper)
            self.assertEqual(self.pin.read_text(), self.old + '\n')
            audit_path.unlink(missing_ok=True)

    def test_prepared_audit_directory_alias_refuses_retry(self):
        def crash(*_):
            raise RuntimeError('simulated crash')
        with self.assertRaises(RuntimeError):
            self.migrate(before_pin_replace=crash)
        audit_directory = migration.audit_path_for(self.out, self.old, self.new).parent
        moved = self.root / 'external-audit'
        audit_directory.rename(moved)
        audit_directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(migration.MigrationError, 'symlink|outside'):
            self.migrate()
        self.assertEqual(self.pin.read_text(), self.old + '\n')

    def test_bad_pins_dirty_repository_and_wrong_head_refuse(self):
        for changes in (dict(expected_old_commit='short'), dict(expected_new_commit='f' * 40)):
            with self.subTest(changes=changes), self.assertRaises(migration.MigrationError):
                self.migrate(**changes)
        self.pin.write_text('f' * 40 + '\n')
        with self.assertRaisesRegex(migration.MigrationError, 'pin'):
            self.migrate()
        self.pin.write_text(self.old + '\n')
        (self.repo / 'untracked').write_text('dirty')
        with self.assertRaisesRegex(migration.MigrationError, 'clean'):
            self.migrate()

    def test_non_fast_forward_refuses(self):
        _git(self.repo, 'checkout', '-q', self.old)
        (self.repo / 'branch.py').write_text('diverged')
        _git(self.repo, 'add', 'branch.py')
        _git(self.repo, 'commit', '-qm', 'diverged')
        head = _git(self.repo, 'rev-parse', 'HEAD')
        self.pin.write_text(self.new + '\n')
        with self.assertRaisesRegex(migration.MigrationError, 'fast-forward'):
            self.migrate(expected_old_commit=self.new, expected_new_commit=head)

    def test_crashes_before_and_after_pin_replace_retry_idempotently(self):
        def crash(*_):
            raise RuntimeError('simulated crash')
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.migrate(before_pin_replace=crash)
        audit_path = migration.audit_path_for(self.out, self.old, self.new)
        audit = audit_path.read_bytes()
        replace = migration._atomic_replace_pin
        def replace_then_crash(*args):
            replace(*args)
            crash()
        with mock.patch.object(migration, '_atomic_replace_pin', side_effect=replace_then_crash), \
             self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.migrate()
        self.assertEqual(self.migrate().status, 'already-applied')
        self.assertEqual(audit_path.read_bytes(), audit)

    def test_changed_data_after_preparation_or_in_hook_refuses_pin_move(self):
        corpus = self.add_corpus()
        for path in (self.raw, corpus / 'progress.sqlite3', self.checkpoint):
            with self.subTest(path=path):
                original = path.read_bytes()
                def change(*_):
                    path.write_bytes(original + b' ')
                with self.assertRaisesRegex(migration.MigrationError, 'changed|match|SHA'):
                    self.migrate(before_pin_replace=change)
                self.assertEqual(self.pin.read_text(), self.old + '\n')
                path.write_bytes(original)
                # Matching bytes allow retry against the immutable prepared audit.
        def crash(*_):
            raise RuntimeError('simulated crash')
        with self.assertRaises(RuntimeError):
            self.migrate(before_pin_replace=crash)
        self.raw.write_bytes(self.raw.read_bytes() + b' ')
        with self.assertRaisesRegex(migration.MigrationError, 'changed|match'):
            self.migrate()

    def test_new_pin_without_audit_and_state_after_applied_migration_refuse(self):
        self.pin.write_text(self.new + '\n')
        with self.assertRaisesRegex(migration.MigrationError, 'audit'):
            self.migrate()
        self.pin.write_text(self.old + '\n')
        self.migrate()
        training = self.out / 'training'
        training.mkdir()
        (training / 'lc0_state.json').write_text('{"completed_chunks":0}')
        with self.assertRaisesRegex(migration.MigrationError, 'training'):
            self.migrate()

    def test_cli_requires_explicit_lc0_mode(self):
        common = ['--repo-root', str(self.repo), '--out-root', str(self.out),
                  '--expected-old-commit', self.old, '--expected-new-commit', self.new]
        args = migration._parse_args(common)
        self.assertEqual(args.mode, 'selfplay')
        with mock.patch.object(migration, 'migrate_source_commit') as legacy:
            legacy.return_value = migration.MigrationResult('fixture', self.out, self.old, self.new)
            self.assertEqual(migration.main(common), 0)
            legacy.assert_called_once()
        self.assertEqual(migration.main(common + ['--mode', 'lc0-pretraining',
            '--bootstrap-root', str(self.bootstrap), '--bootstrap-checkpoint', str(self.checkpoint),
            '--bootstrap-checkpoint-sha256', _sha256(self.checkpoint), '--active-model', str(self.active),
            '--active-model-sha256', _sha256(self.active), '--supervisor-stop-verified']), 0)


if __name__ == '__main__':
    unittest.main()
