"""A stopped LC0 lineage can be stepped back exactly one completed chunk."""

import contextlib
import fcntl
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import lc0_rollback_last_chunk as rollback
from training.nnue import lc0_autopilot

LOSSES = [0.62, 0.6173710015487671, 0.6177, 0.6176661698150635, 0.6176647729110718, 0.6176]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(obj):
    return json.dumps(obj, indent=2, sort_keys=True).encode('utf-8')


class RollbackLastChunkTests(unittest.TestCase):
    """Five committed chunks; chunk 1 is the best and chunk 4 is the newest."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / 'lc0'
        self.training = self.out / 'training'
        (self.training / 'chunks').mkdir(parents=True)
        (self.out / 'launcher.lock').write_text('4242')
        (self.training / 'lc0.lock').write_text('4243')
        self.active = self.root / 'selfplay' / 'active.nnue'
        self.active.parent.mkdir()
        self.active.write_bytes(b'PIENNQ02-active')
        self.records = [self.commit_chunk(number, cursor=10 + number) for number in range(5)]
        for pruned in (0, 2):  # retention keeps the best and the two newest
            for name in ('train/checkpoint.json', 'train/optimizer.pt', 'candidate.nnue'):
                (self.chunk_dir(pruned) / name).unlink()
        self.state_path = self.training / 'lc0_state.json'
        self.write_state(self.build_state())

    def chunk_dir(self, number):
        return self.training / 'chunks' / f'chunk_{number:08d}'

    def slot(self, number=4):
        sha = self.records[number]['checkpoint_sha256']
        return self.training / 'rollbacks' / f'chunk_{number:08d}_{sha[:12]}'

    def commit_chunk(self, number, *, cursor, pass_number=1):
        directory = self.chunk_dir(number)
        (directory / 'train').mkdir(parents=True)
        (directory / 'train/checkpoint.json').write_text(json.dumps({'weights': number}))
        (directory / 'train/optimizer.pt').write_bytes(b'adam-%d' % number)
        (directory / 'candidate.nnue').write_bytes(b'PIENNQ02-%d' % number)
        record = {
            'number': number, 'pass_number': pass_number, 'cursor': cursor, 'chunk_index': 100 + number,
            'chunk_sha256': f'{number:064x}', 'seed': 15715125292834772400 + number,
            'directory': str(directory), 'positions': 700000,
            'checkpoint_sha256': _sha256(directory / 'train/checkpoint.json'),
            'optimizer_sha256': _sha256(directory / 'train/optimizer.pt'),
            'quant_sha256': _sha256(directory / 'candidate.nnue'),
            'validation_loss': LOSSES[number], 'initial_validation_loss': 0.6175767380523681,
            'learning_rate': 0.0003664140808985614, 'finished_at': 1791685360.9944863 + number,
        }
        (directory / 'complete.json').write_bytes(_canonical(record))
        (directory / 'train/metrics.json').write_text(
            json.dumps({'optimizer_state': {'sha256': record['optimizer_sha256']}}))
        return record

    def pending(self, number, *, cursor, pass_number=1):
        return {'number': number, 'pass_number': pass_number, 'cursor': cursor, 'chunk_index': 100 + number,
                'chunk_sha256': f'{number:064x}', 'seed': 15715125292834772400 + number}

    def build_state(self):
        newest, best = self.records[-1], self.records[1]
        return {
            'schema': 'piebot-lc0-autopilot-v1',
            'identity': {'source': {'commit': 'a' * 40, 'digest': 'b' * 64}, 'seed': 7},
            'status': 'error', 'started_at': 1791400000.5, 'deadline_at': 1796669848.0,
            'completed_chunks': 5, 'pass_number': 1, 'cursor': 15,
            'in_progress': self.pending(5, cursor=15), 'history': list(self.records),
            'training_checkpoint_path': str(Path(newest['directory']) / 'train/checkpoint.json'),
            'training_optimizer_path': str(Path(newest['directory']) / 'train/optimizer.pt'),
            'active_model_path': str(self.active), 'active_model_sha256': _sha256(self.active),
            'active_model_blend_percent': 100, 'accepted_models': [],
            'last_gate': {'accepted': False, 'reason': 'screen-rejected'},
            'last_gate_identity': {'candidate': best['quant_sha256'], 'incumbent': _sha256(self.active)},
            'last_evaluation_at': 1791613242.1923144, 'evaluation_pending': False,
            'baseline_validation': {'loss': 0.63, 'checkpoint_sha256': 'c' * 64, 'validation_sha256': 'd' * 64},
            'best_validation_loss': best['validation_loss'],
            'best_checkpoint_path': str(Path(best['directory']) / 'train/checkpoint.json'),
            'best_checkpoint_sha256': best['checkpoint_sha256'],
            'best_quant_path': str(Path(best['directory']) / 'candidate.nnue'),
            'best_quant_sha256': best['quant_sha256'],
            'last_error': {'type': 'ValueError', 'at': 1791754942.6925702,
                           'message': 'initial optimizer model parameters do not match the initial checkpoint'},
        }

    def write_state(self, state):
        self.state_path.write_bytes(_canonical(state))
        os.chmod(self.state_path, 0o664)

    def read_state(self):
        return json.loads(self.state_path.read_bytes())

    def run_rollback(self, **overrides):
        kwargs = dict(out_root=self.out, expected_completed_chunks=5,
                      expected_checkpoint_sha256=self.records[-1]['checkpoint_sha256'],
                      supervisor_stop_verified=True)
        kwargs.update(overrides)
        return rollback.rollback_last_chunk(**kwargs)

    def assert_refused(self, pattern, **overrides):
        before = self.state_path.read_bytes()
        with self.assertRaisesRegex(rollback.RollbackError, pattern):
            self.run_rollback(**overrides)
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertTrue((self.chunk_dir(4) / 'complete.json').is_file())

    def test_state_returns_to_the_boundary_before_the_discarded_chunk(self):
        before = self.read_state()
        result = self.run_rollback()
        after = self.read_state()
        self.assertEqual(result.status, 'rolled-back')
        self.assertEqual(result.completed_chunks, 4)
        self.assertEqual(after['completed_chunks'], 4)
        self.assertEqual(after['history'], self.records[:4])
        self.assertEqual((after['pass_number'], after['cursor']), (1, 14))
        self.assertIsNone(after['in_progress'])
        self.assertEqual(after['training_checkpoint_path'], str(self.chunk_dir(3) / 'train/checkpoint.json'))
        self.assertEqual(after['training_optimizer_path'], str(self.chunk_dir(3) / 'train/optimizer.pt'))
        self.assertEqual((after['status'], after['last_error']), ('paused', None))
        changed = {'completed_chunks', 'history', 'cursor', 'in_progress', 'training_checkpoint_path',
                   'training_optimizer_path', 'status', 'last_error'}
        self.assertEqual({key: value for key, value in after.items() if key not in changed},
                         {key: value for key, value in before.items() if key not in changed})

    def test_the_trainer_resumes_from_the_previous_chunk_and_retrains_the_discarded_one(self):
        self.run_rollback()
        state = self.read_state()
        lc0_autopilot._verify_resume_artifacts(state)
        retrain = self.pending(4, cursor=14)
        self.assertEqual((state['completed_chunks'], state['pass_number'], state['cursor']),
                         (retrain['number'], retrain['pass_number'], retrain['cursor']))
        self.assertIsNone(lc0_autopilot._completed_chunk(self.chunk_dir(4), retrain))

    def test_discarded_chunk_is_kept_outside_the_trainers_retention(self):
        result = self.run_rollback()
        kept = self.slot() / 'chunk_00000004'
        self.assertEqual(result.discarded_directory, kept)
        self.assertFalse(self.chunk_dir(4).exists())
        self.assertEqual(json.loads((kept / 'complete.json').read_bytes()), self.records[4])
        state = self.read_state()
        state['history'].extend(self.commit_chunk(number, cursor=10 + number) for number in (4, 5))
        lc0_autopilot._retain(self.training, state)
        self.assertEqual((kept / 'train/checkpoint.json').read_text(), json.dumps({'weights': 4}))
        self.assertEqual((kept / 'train/optimizer.pt').read_bytes(), b'adam-4')

    def test_prior_state_and_an_audit_record_are_kept(self):
        before = self.state_path.read_bytes()
        result = self.run_rollback()
        prior_sha = hashlib.sha256(before).hexdigest()
        self.assertEqual((self.slot() / f'lc0_state.before.{prior_sha[:12]}.json').read_bytes(), before)
        audit = json.loads(result.audit_path.read_bytes())
        self.assertEqual(result.audit_path, self.slot() / 'rollback.json')
        self.assertEqual(audit['schema'], 'piebot-lc0-chunk-rollback-v1')
        self.assertEqual(audit['state_before_sha256'], prior_sha)
        self.assertEqual(audit['state_after_sha256'], _sha256(self.state_path))
        self.assertEqual(audit['discarded'], self.records[4])
        self.assertEqual(audit['resumes_from'], self.records[3])
        self.assertEqual(audit['discarded_directory'], str(self.slot() / 'chunk_00000004'))

    def test_state_is_written_as_the_trainer_writes_it(self):
        self.run_rollback()
        self.assertEqual(self.state_path.read_bytes(), _canonical(self.read_state()))
        self.assertEqual(stat.S_IMODE(self.state_path.stat().st_mode), 0o664)
        self.assertEqual(sorted(path.name for path in self.training.iterdir()),
                         ['chunks', 'lc0.lock', 'lc0_state.json', 'rollbacks'])

    def test_every_directory_whose_entries_change_is_synced_to_disk(self):
        synced = []
        sync = rollback._fsync_directory
        with mock.patch.object(rollback, '_fsync_directory',
                               side_effect=lambda path: (synced.append(Path(path)), sync(path))[1]):
            self.run_rollback()
        self.assertLessEqual({self.training, self.training / 'rollbacks', self.slot(), self.training / 'chunks'},
                             set(synced))

    def test_chunk_that_ended_a_pass_returns_to_the_end_of_that_pass(self):
        state = self.build_state()
        state.update(pass_number=2, cursor=0, in_progress=self.pending(5, cursor=0, pass_number=2),
                     evaluation_pending=True)
        self.write_state(state)
        self.run_rollback()
        after = self.read_state()
        self.assertEqual((after['pass_number'], after['cursor']), (1, 14))
        self.assertIs(after['evaluation_pending'], False)

    def test_dry_run_changes_nothing(self):
        before = self.state_path.read_bytes()
        result = self.run_rollback(dry_run=True)
        self.assertEqual(result.status, 'dry-run')
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertTrue((self.chunk_dir(4) / 'complete.json').is_file())
        self.assertFalse((self.training / 'rollbacks').exists())

    def test_rerun_finishes_a_rollback_interrupted_after_the_move(self):
        destination = self.slot() / 'chunk_00000004'
        destination.parent.mkdir(parents=True)
        self.chunk_dir(4).rename(destination)
        state = self.build_state()
        state['last_error'] = {'type': 'ValueError', 'at': 1791760000.0,
                               'message': f"committed campaign artifact checksum mismatch: {state['training_checkpoint_path']}"}
        self.write_state(state)
        result = self.run_rollback()
        self.assertEqual(result.status, 'rolled-back')
        self.assertEqual(self.read_state()['completed_chunks'], 4)
        lc0_autopilot._verify_resume_artifacts(self.read_state())

    def test_rerun_after_completion_reports_it_and_changes_nothing(self):
        self.run_rollback()
        before = self.state_path.read_bytes()
        result = self.run_rollback()
        self.assertEqual(result.status, 'already-rolled-back')
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_refuses_without_the_supervisor_stop_attestation(self):
        self.assert_refused('supervisor', supervisor_stop_verified=False)

    def test_refuses_while_the_trainer_holds_its_lock(self):
        with (self.training / 'lc0.lock').open('r+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assert_refused('lc0.lock is held')

    def test_refuses_while_the_launcher_holds_its_lock(self):
        with (self.out / 'launcher.lock').open('r+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assert_refused('launcher.lock is held')

    def test_refuses_a_different_completed_chunk_count(self):
        self.assert_refused('completed_chunks is 5, expected 6', expected_completed_chunks=6)

    def test_refuses_a_different_newest_checkpoint(self):
        self.assert_refused('newest checkpoint', expected_checkpoint_sha256='e' * 64)

    def test_refuses_when_the_previous_checkpoint_does_not_verify(self):
        (self.chunk_dir(3) / 'train/checkpoint.json').write_text(json.dumps({'weights': 'damaged'}))
        self.assert_refused('checksum mismatch.*chunk_00000003')

    def test_refuses_when_the_previous_optimizer_is_missing(self):
        (self.chunk_dir(3) / 'train/optimizer.pt').unlink()
        self.assert_refused('checksum mismatch.*chunk_00000003')

    def test_refuses_when_the_newest_chunk_is_the_best(self):
        state = self.build_state()
        newest = self.records[-1]
        state.update(best_validation_loss=newest['validation_loss'],
                     best_checkpoint_path=state['training_checkpoint_path'],
                     best_checkpoint_sha256=newest['checkpoint_sha256'],
                     best_quant_path=str(self.chunk_dir(4) / 'candidate.nnue'),
                     best_quant_sha256=newest['quant_sha256'])
        self.write_state(state)
        self.assert_refused('best')

    def test_refuses_when_the_chunk_directory_is_not_the_recorded_chunk(self):
        (self.chunk_dir(4) / 'complete.json').write_bytes(_canonical({**self.records[4], 'validation_loss': 0.5}))
        self.assert_refused('complete.json does not match')

    def test_refuses_a_pending_chunk_that_is_not_the_next_one(self):
        state = self.build_state()
        state['in_progress'] = self.pending(7, cursor=17)
        self.write_state(state)
        self.assert_refused('in_progress')

    def test_refuses_a_cursor_that_does_not_follow_the_newest_chunk(self):
        state = self.build_state()
        state.update(cursor=20, in_progress=None)
        self.write_state(state)
        self.assert_refused('cursor')

    def test_refuses_a_state_not_in_the_trainers_serialisation(self):
        self.state_path.write_bytes(json.dumps(self.build_state()).encode('utf-8'))
        self.assert_refused('serialisation')

    def test_refuses_a_lineage_with_a_single_chunk(self):
        state = self.build_state()
        only = self.records[0]
        state.update(history=[only], completed_chunks=1, cursor=11, in_progress=self.pending(1, cursor=11),
                     training_checkpoint_path=str(self.chunk_dir(0) / 'train/checkpoint.json'),
                     training_optimizer_path=str(self.chunk_dir(0) / 'train/optimizer.pt'))
        self.write_state(state)
        self.assert_refused('no earlier chunk', expected_completed_chunks=1,
                            expected_checkpoint_sha256=only['checkpoint_sha256'])

    def test_command_line_reports_the_result_as_json(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = rollback.main(['--out-root', str(self.out), '--expected-completed-chunks', '5',
                                  '--expected-checkpoint-sha256', self.records[-1]['checkpoint_sha256'],
                                  '--supervisor-stop-verified'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()),
                         {'status': 'rolled-back', 'completed_chunks': 4,
                          'audit_path': str(self.slot() / 'rollback.json'),
                          'discarded_directory': str(self.slot() / 'chunk_00000004')})

    def test_command_line_exits_2_when_refused(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = rollback.main(['--out-root', str(self.out), '--expected-completed-chunks', '6',
                                  '--expected-checkpoint-sha256', self.records[-1]['checkpoint_sha256'],
                                  '--supervisor-stop-verified'])
        self.assertEqual(code, 2)
        self.assertIn('LC0 chunk rollback refused', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
