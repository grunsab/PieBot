#!/usr/bin/env python3
"""Measure frozen NNUE candidates against a pinned CCRL boundary cohort.

Each opponent has one sequential arena, for at most three simultaneous games.
The measurement uses local hardware and no tablebases; it is not an official
CCRL rating. This program never changes training state or promotes a model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

BASELINE_SHA = '271a5a108ee03e20a0036b683e108f76dd44fc2e6d289cc3d3f8c839082c519c'
COHORT = ('carp-3.0.1', 'lambergar-1.5', 'schoenemann-0.5.0')


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + '.part')
    with part.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    part.replace(path)


@contextmanager
def exclusive_lock(path: Path):
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('ranked engine monitor is already running') from exc
        yield


def snapshot_model(source: Path, expected: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'model.nnue'
    if target.exists():
        if sha256(target) != expected:
            raise ValueError('existing measurement model SHA mismatch')
        return target
    part = directory / 'model.nnue.part'
    shutil.copyfile(source, part)
    if sha256(part) != expected:
        part.unlink()
        raise ValueError('candidate model SHA mismatch')
    with part.open('rb') as stream:
        os.fsync(stream.fileno())
    part.replace(target)
    return target


def measurement_due(state: dict, model_sha: str, now: float, *, ended=False) -> bool:
    return (not state or bool(state.get('pending')) or
            (model_sha != state.get('last_model_sha256') and
             (ended or now - state.get('last_started_at', state.get('last_completed_at', 0)) >= 86400)))


def choose_candidate(state: dict, campaign: dict, baseline: Path, baseline_sha: str):
    if state.get('pending'):
        pending = state['pending']
        return Path(pending['model_source']), pending['model_sha256']
    if not state.get('measurements'):
        return baseline, baseline_sha
    return (Path(campaign.get('best_quant_path') or baseline),
            campaign.get('best_quant_sha256') or baseline_sha)


def verify_engine(directory: Path) -> dict:
    # Reuse the install verifier: also reject added files, changed modes and
    # symlinks which could affect an engine's automatic runtime/network lookup.
    try:
        from scripts.fetch_ranked_engines import verify_existing
    except ModuleNotFoundError:
        from fetch_ranked_engines import verify_existing
    return verify_existing(directory)


def arena_command(repo: Path, piebot: Path, model: Path, engine: dict, book: Path,
                  results: Path, *, games: int, seed: int) -> list[str]:
    command = [sys.executable, str(repo / 'scripts/uci_elo_arena.py'),
               '--piebot-command', shlex.join([str(piebot)]), '--piebot-nnue', str(model),
               '--piebot-blend', '75', '--piebot-hash', '64',
               '--opponent-command', shlex.join([engine['executable']]),
               '--opponent-sha256', engine['executable_sha256'],
               '--opponent-name', engine['name'], '--opponent-rating', str(engine['rating']),
               '--opponent-rating-source', engine['rating_source'],
               '--opponent-rating-snapshot', engine['rating_snapshot'],
               '--opponent-hash', '64', '--games', str(games),
               '--time-control', '120+1', '--max-plies', '600',
               '--game-wall-time', '1800', '--startup-timeout', '30',
               '--openings-file', str(book), '--seed', str(seed), '--results', str(results)]
    for asset in engine.get('auxiliary_files', []):
        command.extend(['--opponent-asset', asset['path'],
                        '--opponent-asset-sha256', asset['sha256']])
    if (engine.get('qualification') or {}).get('single_thread_qualified'):
        command.append('--opponent-single-thread-qualified')
    return command


def cohort_report(results: dict, games: int) -> dict:
    summaries = {key: value['summary'] for key, value in results.items()}
    eligible = bool(summaries) and all(
        s.get('measurement_eligible') is True and s.get('games') == games and
        s.get('complete_pairs') == games // 2 for s in summaries.values())
    outperformed = eligible and all(
        isinstance(s.get('score_95_ci'), list) and len(s['score_95_ci']) == 2 and
        isinstance(s['score_95_ci'][0], (int, float)) and s['score_95_ci'][0] > .5
        for s in summaries.values())
    return {'schema': 'piebot-ranked-cohort-report-v1', 'opponents': summaries,
            'measurement_eligible': eligible, 'all_opponents_outperformed': outperformed,
            'local_boundary_supported': outperformed and set(results) == set(COHORT),
            'interpretation': 'Local paired matches against the pinned cohort; '
                              'not an official CCRL rating or human Elo estimate.'}


def run_arena(command: list[str], cwd: Path, log: Path):
    with log.open('a') as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--piebot-bin', type=Path, required=True)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--engines', default=','.join(COHORT))
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-sha256', default=BASELINE_SHA)
    parser.add_argument('--book', type=Path, required=True)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.games < 2 or args.games % 2:
        parser.error('--games must be positive and even')
    ids = args.engines.split(',')
    if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        parser.error('--engines must select one to three unique engines')
    repo, piebot, book = (path.resolve(strict=True) for path in
                          (args.repo, args.piebot_bin, args.book))
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    directories = {key: (args.engine_root / key).resolve(strict=True) for key in ids}
    engines = {key: verify_engine(path) for key, path in directories.items()}
    config = {'schema': 'piebot-ranked-monitor-v1', 'games': args.games,
              'seed': args.seed, 'time_control': '120+1', 'blend': 75,
              'piebot_path': str(piebot), 'piebot_sha256': sha256(piebot),
              'book_path': str(book), 'book_sha256': sha256(book),
              'baseline_path': str(args.baseline.resolve()),
              'baseline_sha256': args.baseline_sha256,
              'campaign_root': str(args.campaign_root.resolve()),
              'engine_manifests': {key: sha256(path / 'manifest.json')
                                   for key, path in directories.items()},
              'runner_sha256': sha256(repo / 'scripts/uci_elo_arena.py'),
              'installer_sha256': sha256(repo / 'scripts/fetch_ranked_engines.py'),
              'monitor_sha256': sha256(Path(__file__)), 'max_concurrent_games': len(ids),
              'training_concurrent': True, 'tablebases': False}
    journal = root / 'monitor_state.json'
    campaign_path = args.campaign_root / 'training/lc0_state.json'
    with exclusive_lock(root / 'monitor.lock'):
        state = json.loads(journal.read_text()) if journal.exists() else {}
        if state and state.get('config') != config:
            raise ValueError('measurement configuration changed; use a new output root')
        while True:
            campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
            now = time.time()
            terminal = campaign.get('status') in ('complete', 'completed', 'deadline-reached')
            deadline_reached = (campaign.get('deadline_at') is not None and now >= campaign['deadline_at'])
            source, expected = choose_candidate(state, campaign, args.baseline, args.baseline_sha256)
            if measurement_due(state, expected, now, ended=terminal or deadline_reached):
                pending = state.get('pending')
                directory = (Path(pending['directory']) if pending else
                             root / f"measurement_{state.get('measurements', 0):03d}_{expected[:12]}")
                try:
                    model = snapshot_model(source, expected, directory)
                except FileNotFoundError:
                    time.sleep(60)
                    continue
                for key, path in directories.items():
                    if sha256(path / 'manifest.json') != config['engine_manifests'][key]:
                        raise ValueError('opponent manifest changed during measurement')
                    verify_engine(path)
                if sha256(piebot) != config['piebot_sha256'] or sha256(book) != config['book_sha256']:
                    raise ValueError('measurement binary/book changed')
                if not pending:
                    pending = {'directory': str(directory), 'model_source': str(source),
                               'model_sha256': expected, 'started_at': now}
                    state = {**state, 'config': config, 'pending': pending}
                    atomic_json(journal, state)
                    atomic_json(directory / 'measurement_identity.json',
                                {**config, **pending, 'opponents': engines})
                print(json.dumps({'event': 'ranked-start', **pending}), flush=True)
                with ThreadPoolExecutor(max_workers=len(ids)) as pool:
                    jobs = [pool.submit(run_arena,
                                        arena_command(repo, piebot, model, engines[key], book,
                                                      directory / f'{key}.json',
                                                      games=args.games, seed=args.seed),
                                        Path(engines[key]['cwd']), directory / f'{key}.log')
                            for key in ids]
                    for job in jobs:
                        job.result()
                results = {key: json.loads((directory / f'{key}.json').read_text()) for key in ids}
                report = cohort_report(results, args.games)
                report.update({'model_sha256': expected, 'configuration': config,
                               'completed_at': time.time()})
                atomic_json(directory / 'cohort_report.json', report)
                state = {'config': config, 'measurements': state.get('measurements', 0) + 1,
                         'last_model_sha256': expected, 'last_completed_at': time.time(),
                         'last_started_at': pending['started_at'],
                         'last_report': str(directory / 'cohort_report.json')}
                atomic_json(journal, state)
                print(json.dumps({'event': 'ranked-complete', **state}), flush=True)
                if args.once:
                    return 0
                # Re-read the campaign before deciding whether its final model was measured.
                continue
            # The controller may finish and publish its final chunk after the
            # wall deadline. Only its committed terminal status permits exit.
            if terminal:
                return 0
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
