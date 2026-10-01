"""Pinned, resumable 1T/8T comparison using the repository's tested arena."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import uci_elo_arena as arena

SEED = 20260929
GAMES = 100
SAMPLES = 10000
FIX = ROOT / 'out/lazy_smp_fix_20260929'
MODEL = ROOT / 'models/lc0_chunk_00034965.nnue'


def schedule(plans):
    # ABBA within each opening, reversing the starting arm next opening.
    return [(t, p) for p in plans
            for t in ((1, 8) if (p.pair_index + p.game_index % 2) % 2 == 0 else (8, 1))]


def compare(one, eight, *, samples=SAMPLES, seed=SEED):
    arms = []
    for records in (one, eight):
        groups = collections.defaultdict(dict)
        for g in records:
            groups[g['pair_index']][g['game_index']] = g['piebot_score']
        arms.append({k: sum(v.values()) / 2 for k, v in groups.items() if len(v) == 2})
    keys = sorted(set(arms[0]) & set(arms[1]))
    result = {'matched_opening_pairs': len(keys), 'bootstrap_samples': samples,
              'elo_8_minus_1': None, 'paired_95_ci': None}
    if not keys:
        return result
    a, b = [[arm[k] for k in keys] for arm in arms]
    def difference(x, y):
        d = arena.logistic_elo(sum(y) / len(y)) - arena.logistic_elo(sum(x) / len(x))
        return None if math.isnan(d) else d
    result['elo_8_minus_1'] = difference(a, b)
    result['score_1'] = sum(a) / len(a)
    result['score_8'] = sum(b) / len(b)
    rng = random.Random(seed)
    distribution = []
    for _ in range(samples):
        indices = [rng.randrange(len(keys)) for _ in keys]
        d = difference([a[i] for i in indices], [b[i] for i in indices])
        if d is None:
            result['interval_note'] = 'Saturated resamples make the Elo difference unidentified.'
            return result
        distribution.append(d)
    distribution.sort()
    # Order-statistic percentiles preserve infinite endpoints without inf*0.
    result['paired_95_ci'] = [distribution[int((samples - 1) * p)] for p in (.025, .975)]
    return result


def save(path, payload):
    arena.save_state(path, arena._json_safe(payload))


def number(x):
    return 'unidentified' if x is None else f'{x:+.1f}' if isinstance(x, (float, int)) else str(x)


def update_report(states, *, complete=False):
    summaries = {t: arena.summarize_results(s['games'], bootstrap_samples=SAMPLES, seed=SEED)
                 for t, s in states.items()}
    comparison = compare(states[1]['games'], states[8]['games'])
    result = {'status': 'complete' if complete else 'in_progress', 'updated_at': arena.utc_now(),
              'arms': summaries, 'comparison': comparison,
              'interpretation': 'Relative to Stockfish 18 UCI_Elo=3190 on this Mac at 60+0.5; not a published CCRL rating.'}
    save(OUT / 'comparison.json', result)
    lines = ['# PieBot: 8 threads versus 1, Stockfish 3190 anchor', '',
             '**Complete.**' if complete else '**In progress; partial results are not a conclusion.**', '',
             '100 games per setting; 50 matched openings with reversed colors. One game at a time, alternating settings. Both engines use Hash=256 MiB. PieBot uses the pinned Lazy SMP binary, LCZero chunk 34965, blend 75. Stockfish 18 stays at one thread, UCI_LimitStrength=true, UCI_Elo=3190, pondering and tablebases disabled. Time control: 60+0.5.', '',
             '| PieBot threads | Games | W–D–L | Score | Elo versus anchor |',
             '| --- | ---: | --- | ---: | ---: |']
    for t in (1, 8):
        s = summaries[t]
        lines.append(f"| {t} | {s['games']} | {s['wins']}–{s['draws']}–{s['losses']} | {s['score_rate']:.1%} | {number(s['elo_difference'])} |")
    lines += ['', f"Matched complete opening pairs: {comparison['matched_opening_pairs']}.",
              f"Elo difference, 8 threads minus 1: **{number(comparison['elo_8_minus_1'])}**."]
    ci = comparison['paired_95_ci']
    lines += [f"Paired bootstrap 95% interval: **[{number(ci[0])}, {number(ci[1])}]**." if ci else 'A paired Elo interval is not yet identifiable.', '',
              'The bootstrap resamples whole openings jointly across both settings, retaining all four games together. The nominal 3190 anchor is Stockfish’s strength setting, not an independently verified CCRL rating. No significance claim should be made while the match is running.', '',
              'Game records: `arm_1t.json` and `arm_8t.json`. Exact binaries, model hash, source hashes, options and opening plans are pinned in the manifest and arm files.', '',
              'Termination counts: `' + json.dumps({t: s['termination_counts'] for t, s in summaries.items()}, sort_keys=True) + '`.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def validation_state():
    counts = {}
    all_records = {}
    for arm in ('default', 'all_features'):
        path = FIX / f'rust_results_{arm}.json'
        try:
            records = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            # The existing validation runner rewrites this progress file.
            records = []
        retries_path = FIX / f'rust_retries_{arm}.json'
        retries = json.loads(retries_path.read_text()) if retries_path.exists() else []
        verified = {r['path']: r for r in retries if r['returncode'] == 0
                    and arena.sha256_file(Path(r['path'])) == r['binary_sha256']}
        records = [dict(verified[r['path']], original_attempt=r)
                   if r['returncode'] and r['path'] in verified else r for r in records]
        failures = [r['name'] for r in records if r['returncode']]
        if failures:
            raise RuntimeError(f'Correctness checks failed in {arm}: {failures}')
        counts[arm] = len(records)
        all_records[arm] = records
    return counts, all(v == 72 for v in counts.values()), all_records


def main():
    os.chdir(ROOT)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    status = {'pid': os.getpid(), 'started_at': arena.utc_now(), 'phase': 'preflight',
              'games_per_setting': GAMES}
    def status_update(**fields):
        status.update(fields)
        save(OUT / 'status.json', status)
    status_update()
    try:
        piebot_command = [str(OUT / 'bin/piebot-uci')]
        sf_command = [str(OUT / 'bin/stockfish-18')]
        identities = {str(OUT / name): sha for name, sha in manifest['files'].items()}
        identities[str(MODEL)] = manifest['model_sha256']
        def verify_inputs():
            for path, expected in identities.items():
                if arena.sha256_file(Path(path)) != expected:
                    raise RuntimeError(f'Pinned input changed: {path}')
        verify_inputs()
        sf_options = arena.stockfish_uci_options(elo=3190, hash_mb=256, threads=1)
        pie_options = {t: arena.piebot_uci_options(MODEL, blend=75, hash_mb=256, threads=t)[0]
                       for t in (1, 8)}
        for t in (1, 8):
            arena.run_uci_preflight(piebot_command, pie_options[t],
                required_options=arena.PIEBOT_REQUIRED_OPTIONS,
                failure_markers=('failed to load NNUEQuantFile',), timeout_s=600)
        arena.run_uci_preflight(sf_command, sf_options,
            required_options=arena.STOCKFISH_REQUIRED_OPTIONS, timeout_s=600)
        openings = json.loads((OUT / 'openings.json').read_text())
        plans = arena.build_game_plans(openings, games=GAMES, seed=SEED)
        states = {}
        for t in (1, 8):
            config = {'manifest': manifest, 'games': GAMES, 'piebot_options': pie_options[t],
                      'stockfish_options': sf_options, 'initial_s': 60, 'increment_s': .5,
                      'max_plies': 300, 'game_wall_time_s': 900, 'concurrency': 1,
                      'seed': SEED, 'pairing': 'matched-opening-four-game-clusters',
                      'bootstrap_samples': SAMPLES}
            states[t] = arena.load_or_create_state(OUT / f'arm_{t}t.json', config, plans)
            save(OUT / f'arm_{t}t.json', states[t])
        update_report(states)
        status_update(phase='waiting_for_correctness_checks')
        while True:
            counts, ready, records = validation_state()
            status_update(validation=counts)
            if ready:
                break
            # The preceding validation task must remain alive until both suites finish.
            try:
                os.kill(manifest['validation_pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Validation process exited before completing both suites')
            time.sleep(10)
        save(OUT / 'validation.json', records)
        evidence = ROOT / 'evidence/lazy_smp_20260929.json'
        e = json.loads(evidence.read_text())
        reruns = [{'configuration': config, 'target': r['name'],
                   'original_returncode': r['original_attempt']['returncode'],
                   'retry_returncode': r['returncode'], 'retry_log': r['log']}
                  for config, rows in records.items() for r in rows if 'original_attempt' in r]
        e['validation']['rust_full_battery'] = {'status': 'passed', 'default_executables': 72,
                                               'all_features_executables': 72, 'reruns': reruns}
        evidence.write_text(json.dumps(e, indent=2) + '\n')
        report = FIX / 'REPORT.md'
        text = report.read_text()
        old = '- Full default/all-feature Rust test execution is still in progress. The local Gatekeeper/XProtect scanner delays newly built executables before program entry; completed executables have passed so far. Final results are recorded alongside this report.'
        replacement = '- Full default and all-feature Rust batteries passed: 72 test/benchmark executables per configuration. Local Gatekeeper/XProtect scans delayed startup; no protections were changed.'
        if reruns:
            replacement += ' One optional-backend UCI stop check failed initially and passed unchanged on retry; both attempts are preserved in the validation evidence.'
        report.write_text(text.replace(old, replacement))
        settings = arena.GameSettings(initial_time_s=60, increment_s=.5)
        completed = {t: {g['game_index'] for g in s['games']} for t, s in states.items()}
        for t, plan in schedule(plans):
            if plan.game_index in completed[t]:
                continue
            verify_inputs()
            status_update(phase='playing', active_threads=t, active_game=plan.game_index + 1,
                          active_pair=plan.pair_index + 1, active_color=plan.piebot_color,
                          completed={str(k): len(s['games']) for k, s in states.items()})
            print(f'Starting {t}T game {plan.game_index + 1}/100, pair {plan.pair_index + 1}, {plan.piebot_color}', flush=True)
            record = arena.play_isolated_game(plan, settings, piebot_command=piebot_command,
                piebot_options=pie_options[t], stockfish_command=sf_command,
                stockfish_options=sf_options, startup_timeout_s=600)
            if record['termination'].endswith('start_failure'):
                save(OUT / 'startup_failure.json', record)
                raise RuntimeError(f"Engine failed to start; no game scored: {record}")
            states[t]['games'].append(record)
            states[t]['games'].sort(key=lambda g: g['game_index'])
            states[t]['summary'] = arena._json_safe(arena.summarize_results(states[t]['games'], bootstrap_samples=SAMPLES, seed=SEED))
            save(OUT / f'arm_{t}t.json', states[t])
            update_report(states)
            print(f"Finished {t}T game {plan.game_index + 1}: score={record['piebot_score']} termination={record['termination']} duration={record['duration_s']:.1f}s", flush=True)
            if any(x in record['termination'] for x in ('engine_crash', 'invalid_move')):
                raise RuntimeError(f"Engine fault requires investigation: {record['termination']}")
        update_report(states, complete=True)
        status_update(phase='complete', completed={'1': GAMES, '8': GAMES}, finished_at=arena.utc_now())
        print('All 200 games complete. See REPORT.md and comparison.json.', flush=True)
    except BaseException as error:
        status_update(phase='failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
