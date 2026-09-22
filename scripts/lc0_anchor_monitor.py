#!/usr/bin/env python3
"""Independent fixed-SF16 baseline/daily measurements alongside LCZero training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ANCHOR_SHA = '8f60a016dc767e0d648a8665b8ede3e6e4d28c086ad90517ad26f55b9960bd84'
ACTIVE_SHA = '271a5a108ee03e20a0036b683e108f76dd44fc2e6d289cc3d3f8c839082c519c'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot_model(source: Path, expected: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / 'model.nnue'
    part = directory / 'model.nnue.part'
    shutil.copyfile(source, part)
    if sha256(part) != expected:
        part.unlink()
        raise ValueError('anchor model snapshot SHA mismatch')
    part.replace(dest)
    return dest


def measurement_due(state: dict, model_sha: str, now: float) -> bool:
    return (not state or (model_sha != state.get('last_model_sha256')
                          and now - state.get('last_completed_at', 0) >= 86400))


def ladder_command(repo: Path, model: Path, anchor: Path, output: Path, seed: int) -> list[str]:
    # The later August rung-dependence study supersedes the early 1500/1800 scale.
    return [sys.executable, str(repo / 'scripts/uci_elo_ladder.py'),
            '--piebot-command', str(repo / 'PieBot/target/release/uci'),
            '--piebot-nnue', str(model), '--piebot-blend', '75',
            '--stockfish-command', str(anchor), '--rungs', '3000,3190',
            '--games', '100', '--time-control', '60+0.5',
            '--seed', str(seed), '--out-dir', str(output)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--campaign-root', type=Path, default=Path('/workspace/piebot_lc0_20260907'))
    parser.add_argument('--anchor', type=Path, default=Path('/workspace/stockfish16'))
    parser.add_argument('--baseline', type=Path, default=Path('/workspace/piebot_campaign_v8/cycles/cycle_000168/nnue_quant.nnue'))
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from training.nnue.autopilot import _atomic_write_json, _single_instance_lock
    if sha256(args.anchor) != ANCHOR_SHA:
        raise ValueError('canonical SF16 anchor SHA mismatch')
    output = args.campaign_root / 'anchor'
    output.mkdir(parents=True, exist_ok=True)
    journal = output / 'monitor_state.json'
    campaign_path = args.campaign_root / 'training/lc0_state.json'
    with _single_instance_lock(output / 'monitor.lock'):
        state = json.loads(journal.read_text()) if journal.exists() else {}
        while True:
            campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
            if not state:
                model, expected = args.baseline, ACTIVE_SHA
            else:
                model = Path(campaign.get('best_quant_path') or args.baseline)
                expected = campaign.get('best_quant_sha256') or ACTIVE_SHA
            now = time.time()
            ended = (campaign.get('status') in ('complete', 'completed', 'deadline-reached')
                     or (campaign.get('deadline_at') is not None and now >= campaign['deadline_at']))
            if measurement_due(state, expected, now) or (ended and expected != state.get('last_model_sha256')):
                attempt = int(state.get('measurements', 0))
                directory = output / f'measurement_{attempt:03d}_{expected[:12]}'
                try:
                    frozen = snapshot_model(model, expected, directory)
                except FileNotFoundError:
                    time.sleep(60)
                    continue
                report = directory / 'ladder_report.json'
                identity = {'model_sha256': expected, 'anchor_sha256': ANCHOR_SHA,
                            'blend_percent': 75, 'rungs': [3000, 3190],
                            'time_control': '60+0.5', 'games_per_rung': 100,
                            'concurrent_training': True, 'started_at': now}
                _atomic_write_json(directory / 'measurement_identity.json', identity)
                print(json.dumps({'event': 'anchor-start', **identity}), flush=True)
                if not report.exists():
                    with (directory / 'runner.log').open('a') as log:
                        subprocess.run(ladder_command(repo, frozen, args.anchor, directory,
                                                      20260907 + attempt),
                                       cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads(report.read_text())
                state = {'last_completed_at': time.time(), 'last_model_sha256': expected,
                         'measurements': attempt + 1, 'last_report': str(report),
                         'last_result': result}
                _atomic_write_json(journal, state)
                print(json.dumps({'event': 'anchor-complete', 'report': str(report)}), flush=True)
            if ended:
                return 0
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
