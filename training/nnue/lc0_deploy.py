"""Verified acquisition -> preparation -> supervised LCZero training launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

CHECKPOINT_SHA = '144699077a19f50f7097de8426ed73f0f9a9fa30971d9ba0219250044311697d'
ACTIVE_SHA = '271a5a108ee03e20a0036b683e108f76dd44fc2e6d289cc3d3f8c839082c519c'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assert_separate_root(output: Path, protected: Path) -> None:
    output, protected = output.resolve(), protected.resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('LCZero output must be separate from the self-play restart root')


def validate_bootstrap(checkpoint: Path, checkpoint_sha: str, quant: Path, quant_sha: str) -> None:
    for path, expected in ((checkpoint, checkpoint_sha), (quant, quant_sha)):
        if sha256(path) != expected:
            raise ValueError(f'bootstrap SHA-256 mismatch: {path}')
    with checkpoint.open() as stream:
        metadata = json.load(stream)
    if (metadata.get('format') != 'piebot-halfkp-dp-screlu-v1-torch'
            or metadata.get('hidden_dim') != 1024 or metadata.get('input_dim') != 40960):
        raise ValueError('bootstrap must be the compatible v2 h1024 float checkpoint')
    with quant.open('rb') as stream:
        if stream.read(8) != b'PIENNQ02':
            raise ValueError('incumbent must be a v2 PIENNQ02 model')


def pin_source(output: Path, commit: str) -> None:
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('source commit must be a full lowercase Git SHA')
    output.mkdir(parents=True, exist_ok=True)
    pin = output / 'source_git_commit'
    if pin.exists():
        if pin.read_text().strip() != commit:
            raise ValueError('LCZero source pin differs; use a new lineage for changed code')
        return
    # Exclusive creation avoids silently replacing another launcher's source pin.
    with pin.open('x') as stream:
        stream.write(commit + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def wait_for_training(command: list[str], **kwargs):
    # Supervisor signals the entire group. The trainer saves at a chunk boundary;
    # keep its supervised parent alive until that save and lock release finish.
    previous = signal.signal(signal.SIGTERM, lambda *_: None)
    try:
        return subprocess.run(command, **kwargs)
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out-root', type=Path, default=Path('/workspace/piebot_lc0_20260907'))
    parser.add_argument('--selfplay-root', type=Path, default=Path('/workspace/piebot_campaign_v8'))
    parser.add_argument('--hours', type=float, default=336)
    parser.add_argument('--since', default='2026-07-07')
    parser.add_argument('--until', default='2026-09-07')
    parser.add_argument('--min-free-gib', type=float, default=50)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = args.out_root.resolve()
    assert_separate_root(output, args.selfplay_root)
    checkpoint = args.selfplay_root / 'cycles/cycle_000206/train/checkpoint.json'
    active = args.selfplay_root / 'cycles/cycle_000168/nnue_quant.nnue'
    validate_bootstrap(checkpoint, CHECKPOINT_SHA, active, ACTIVE_SHA)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    for command in (['git', '-C', str(repo), 'diff', '--quiet'],
                    ['git', '-C', str(repo), 'diff', '--cached', '--quiet']):
        subprocess.run(command, check=True)
    for binary in ('compare_play', 'uci', 'accept', 'accept_temp'):
        if not os.access(repo / 'PieBot/target/release' / binary, os.X_OK):
            raise ValueError(f'missing release binary: {binary}')
    if shutil.which('curl') is None:
        raise ValueError('curl is required for the verified archive transport')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required; refusing a silent CPU training fallback')
    if torch.ones(1, device='cuda').sum().item() != 1:
        raise ValueError('CUDA preflight failed')
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    min_free = int(args.min_free_gib * 1024 ** 3)
    if shutil.disk_usage(ancestor).free < min_free:
        raise ValueError('insufficient free disk reserve')
    print(json.dumps({'preflight': 'passed', 'source_commit': commit,
                      'checkpoint_sha256': CHECKPOINT_SHA, 'active_sha256': ACTIVE_SHA}), flush=True)
    if args.preflight_only:
        return 0
    # Nothing writes a source pin until bootstrap, binaries, CUDA and disk pass.
    pin_source(output, commit)
    from .autopilot import _single_instance_lock
    with _single_instance_lock(output / 'launcher.lock'):
        raw = output / 'data/raw'
        manifest = raw / 'manifest.json'
        subprocess.run([
            sys.executable, '-m', 'training.nnue.fetch_lc0_bins', '--out', str(raw),
            '--manifest', str(manifest), '--since', args.since, '--until', args.until,
            '--suites', 'test91', '--limit-per-suite', '0', '--backend', 'curl',
            '--skip-existing'], cwd=repo, check=True)
        from .lc0_corpus import prepare_corpus
        corpus = prepare_corpus(manifest, output / 'data/corpus', since=args.since,
                                until=args.until, min_free_bytes=min_free, workers=16)
        command = [sys.executable, '-m', 'training.nnue.lc0_autopilot',
                   '--corpus-manifest', str(corpus), '--out-root', str(output / 'training'),
                   '--initial-checkpoint', str(checkpoint), '--initial-active-model', str(active),
                   '--source-commit', commit, '--piebot-dir', str(repo / 'PieBot'),
                   '--hours', str(args.hours), '--device', 'cuda',
                   '--disk-reserve-gib', str(args.min_free_gib)]
        # Stay in the supervisor process group through every data/training stage.
        wait_for_training(command, cwd=repo, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
