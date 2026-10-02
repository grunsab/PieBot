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

from .disk_budget import available_bytes, decimal_gb_bytes
from .fetch_lc0_bins import validate_download_concurrency

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


def resolve_bootstrap(args) -> tuple[Path, str]:
    """Learner starting weights: the preserved self-play checkpoint unless one is named."""
    if args.initial_checkpoint is None and args.initial_checkpoint_sha256 is None:
        return args.selfplay_root / 'cycles/cycle_000206/train/checkpoint.json', CHECKPOINT_SHA
    sha = args.initial_checkpoint_sha256
    if args.initial_checkpoint is None or sha is None:
        raise ValueError('--initial-checkpoint and --initial-checkpoint-sha256 must be given together')
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('initial checkpoint SHA-256 must be 64 lowercase hex characters')
    return args.initial_checkpoint, sha


def corpus_windows(args, output: Path) -> list[tuple[str, str, Path, Path]]:
    """(since, until, raw root, corpus root) per collection window, in acquisition order."""
    windows = [(args.since, args.until, output / 'data/raw', output / 'data/corpus')]
    for number, window in enumerate(args.extra_window, 1):
        bounds = window.split(',')
        if len(bounds) != 2 or not all(bounds):
            raise ValueError('--extra-window must be SINCE,UNTIL')
        root = output / f'data/window_{number:02}'
        windows.append((bounds[0], bounds[1], root / 'raw', root / 'corpus'))
    return windows


def training_command(args, *, python: str, repo: Path, output: Path, corpora: list[Path],
                     checkpoint: Path, active: Path, commit: str) -> list[str]:
    # A named primary corpus keeps its fixed validation set, so the loss scale
    # stays comparable; the corpora prepared by this launch only add chunks.
    primary, extras = corpora[0], list(corpora[1:])
    if args.primary_corpus_manifest is not None:
        primary, extras = args.primary_corpus_manifest, list(corpora)
    command = [python, '-m', 'training.nnue.lc0_autopilot',
               '--corpus-manifest', str(primary), '--out-root', str(output / 'training'),
               '--initial-checkpoint', str(checkpoint), '--initial-active-model', str(active),
               '--source-commit', commit, '--piebot-dir', str(repo / 'PieBot'),
               '--hours', str(args.hours), '--device', 'cuda',
               '--disk-reserve-gib', str(args.min_free_gib),
               '--disk-capacity-gb', str(args.disk_capacity_gb)]
    for extra in extras:
        command += ['--extra-corpus-manifest', str(extra)]
    for flag, value in (('--learning-rate', args.learning_rate), ('--lr-gamma', args.lr_gamma),
                        ('--lr-epoch-positions', args.lr_epoch_positions),
                        ('--gate-movetime-ms', args.gate_movetime_ms)):
        if value is not None:
            command += [flag, str(value)]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out-root', type=Path, default=Path('/workspace/piebot_lc0_20260907'))
    parser.add_argument('--selfplay-root', type=Path, default=Path('/workspace/piebot_campaign_v8'))
    parser.add_argument('--hours', type=float, default=720)
    parser.add_argument('--since', default='2026-07-07')
    parser.add_argument('--until', default='2026-09-07')
    parser.add_argument('--extra-window', action='append', default=[], metavar='SINCE,UNTIL',
                        help='Further collection window, acquired and prepared after the first; '
                             'each whole window is downloaded before conversion evicts it')
    parser.add_argument('--min-free-gib', type=float, default=50)
    parser.add_argument('--download-concurrency', type=int, default=4,
                        help='Concurrent archive downloads (1-16); independent of preparation workers')
    parser.add_argument('--evict-raw', action=argparse.BooleanOptionalAction, default=False,
                        help='Evict each raw archive only after durable corpus receipt verification')
    parser.add_argument('--disk-capacity-gb', type=float, default=0,
                        help='Decimal disk capacity ceiling; 0 uses filesystem free space')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Acquire and prepare the corpus, then exit before training')
    parser.add_argument('--initial-checkpoint', type=Path, default=None)
    parser.add_argument('--initial-checkpoint-sha256', default=None)
    parser.add_argument('--primary-corpus-manifest', type=Path, default=None,
                        help='Existing frozen corpus that stays primary (and supplies validation)')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--lr-gamma', type=float, default=None)
    parser.add_argument('--lr-epoch-positions', type=int, default=None)
    parser.add_argument('--gate-movetime-ms', type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    capacity = decimal_gb_bytes(args.disk_capacity_gb)
    validate_download_concurrency(args.download_concurrency)
    corpus_windows(args, args.out_root)  # Reject malformed windows before any side effect.
    repo = args.repo.resolve()
    output = args.out_root.resolve()
    assert_separate_root(output, args.selfplay_root)
    checkpoint, checkpoint_sha = resolve_bootstrap(args)
    active = args.selfplay_root / 'cycles/cycle_000168/nnue_quant.nnue'
    validate_bootstrap(checkpoint, checkpoint_sha, active, ACTIVE_SHA)
    if args.primary_corpus_manifest is not None and not args.primary_corpus_manifest.is_file():
        raise ValueError('primary corpus manifest is missing')
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
    minimum = max(min_free, 1 if capacity is not None else 0)
    if available_bytes(ancestor, capacity_bytes=capacity) < minimum:
        raise ValueError('insufficient free disk reserve')
    print(json.dumps({'preflight': 'passed', 'source_commit': commit,
                      'checkpoint_sha256': checkpoint_sha, 'active_sha256': ACTIVE_SHA}), flush=True)
    if args.preflight_only:
        return 0
    # Nothing writes a source pin until bootstrap, binaries, CUDA and disk pass.
    pin_source(output, commit)
    from .autopilot import _single_instance_lock
    with _single_instance_lock(output / 'launcher.lock'):
        from .lc0_corpus import prepare_corpus
        corpora = []
        for since, until, raw, corpus_root in corpus_windows(args, output):
            manifest = raw / 'manifest.json'
            fetch_command = [
                sys.executable, '-m', 'training.nnue.fetch_lc0_bins', '--out', str(raw),
                '--manifest', str(manifest), '--since', since, '--until', until,
                '--suites', 'test91', '--limit-per-suite', '0', '--backend', 'curl',
                '--skip-existing', '--min-free-gib', str(args.min_free_gib),
                '--disk-capacity-gb', str(args.disk_capacity_gb),
                '--concurrency', str(args.download_concurrency)]
            if args.evict_raw:
                fetch_command += ['--eviction-corpus', str(corpus_root)]
            subprocess.run(fetch_command, cwd=repo, check=True)
            corpus_options = dict(since=since, until=until, min_free_bytes=min_free,
                                  workers=16, capacity_bytes=capacity)
            if args.evict_raw:
                corpus_options.update(evict_raw=True, raw_root=raw)
            corpora.append(prepare_corpus(manifest, corpus_root, **corpus_options))
        if args.prepare_only:
            print(json.dumps({'prepared': [str(corpus) for corpus in corpora]}), flush=True)
            return 0
        command = training_command(args, python=sys.executable, repo=repo, output=output, corpora=corpora,
                                   checkpoint=checkpoint, active=active, commit=commit)
        # Stay in the supervisor process group through every data/training stage.
        wait_for_training(command, cwd=repo, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
