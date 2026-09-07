"""Build a frozen, resumable NNUE corpus from dated official LCZero game archives.

Archive dates select source files; tar member mtimes are the game's collection
timestamp (not a verified played date). Raw archives remain intact. Work is
committed one archive at a time, so only an unfinished archive repeats on resume.
"""

from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
import contextlib
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import tarfile
from typing import BinaryIO, Iterator

from . import lc0_bin
from .fetch_lc0_bins import sha256_file, utc_bound, write_json_atomic

SCHEMA = 'piebot-lc0-corpus-v1'
GAME_NAME = re.compile(r'^(?:training\.[A-Za-z0-9_-]+|game_[A-Za-z0-9_-]+)(?:\.bin)?(?:\.gz)?$')
_HEADER = struct.Struct('<II')
_VALUE_BODY = struct.Struct('<104Q8B15fIHHfI')


def iter_value_records(stream: BinaryIO) -> Iterator[lc0_bin.V6Record]:
    """Decode only value/board fields, skipping 1,858 unused policy floats."""
    while True:
        raw = stream.read(lc0_bin.RECORD_SIZE)
        if not raw:
            return
        if len(raw) != lc0_bin.RECORD_SIZE:
            raise ValueError('Truncated V6 record encountered')
        version, fmt = _HEADER.unpack_from(raw)
        if version != 6:
            raise ValueError(f'unsupported LCZero record version {version}; expected 6')
        body = _VALUE_BODY.unpack_from(raw, 8 + 1858 * 4)
        yield lc0_bin.V6Record(version, fmt, [], list(body[:104]), *body[104:])


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def is_holdout_game(run_id: str, game_id: str, seed: int, fraction: float) -> bool:
    """The same whole game always belongs to the same split across archive copies."""
    return int(_hash([seed, run_id, game_id]), 16) < int(fraction * 2**256)


def _space(path: Path, reserve: int) -> None:
    if shutil.disk_usage(path).free < reserve:
        raise OSError('corpus preparation reached the protected disk reserve')


class _Chunks:
    def __init__(self, stage: Path, prefix: str, limit: int, reserve: int):
        self.stage, self.prefix, self.limit, self.reserve = stage, prefix, limit, reserve
        self.fp = None
        self.count = 0
        self.entries: list[dict] = []

    def write(self, sample: dict) -> None:
        if self.fp is None or self.count == self.limit:
            self.close()
            _space(self.stage, self.reserve)
            path = self.stage / f'{self.prefix}_{len(self.entries):06}.jsonl.gz'
            self.fp = gzip.open(path, 'wt', encoding='utf-8', compresslevel=1)
            self.entries.append({'name': path.name, 'positions': 0})
            self.count = 0
        self.fp.write(json.dumps(sample, separators=(',', ':'), allow_nan=False) + '\n')
        self.count += 1
        self.entries[-1]['positions'] = self.count
        if self.count % 4096 == 0:
            _space(self.stage, self.reserve)

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None
            path = self.stage / self.entries[-1]['name']
            with path.open('rb') as fh:
                os.fsync(fh.fileno())
            self.entries[-1]['sha256'] = sha256_file(path)


def _sample(record: lc0_bin.V6Record) -> dict | None:
    if record.version != 6:
        raise ValueError(f'unsupported LCZero record version {record.version}; expected 6')
    if record.input_format not in {1, 2, 3, 4, 5}:
        raise ValueError(f'unsupported standard-chess LCZero input format {record.input_format}')
    if record.invariance_info & (1 << 6):
        return None
    if any(not math.isfinite(q) or not -1 <= q <= 1
           for q in (record.best_q, record.result_q, record.root_q)):
        return None
    fen_info = lc0_bin.build_fen_from_planes(record, lc0_bin.decode_planes(record))
    # The NNUE feature set is standard chess, not Chess960 or Armageddon.
    if any(ch not in 'KQkq-' for ch in fen_info['castling'].to_fen()):
        raise ValueError('Chess960 castling rights are unsupported in this corpus')
    import chess
    board = chess.Board(fen_info['fen'])
    if not board.is_valid():
        return None
    perspective = -1 if fen_info['black_to_move'] else 1
    result = record.result_q * perspective
    return {'fen': fen_info['fen'], 'best_q': record.best_q * perspective,
            'result_q': result, 'root_q': record.root_q * perspective,
            'result': 1 if result > 0 else (-1 if result < 0 else 0),
            'best_q_side_to_move': record.best_q,
            'result_q_side_to_move': record.result_q,
            'value_perspective': 'white',
            'outcome_valid': not bool(record.invariance_info & (1 << 4))}


def _archive_entries(raw: dict) -> list[dict]:
    if raw.get('files') is not None:
        return raw['files']
    entries = []
    for suite, files in raw.get('suites', {}).items():
        if isinstance(files, dict):
            raise ValueError(f'failed source suite listing: {suite}')
        entries.extend(dict(entry, suite=suite) for entry in files)
    return entries


def _decode_game_payload(payload: bytes, compressed: bool) -> tuple[list[tuple[int, dict]], int]:
    """Bound one chess game's expansion before sending rows back to the writer."""
    samples, rejected = [], 0
    with contextlib.ExitStack() as stack:
        source = io.BytesIO(payload)
        stream = stack.enter_context(gzip.GzipFile(fileobj=source)) if compressed else source
        for ply, record in enumerate(iter_value_records(stream)):
            # Even theoretical maximum-length standard games fit below this bound.
            if ply >= 32768:
                raise ValueError('LCZero game exceeds maximum supported chess game length')
            sample = _sample(record)
            if sample is None:
                rejected += 1
            else:
                samples.append((ply, sample))
    return samples, rejected


def _convert_archive(entry: dict, stage: Path, conn: sqlite3.Connection, config: dict) -> dict:
    lower, upper = utc_bound(config['since']).timestamp(), utc_bound(config['until']).timestamp()
    run = 'lc0:' + entry.get('suite', 'test91').strip('/')
    stats = {'games': 0, 'holdout_games': 0, 'training_positions': 0, 'holdout_positions': 0,
             'duplicate_games': 0, 'outside_window_games': 0, 'rejected_records': 0}
    dates: set[str] = set()
    training = _Chunks(stage, 'train', config['chunk_positions'], config['min_free_bytes'])
    holdout = _Chunks(stage, 'holdout', config['chunk_positions'], config['min_free_bytes'])
    pending = deque()

    def emit(game: str, collected: str, is_holdout: bool, decoded: tuple) -> None:
        samples, rejected = decoded
        stats['rejected_records'] += rejected
        for ply, sample in samples:
            sample.update(run_id=run, game_id=game, ply=ply,
                          record_id=_hash([run, game, ply]),
                          collected_at=collected, source_archive_sha256=entry['sha256'])
            (holdout if is_holdout else training).write(sample)
            stats['holdout_positions' if is_holdout else 'training_positions'] += 1

    try:
        # Stream mode avoids tarfile retaining millions of member headers in RAM.
        with contextlib.ExitStack() as resources:
            workers = config['workers']
            pool = (resources.enter_context(concurrent.futures.ProcessPoolExecutor(max_workers=workers))
                    if workers > 1 else None)
            archive = resources.enter_context(tarfile.open(entry['dest'], 'r|'))
            for member in archive:
                # Python <3.13 lacks TarFile(stream=True). This also bounds memory
                # when a large backfill archive contains only out-of-window games.
                archive.members.clear()
                game = Path(member.name).name
                if not member.isfile() or not GAME_NAME.fullmatch(game):
                    continue
                if not math.isfinite(member.mtime):
                    raise ValueError(f'invalid collection timestamp: {member.name}')
                if not lower <= member.mtime < upper:
                    stats['outside_window_games'] += 1
                    continue
                game_key = _hash([run, game])
                if conn.execute('SELECT 1 FROM games WHERE game_key=?', (game_key,)).fetchone():
                    stats['duplicate_games'] += 1
                    continue
                conn.execute('INSERT INTO games(game_key) VALUES(?)', (game_key,))
                is_holdout = is_holdout_game(run, game, config['seed'], config['validation_fraction'])
                stats['games'] += 1
                stats['holdout_games'] += int(is_holdout)
                collected = datetime.fromtimestamp(member.mtime, timezone.utc).isoformat()
                dates.add(collected[:10])
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f'cannot read tar member: {member.name}')
                with extracted:
                    if member.size > 64 * 1024**2:
                        raise ValueError(f'LCZero game member exceeds 64 MiB: {member.name}')
                    payload = extracted.read()
                if pool is None:
                    emit(game, collected, is_holdout, _decode_game_payload(payload, game.endswith('.gz')))
                else:
                    pending.append((game, collected, is_holdout,
                                    pool.submit(_decode_game_payload, payload, game.endswith('.gz'))))
                    # Archive/game order is preserved, independently of worker completion.
                    if len(pending) >= workers * 2:
                        name, stamp, split, future = pending.popleft()
                        emit(name, stamp, split, future.result())
            while pending:
                name, stamp, split, future = pending.popleft()
                emit(name, stamp, split, future.result())
    finally:
        training.close()
        holdout.close()
    return {'chunks': training.entries, 'holdout': holdout.entries, 'stats': stats,
            'collection_dates': sorted(dates)}


def _validation(archives: list[tuple[Path, dict]], dest: Path, size: int, seed: int) -> dict:
    # Bottom-k hash reservoir is deterministic irrespective of archive traversal order.
    heap: list[tuple[int, str, str]] = []
    for directory, info in archives:
        for entry in info['holdout']:
            with gzip.open(directory / entry['name'], 'rt', encoding='utf-8') as fh:
                for line in fh:
                    record = json.loads(line)
                    rank = int(_hash([seed, 'validation', record['record_id']]), 16)
                    item = (-rank, record['record_id'], line)
                    if len(heap) < size:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
    part = Path(str(dest) + '.part')
    try:
        with part.open('w', encoding='utf-8') as fh:
            for _rank, _record_id, line in sorted(heap, key=lambda item: -item[0]):
                fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return {'path': str(dest), 'positions': len(heap), 'sha256': sha256_file(dest)}


def prepare_corpus(raw_manifest: Path, out_dir: Path, *, since: str, until: str,
                   chunk_positions: int = 700000, validation_fraction: float = .01,
                   validation_samples: int = 100000, seed: int = 20260907,
                   min_free_bytes: int = 50 * 1024**3, workers: int = 8) -> Path:
    """Return a complete frozen corpus; completed archives survive interruptions."""
    if chunk_positions < 1 or not 0 <= validation_fraction < 1 or validation_samples < 1 or workers < 1:
        raise ValueError('invalid chunk size or validation settings')
    raw = json.loads(raw_manifest.read_text())
    if raw.get('failures'):
        raise ValueError('raw manifest has acquisition failures')
    entries = _archive_entries(raw)
    if not entries:
        raise ValueError('raw manifest contains no archives')
    lower, upper = utc_bound(since), utc_bound(until, upper=True)
    if raw.get('until'):
        upper = min(upper, utc_bound(raw['until']))
    if lower >= upper:
        raise ValueError('empty collection timestamp window')
    for entry in entries:
        if not entry.get('sha256') or len(entry['sha256']) != 64:
            raise ValueError(f"archive has no verified checksum: {entry.get('dest')}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {'schema': SCHEMA, 'since': lower.isoformat(), 'until': upper.isoformat(),
              'chunk_positions': chunk_positions, 'validation_fraction': validation_fraction,
              'validation_samples': validation_samples, 'seed': seed,
              'sources': [{k: entry.get(k) for k in ('url', 'sha256', 'size', 'suite')} for entry in entries]}
    identity = _hash(config)
    config_path = out_dir / 'identity.json'
    if config_path.exists():
        if json.loads(config_path.read_text())['corpus_id'] != identity:
            raise ValueError('corpus identity changed; use a new output root')
    else:
        write_json_atomic(config_path, dict(config, corpus_id=identity))
    manifest_path = out_dir / 'corpus_manifest.json'
    config['min_free_bytes'] = min_free_bytes
    config['workers'] = workers
    import fcntl
    with (out_dir / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conn = sqlite3.connect(out_dir / 'progress.sqlite3')
        try:
            conn.execute('CREATE TABLE IF NOT EXISTS games(game_key TEXT PRIMARY KEY)')
            conn.execute('CREATE TABLE IF NOT EXISTS archives(archive_key TEXT PRIMARY KEY, directory TEXT, metadata TEXT)')
            conn.commit()
            all_archives = []
            for index, entry in enumerate(entries):
                archive_key = _hash([entry.get('url'), entry['sha256']])
                done = conn.execute('SELECT directory, metadata FROM archives WHERE archive_key=?', (archive_key,)).fetchone()
                if done:
                    directory, info_json = done
                    info = json.loads(info_json)
                    for cached in info['chunks'] + info['holdout']:
                        if sha256_file(Path(directory) / cached['name']) != cached['sha256']:
                            raise ValueError(f'corpus cache checksum mismatch: {cached["name"]}')
                    all_archives.append((Path(directory), info))
                    continue
                source = Path(entry['dest'])
                if sha256_file(source) != entry['sha256']:
                    raise ValueError(f'archive checksum mismatch: {source}')
                _space(out_dir, min_free_bytes)
                directory = out_dir / 'archives' / f'{index:06}_{archive_key[:16]}'
                stage = directory.with_name(directory.name + '.part')
                # Only these reproducible outputs from an unfinished transaction are removed.
                for incomplete in (stage, directory):
                    if incomplete.exists():
                        shutil.rmtree(incomplete)
                stage.mkdir(parents=True)
                try:
                    info = _convert_archive(entry, stage, conn, config)
                    write_json_atomic(stage / 'archive.json', info)
                    os.replace(stage, directory)
                    conn.execute('INSERT INTO archives VALUES(?,?,?)',
                                 (archive_key, str(directory), json.dumps(info, sort_keys=True)))
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                all_archives.append((directory, info))
                print(f"corpus archive {index+1}/{len(entries)}: {info['stats']}", flush=True)
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text())
                val = existing['validation']
                if sha256_file(Path(val['path'])) != val['sha256']:
                    raise ValueError('validation cache checksum mismatch')
                return manifest_path
            validation = _validation(all_archives, out_dir / 'validation.jsonl', validation_samples, seed)
            if validation_fraction > 0 and validation['positions'] == 0:
                raise ValueError('no valid holdout positions; the corpus cannot validate training')
            chunks, stats, dates = [], {}, set()
            for directory, info in all_archives:
                chunks.extend({'path': str(directory / chunk['name']), 'positions': chunk['positions'],
                               'sha256': chunk['sha256']} for chunk in info['chunks'])
                for key, value in info['stats'].items():
                    stats[key] = stats.get(key, 0) + value
                dates.update(info['collection_dates'])
            if not chunks:
                raise ValueError('no valid training positions in requested collection window')
            requested_dates, day = [], lower.date()
            while day <= (upper - timedelta(microseconds=1)).date():
                requested_dates.append(day.isoformat())
                day += timedelta(days=1)
            manifest = {'schema': SCHEMA, 'complete': True, 'corpus_id': identity,
                        'since': lower.isoformat(), 'until': upper.isoformat(),
                        'date_basis': 'tar_member_mtime', 'seed': seed,
                        'raw_manifest_sha256': sha256_file(raw_manifest),
                        'chunks': chunks, 'validation': validation, 'stats': stats,
                        'collection_dates': sorted(dates),
                        'missing_collection_dates': sorted(set(requested_dates) - dates)}
            write_json_atomic(manifest_path, manifest)
            return manifest_path
        finally:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--since', required=True)
    parser.add_argument('--until', required=True)
    parser.add_argument('--chunk-positions', type=int, default=700000)
    parser.add_argument('--validation-fraction', type=float, default=.01)
    parser.add_argument('--validation-samples', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--min-free-gib', type=float, default=50.)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(prepare_corpus(args.raw_manifest, args.out, since=args.since, until=args.until,
                         chunk_positions=args.chunk_positions, validation_fraction=args.validation_fraction,
                         validation_samples=args.validation_samples, seed=args.seed,
                         min_free_bytes=int(args.min_free_gib * 1024**3), workers=args.workers))


if __name__ == '__main__':
    main()
