#!/usr/bin/env python3
"""Convert LCZero BIN training data into JSONL shards for NNUE training."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zstd = None

from . import lc0_bin


def _looks_like_bin(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith('.bin') or lowered.endswith('.bin.gz') or lowered.endswith('.bin.zst')


def _serialize_sample(sample: dict) -> dict:
    data = dict(sample)
    data['policy_top'] = [{'move': move, 'p': float(prob)} for move, prob in sample['policy_top']]
    data['policy_top_canonical'] = [
        {'move': move, 'p': float(prob)} for move, prob in sample['policy_top_canonical']
    ]
    return data


def _iter_records_from_stream(name: str, stream: io.BufferedReader) -> Iterator[lc0_bin.V6Record]:
    lowered = name.lower()
    if lowered.endswith('.gz'):
        with gzip.GzipFile(fileobj=stream) as gz:
            yield from lc0_bin.iter_v6_records(gz)
    elif lowered.endswith('.zst') or lowered.endswith('.zstd'):
        if zstd is None:
            raise RuntimeError('zstandard package is required to decompress .zst files')
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(stream) as reader:
            yield from lc0_bin.iter_v6_records(io.BufferedReader(reader))
    else:
        yield from lc0_bin.iter_v6_records(stream)


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = max(1, shard_size)
        self.current_index = 0
        self.current_count = 0
        self.fp: Optional[io.TextIOBase] = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.fp:
            self.fp.close()
        shard_path = self.out_dir / f'shard_{self.current_index:06}.jsonl'
        self.fp = shard_path.open('w', encoding='utf-8')
        self.current_index += 1
        self.current_count = 0

    def write(self, sample: dict) -> None:
        if self.fp is None or self.current_count >= self.shard_size:
            self._open()
        json.dump(sample, self.fp)
        self.fp.write('\n')
        self.current_count += 1

    def close(self) -> None:
        if self.fp:
            self.fp.close()
            self.fp = None


def process_bin_stream(stream: io.BufferedReader, source_name: str, writer: ShardWriter,
                       top_policy: int, remaining: Optional[int]) -> int:
    processed = 0
    for record in _iter_records_from_stream(source_name, stream):
        sample = lc0_bin.record_to_sample(record, top_policy=top_policy)
        writer.write(_serialize_sample(sample))
        processed += 1
        if remaining is not None and processed >= remaining:
            break
    return processed


def process_tar(path: Path, writer: ShardWriter, top_policy: int, remaining: Optional[int]) -> int:
    processed = 0
    with tarfile.open(path, 'r') as archive:
        for member in archive:
            if remaining is not None and processed >= remaining:
                break
            if not member.isfile() or not _looks_like_bin(member.name):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                buffered = io.BufferedReader(extracted)
                processed += process_bin_stream(buffered, member.name, writer, top_policy,
                                                None if remaining is None else remaining - processed)
    return processed


def process_single_path(path: Path, writer: ShardWriter, top_policy: int,
                        remaining: Optional[int]) -> int:
    if remaining is not None and remaining <= 0:
        return 0
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.tar'):
        return process_tar(path, writer, top_policy, remaining)
    if suffixes.endswith('.zst') or suffixes.endswith('.zstd'):
        if zstd is None:
            raise RuntimeError('zstandard package is required to decode compressed BIN files')
        with path.open('rb') as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            buffered = io.BufferedReader(reader)
            return process_bin_stream(buffered, path.name, writer, top_policy, remaining)
    if suffixes.endswith('.bin') or suffixes.endswith('.bin.gz'):
        count = 0
        for record in lc0_bin.make_record_stream(path):
            sample = lc0_bin.record_to_sample(record, top_policy=top_policy)
            writer.write(_serialize_sample(sample))
            count += 1
            if remaining is not None and count >= remaining:
                break
        return count
    if path.is_file() and _looks_like_bin(path.name):
        count = 0
        for record in lc0_bin.make_record_stream(path):
            sample = lc0_bin.record_to_sample(record, top_policy=top_policy)
            writer.write(_serialize_sample(sample))
            count += 1
            if remaining is not None and count >= remaining:
                break
        return count
    return 0


def process_inputs(inputs: Iterable[Path], writer: ShardWriter, glob_pattern: str,
                   top_policy: int, max_records: Optional[int]) -> int:
    total = 0
    for input_path in inputs:
        if input_path.is_dir():
            files = sorted(input_path.rglob(glob_pattern))
            for file_path in files:
                remaining = None if max_records is None else max_records - total
                if remaining is not None and remaining <= 0:
                    return total
                total += process_single_path(file_path, writer, top_policy, remaining)
        else:
            remaining = None if max_records is None else max_records - total
            if remaining is not None and remaining <= 0:
                return total
            total += process_single_path(input_path, writer, top_policy, remaining)
    return total


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', nargs='+', type=Path, required=True,
                    help='Input BIN files, tar archives, or directories')
    ap.add_argument('--out', type=Path, required=True, help='Output directory for JSONL shards')
    ap.add_argument('--glob', default='*.bin*', help='Glob pattern when scanning directories')
    ap.add_argument('--shard-size', type=int, default=200_000,
                    help='Positions per output shard')
    ap.add_argument('--top-policy', type=int, default=8,
                    help='Number of top policy moves to store per record')
    ap.add_argument('--max-records', type=int, default=0,
                    help='Optional cap on total records processed (0 = unlimited)')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    writer = ShardWriter(args.out, args.shard_size)
    max_records = args.max_records if args.max_records > 0 else None
    try:
        total = process_inputs(args.inputs, writer, args.glob, args.top_policy, max_records)
    finally:
        writer.close()
    print(f'Processed {total} records')


if __name__ == '__main__':
    main()
