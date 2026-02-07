#!/usr/bin/env python3
"""
Convert PGN files (e.g., LCZero training PGNs) into training JSONL suitable for NNUE training.

Each output line is a JSON object with at least:
  {"fen": "...", "result": 1|0|-1}

Options:
  - Shard output into multiple JSONL files (by number of positions per shard)
  - Sample every K plies to reduce volume
  - Limit games processed
  - Read .pgn or .pgn.zst (requires zstandard)

Usage:
  python training/nnue/process_pgns.py --in-dir data/lc0_pgns/test80 \
    --out data/nnue_jsonl/test80 --shard-size 200000 --sample-every 1 --max-games 0

Dependencies:
  - python-chess (pip install python-chess)
  - zstandard (pip install zstandard) for .zst
  - tqdm (optional) for progress
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Optional

try:
    import tqdm  # type: ignore
    def _tqdm(it, **kw):
        return tqdm.tqdm(it, **kw)
except Exception:
    def _tqdm(it, **kw):
        return it

def open_text(path: Path):
    if path.suffix.lower() in (".zst", ".zstd"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(open(path, 'rb'))
    return open(path, 'rb')

def iter_games_from_pgn(path: Path):
    import chess.pgn
    with open_text(path) as f:
        # Wrap in TextIO
        import io
        if hasattr(f, 'read') and isinstance(f.read(0), (bytes, bytearray)):
            tf = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
        else:
            tf = f
        while True:
            game = chess.pgn.read_game(tf)
            if game is None:
                break
            yield game

def game_result_to_wdl(headers: dict) -> Optional[int]:
    res = headers.get('Result')
    if res == '1-0':
        return 1
    if res == '0-1':
        return -1
    if res == '1/2-1/2':
        return 0
    return None

def process_pgn(path: Path, out_dir: Path, shard_size: int, sample_every: int, max_games: int = 0):
    import chess
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    in_shard = 0
    shard_file: Optional[Path] = None
    shard_fp = None

    def open_new_shard() -> tuple[Path, object]:
        nonlocal shard_index
        shard_path = out_dir / f"shard_{shard_index:06}.jsonl"
        shard_index += 1
        return shard_path, open(shard_path, 'w', encoding='utf-8')

    processed_games = 0
    for game in _tqdm(iter_games_from_pgn(path), desc=f"{path.name}"):
        if max_games and processed_games >= max_games:
            break
        result = game_result_to_wdl(game.headers)
        if result is None:
            continue
        board = chess.Board()
        ply = 0
        for move in game.mainline_moves():
            if ply % max(1, sample_every) == 0:
                if shard_fp is None or in_shard >= shard_size:
                    if shard_fp is not None:
                        shard_fp.close()
                    shard_file, shard_fp = open_new_shard()
                    in_shard = 0
                rec = {"fen": board.fen(), "result": result}
                shard_fp.write(json.dumps(rec) + "\n")
                in_shard += 1
            board.push(move)
            ply += 1
        processed_games += 1
    if shard_fp is not None:
        shard_fp.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--glob', default='*.pgn*', help='Glob pattern for PGNs')
    ap.add_argument('--shard-size', type=int, default=200_000)
    ap.add_argument('--sample-every', type=int, default=1)
    ap.add_argument('--max-games', type=int, default=0, help='0 = unlimited')
    args = ap.parse_args()

    paths = sorted(args.in_dir.glob(args.glob))
    if not paths:
        print("No PGN files matched", args.in_dir / args.glob)
        return
    for p in paths:
        process_pgn(p, args.out, args.shard_size, args.sample_every, args.max_games)

if __name__ == '__main__':
    main()
