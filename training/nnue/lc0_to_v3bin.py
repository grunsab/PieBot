#!/usr/bin/env python3
"""Convert LCZero V6 training archives into compact fixed-size records.

One record is 32 bytes (``V3_DTYPE``): the board as an occupancy bitboard
(python-chess square order, a1 = bit 0) plus one 4-bit piece code per occupied
square in ascending square order (``color * 6 + piece_type - 1``, white 0..5,
black 6..11), the side to move, LCZero's best Q and the game result, both from
the side to move's point of view. Features are computed from this on the GPU
by ``train_v3.py``, so the data stays small enough for 10^9 positions.

Filters (each counted in the stats): records marked for deletion or with
non-finite Q, invalid boards, the first ``--skip-plies`` plies of each game,
positions in check, and positions whose LCZero best move is a capture.
Whole games go to the hold-out split by a stable hash of the archive member.

Usage::

    python3 training/nnue/lc0_to_v3bin.py --out-dir /workspace/lc0/bin \
        --workers 28 /workspace/lc0/raw/*.tar
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import tarfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc0_bin  # noqa: E402

V3_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", (16,)),
    ("stm", "u1"),        # 0 white, 1 black
    ("result", "i1"),     # -1 / 0 / +1 for the side to move
    ("q", "<i2"),         # best Q * 32767, side to move
    ("ply", "<u2"),
    ("flags", "u1"),      # bit0: outcome valid
    ("_pad", "u1"),
])
assert V3_DTYPE.itemsize == 32

RECORD_BYTES = 8356


def encode_board(board) -> tuple[int, bytes]:
    """Occupancy and packed nibbles for a python-chess board."""
    occ = 0
    codes = []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None:
            continue
        occ |= 1 << sq
        codes.append((0 if p.color else 1) * 6 + p.piece_type - 1)
    if len(codes) > 32:
        raise ValueError("more than 32 pieces")
    packed = bytearray(16)
    for i, c in enumerate(codes):
        packed[i // 2] |= c << (4 * (i % 2))
    return occ, bytes(packed)


def decode_board(occ: int, pcs: bytes) -> dict:
    """Inverse of ``encode_board``: {square: code} (used by tests)."""
    out = {}
    i = 0
    for sq in range(64):
        if occ >> sq & 1:
            out[sq] = (pcs[i // 2] >> (4 * (i % 2))) & 0xF
            i += 1
    return out


def is_holdout(member: str, fraction: float) -> bool:
    h = int.from_bytes(hashlib.sha256(member.encode()).digest()[:8], "little")
    return (h % 1_000_000) < fraction * 1_000_000


def convert_game(payload: bytes, skip_plies: int, stats: dict) -> list[tuple]:
    import chess

    rows = []
    data = gzip.decompress(payload)
    n = len(data) // RECORD_BYTES
    for ply in range(n):
        stats["seen"] += 1
        rec = lc0_bin.parse_v6_record(data[ply * RECORD_BYTES:(ply + 1) * RECORD_BYTES])
        if rec.version != 6 or rec.input_format not in {1, 2, 3, 4, 5}:
            stats["bad_format"] += 1
            continue
        if rec.invariance_info & (1 << 6):
            stats["deleted"] += 1
            continue
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (rec.best_q, rec.result_q)):
            stats["bad_q"] += 1
            continue
        if ply < skip_plies:
            stats["early"] += 1
            continue
        try:
            info = lc0_bin.build_fen_from_planes(rec, lc0_bin.decode_planes(rec))
            board = chess.Board(info["fen"])
        except Exception:
            stats["bad_board"] += 1
            continue
        if not board.is_valid():
            stats["bad_board"] += 1
            continue
        if board.is_check():
            stats["in_check"] += 1
            continue
        transform = rec.invariance_info & 0x7 if lc0_bin._is_canonical(rec.input_format) else 0
        try:
            best = lc0_bin.move_from_nn_index(rec.best_idx, transform)
            if info["black_to_move"]:
                best = lc0_bin.flip_move_uci(best)
            mv = chess.Move.from_uci(best)
            # Castling written as king-takes-rook lands on an own piece, which
            # is_capture does not count, so no legality pass is needed here.
            if board.is_capture(mv):
                stats["best_capture"] += 1
                continue
        except Exception:
            stats["no_best_move"] += 1
        occ, pcs = encode_board(board)
        stm = 0 if board.turn else 1
        outcome_valid = not bool(rec.invariance_info & (1 << 4))
        res = rec.result_q
        result = 1 if res > 1e-6 else (-1 if res < -1e-6 else 0)
        rows.append((occ, pcs, stm, result, int(round(rec.best_q * 32767)), min(ply, 65535),
                     1 if outcome_valid else 0))
        stats["kept"] += 1
    return rows


def convert_archive(tar_path: str, out_dir: str, skip_plies: int, holdout: float) -> dict:
    stats = {k: 0 for k in ("games", "bad_games", "seen", "kept", "bad_format", "deleted", "bad_q",
                            "early", "bad_board", "in_check", "best_capture", "no_best_move")}
    train, hold = [], []
    name = Path(tar_path).stem
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".gz"):
                continue
            try:
                rows = convert_game(t.extractfile(m).read(), skip_plies, stats)
            except (EOFError, OSError, gzip.BadGzipFile, zlib.error, ValueError):
                stats["bad_games"] += 1
                continue
            stats["games"] += 1
            (hold if is_holdout(m.name, holdout) else train).extend(rows)
    out = Path(out_dir)
    for split, rows in (("train", train), ("holdout", hold)):
        arr = np.zeros(len(rows), dtype=V3_DTYPE)
        for i, (occ, pcs, stm, result, q, ply, flags) in enumerate(rows):
            arr[i]["occ"] = occ
            arr[i]["pcs"] = np.frombuffer(pcs, dtype=np.uint8)
            arr[i]["stm"] = stm
            arr[i]["result"] = result
            arr[i]["q"] = q
            arr[i]["ply"] = ply
            arr[i]["flags"] = flags
        (out / split).mkdir(parents=True, exist_ok=True)
        tmp = out / split / f"{name}.bin.tmp"
        arr.tofile(tmp)
        tmp.replace(out / split / f"{name}.bin")
    stats["archive"] = name
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.01)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in a.archives if not (out / "train" / f"{Path(p).stem}.bin").exists()]
    total = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool, open(out / "stats.jsonl", "a") as log:
        futs = {pool.submit(convert_archive, p, str(out), a.skip_plies, a.holdout): p for p in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
            except Exception as e:  # keep going; record the failure
                s = {"archive": Path(futs[f]).stem, "error": repr(e)}
            log.write(json.dumps(s) + "\n")
            log.flush()
            for k, v in s.items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
    print(json.dumps(total))


if __name__ == "__main__":
    main()
