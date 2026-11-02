"""Threat input features: reference implementation (plain Python).

A threat feature says that one piece attacks another, seen from one side
(the "perspective"). It is the tuple

    (attacker colour, attacker type, from, to, attacked colour, attacked type)

with colours relative to the perspective (0 = its own pieces) and squares
oriented the same way as the piece features in train_v3.piece_features:
ranks flipped for Black, files mirrored when the perspective's king stands
on files e-h. A feature is active when the attacker's attack set, with real
occupancy (sliders stop at the first piece), contains the attacked piece's
square. Defending one's own piece counts as well.

Which pairs exist follows Stockfish's FullThreats (src/nnue/features/
full_threats.cpp, SFNNv10+), reimplemented here, giving 59,808 features:

* kings neither attack nor get attacked;
* pawns attack only knights and rooks;
* bishops and rooks attack pawns, knights, bishops and rooks;
* knights and queens attack pawns, knights, bishops, rooks and queens;
* when attacker and attacked have the same type (except own pawn onto own
  pawn, which never exists here), only the direction with oriented
  from > to counts, so a mutual attack is one feature, not two.

Index layout: attacker blocks in the order own P N B R Q K, enemy P N B R Q
K. Inside a block, one slot per (attacked colour, attacked type) in the
order above, and inside a slot one entry per (from, to) the empty-board
attack mask allows, from-major, to ascending. Pawns use only ranks 2-7 and
the attack direction of their (relative) colour: own pawns attack towards
rank 8.

Evaluation only; nothing here produces training labels.
"""
from __future__ import annotations

import chess

N_THREAT_FEATURES = 59808
# 30 non-king pieces, each attacking at most 8 pieces (one per ray or knight
# jump; a pawn at most 2), so 240 can't be exceeded; 256 matches Stockfish.
MAX_ACTIVE = 256

P, N, B, R, Q, K = range(6)
# TARGETS[attacker type] = attacked types, in slot order.
TARGETS = {P: (N, R), N: (P, N, B, R, Q), B: (P, N, B, R), R: (P, N, B, R), Q: (P, N, B, R, Q), K: ()}


def _on(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def empty_attacks(pt: int, sq: int, up: bool = True) -> list[int]:
    """Attacked squares on an empty board (pawns: towards rank 8 if ``up``)."""
    f, r = sq & 7, sq >> 3
    if pt == P:
        if not 1 <= r <= 6:
            return []
        dr = 1 if up else -1
        return sorted((r + dr) * 8 + f + df for df in (-1, 1) if _on(f + df, r + dr))
    if pt == N:
        steps = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
        return sorted((r + b) * 8 + f + a for a, b in steps if _on(f + a, r + b))
    if pt == K:
        return sorted((r + b) * 8 + f + a for a in (-1, 0, 1) for b in (-1, 0, 1)
                      if (a, b) != (0, 0) and _on(f + a, r + b))
    dirs = []
    if pt in (B, Q):
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    if pt in (R, Q):
        dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
    out = []
    for a, b in dirs:
        ff, rr = f + a, r + b
        while _on(ff, rr):
            out.append(rr * 8 + ff)
            ff, rr = ff + a, rr + b
    return sorted(out)


def _build():
    # Per attacker type: base offset of each from-square, and the rank of
    # each to-square inside that from-square's mask (shared by both pawn
    # directions: a pawn has the same number of attacks either way).
    sq_base, to_rank, per_type = {}, {}, {}
    for pt in range(6):
        off = 0
        for frm in range(64):
            sq_base[pt, frm] = off
            for up in (True, False):
                for i, to in enumerate(empty_attacks(pt, frm, up)):
                    to_rank[pt, frm, to] = i
            off += len(empty_attacks(pt, frm))
        per_type[pt] = off
    block, off = {}, 0
    for ca in (0, 1):
        for pt in range(6):
            block[ca, pt] = off
            off += 2 * len(TARGETS[pt]) * per_type[pt]
    table = {}
    for ca in (0, 1):
        for pa in range(6):
            for frm in range(64):
                # Own pawns (relative colour 0) attack upwards, enemy pawns downwards.
                for to in empty_attacks(pa, frm, up=(ca == 0)):
                    for cb in (0, 1):
                        for slot, pb in enumerate(TARGETS[pa]):
                            s = cb * len(TARGETS[pa]) + slot
                            table[ca, pa, frm, to, cb, pb] = (block[ca, pa] + s * per_type[pa]
                                                              + sq_base[pa, frm] + to_rank[pa, frm, to])
    return sq_base, to_rank, per_type, block, off, table


SQ_BASE, TO_RANK, PER_TYPE, BLOCK, _TOTAL, TABLE = _build()
assert _TOTAL == N_THREAT_FEATURES, _TOTAL
SLOT = {(pa, pb): i for pa in range(6) for i, pb in enumerate(TARGETS[pa])}


def index(ca: int, pa: int, frm: int, to: int, cb: int, pb: int) -> int:
    """Feature index for oriented squares and relative colours (no exclusion test)."""
    s = cb * len(TARGETS[pa]) + SLOT[pa, pb]
    return BLOCK[ca, pa] + s * PER_TYPE[pa] + SQ_BASE[pa, frm] + TO_RANK[pa, frm, to]


def orientation(board: chess.Board, persp: int) -> int:
    """XOR mask taking a real square to the perspective's oriented square."""
    king = board.king(chess.WHITE if persp == 0 else chess.BLACK)
    k = king if persp == 0 else king ^ 56
    return (56 if persp else 0) ^ (7 if (k & 7) >= 4 else 0)


def threat_features(board: chess.Board, persp: int) -> list[int]:
    """Sorted active threat feature indices for perspective 0 (White) or 1 (Black)."""
    flip = orientation(board, persp)
    me = chess.WHITE if persp == 0 else chess.BLACK
    out = []
    for frm, a in board.piece_map().items():
        pa = a.piece_type - 1
        if not TARGETS[pa]:
            continue
        ca = 0 if a.color == me else 1
        for to in board.attacks(frm):
            b = board.piece_at(to)
            if b is None:
                continue
            pb = b.piece_type - 1
            if (pa, pb) not in SLOT:
                continue
            fo, tt = frm ^ flip, to ^ flip
            if pa == pb and fo < tt:
                continue
            out.append(index(ca, pa, fo, tt, 0 if b.color == me else 1, pb))
    return sorted(out)


def write_threat_net(v5_path, out_path, threat_w) -> None:
    """Write a version-6 file: a version-5 (pairwise) file plus i8 threat rows.

    ``threat_w`` is an int8 array of shape (N_THREAT_FEATURES, hidden).
    """
    import struct
    from pathlib import Path

    import numpy as np

    raw = bytearray(Path(v5_path).read_bytes())
    assert raw[:8] == b"PIENNQ03", "not a PIENNQ03 file"
    (version,) = struct.unpack_from("<I", raw, 8)
    assert version == 5, f"expected a version-5 file, got {version}"
    w = np.ascontiguousarray(threat_w, dtype=np.int8)
    assert w.ndim == 2 and w.shape[0] == N_THREAT_FEATURES, w.shape
    struct.pack_into("<I", raw, 8, 6)
    raw += struct.pack("<I", N_THREAT_FEATURES) + w.tobytes()
    Path(out_path).write_bytes(bytes(raw))
