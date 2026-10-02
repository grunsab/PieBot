"""Training-time position filters for prepared LCZero corpus chunks.

Corpus rows carry a FEN, a ply and a game id, in game order. They carry no best
move, so "the best move is a capture" is approximated by "the move actually
played was a capture": the next row of the same game has fewer pieces.
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator

_KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_ROOK = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def in_check(fen: str) -> bool:
    """Whether the side to move is in check. `fen` must have a valid board field."""
    placement, side = fen.split(' ', 2)[:2]
    board = {}
    rank = 7
    for row in placement.split('/'):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                board[(file, rank)] = ch
                file += 1
        rank -= 1
    white = side == 'w'
    king = 'K' if white else 'k'
    for square, piece in board.items():
        if piece == king:
            kf, kr = square
            break
    else:
        raise ValueError('no king for the side to move')

    def enemy(piece: str, kinds: str) -> bool:
        return piece.isupper() != white and piece.upper() in kinds

    pawn_rank = kr + (1 if white else -1)
    for df in (-1, 1):
        piece = board.get((kf + df, pawn_rank))
        if piece is not None and enemy(piece, 'P'):
            return True
    for steps, kinds in ((_KNIGHT, 'N'), (_KING, 'K')):
        for df, dr in steps:
            piece = board.get((kf + df, kr + dr))
            if piece is not None and enemy(piece, kinds):
                return True
    for rays, kinds in ((_ROOK, 'RQ'), (_BISHOP, 'BQ')):
        for df, dr in rays:
            f, r = kf + df, kr + dr
            while 0 <= f < 8 and 0 <= r < 8:
                piece = board.get((f, r))
                if piece is not None:
                    if enemy(piece, kinds):
                        return True
                    break
                f += df
                r += dr
    return False


def _pieces(fen: str) -> int:
    return sum(ch.isalpha() for ch in fen.split(' ', 1)[0])


def _parse(line: bytes):
    """(game, ply, fen) for a well-formed row, else None."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    game, ply, fen = record.get('game_id'), record.get('ply'), record.get('fen')
    if not isinstance(fen, str) or not isinstance(ply, int) or isinstance(ply, bool):
        return None
    return game, ply, fen


def keep_rows(lines: Iterable[bytes], *, skip_early_plies: int = 0, skip_in_check: bool = False,
              skip_before_capture: bool = False) -> Iterator[bytes]:
    """Yield the rows to train on, unchanged and in order.

    Rows that cannot be parsed pass through untouched; rejecting malformed data
    is the trainer's job, not this filter's.
    """
    if skip_early_plies < 0:
        raise ValueError('skip_early_plies must be nonnegative')
    if not (skip_early_plies or skip_in_check or skip_before_capture):
        yield from lines
        return
    held = None  # (line, game, ply, pieces): kept so far, awaiting the capture lookahead
    for line in lines:
        parsed = _parse(line)
        if parsed is None:
            if held is not None:
                yield held[0]
                held = None
            yield line
            continue
        game, ply, fen = parsed
        pieces = _pieces(fen) if skip_before_capture else 0
        if held is not None:
            captured = held[1] == game and held[2] + 1 == ply and pieces < held[3]
            if not captured:
                yield held[0]
            held = None
        # A dropped row still serves as the lookahead for the row before it,
        # which is why that comparison happens first.
        if ply < skip_early_plies or (skip_in_check and in_check(fen)):
            continue
        if skip_before_capture:
            held = (line, game, ply, pieces)
        else:
            yield line
    if held is not None:
        yield held[0]
