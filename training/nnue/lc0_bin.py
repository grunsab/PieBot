"""Utilities for reading LCZero BIN training data (V6)."""

from __future__ import annotations

import gzip
import io
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

FULL_MASK = 0xFFFFFFFFFFFFFFFF

# Enumeration values from proto/net.proto
INPUT_CLASSICAL_112_PLANE = 1
INPUT_112_WITH_CASTLING_PLANE = 2
INPUT_112_WITH_CANONICALIZATION = 3
INPUT_112_WITH_CANONICALIZATION_HECTOPLIES = 4
INPUT_112_WITH_CANONICALIZATION_V2 = 5
INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON = 132
INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON = 133

FLIP_TRANSFORM = 1
MIRROR_TRANSFORM = 2
TRANSPOSE_TRANSFORM = 4

K_PLANES_PER_BOARD = 13
K_MOVE_HISTORY = 8
K_AUX_PLANE_BASE = K_PLANES_PER_BOARD * K_MOVE_HISTORY

STRUCT_V6 = struct.Struct(
    "<II1858f104Q8B15fIHHfI"
)
RECORD_SIZE = STRUCT_V6.size


@dataclass
class V6Record:
    version: int
    input_format: int
    probabilities: List[float]
    planes: List[int]
    castling_us_ooo: int
    castling_us_oo: int
    castling_them_ooo: int
    castling_them_oo: int
    side_to_move_or_enpassant: int
    rule50_count: int
    invariance_info: int
    dummy: int
    root_q: float
    best_q: float
    root_d: float
    best_d: float
    root_m: float
    best_m: float
    plies_left: float
    result_q: float
    result_d: float
    played_q: float
    played_d: float
    played_m: float
    orig_q: float
    orig_d: float
    orig_m: float
    visits: int
    played_idx: int
    best_idx: int
    policy_kld: float
    reserved: int


@dataclass
class Plane:
    mask: int = 0
    value: float = 0.0


@dataclass
class CastlingInfo:
    white_kingside: bool = False
    white_kingside_file: int = 7
    white_queenside: bool = False
    white_queenside_file: int = 0
    black_kingside: bool = False
    black_kingside_file: int = 7
    black_queenside: bool = False
    black_queenside_file: int = 0

    def mirror(self) -> 'CastlingInfo':
        return CastlingInfo(
            white_kingside=self.black_kingside,
            white_kingside_file=7 - self.black_kingside_file,
            white_queenside=self.black_queenside,
            white_queenside_file=7 - self.black_queenside_file,
            black_kingside=self.white_kingside,
            black_kingside_file=7 - self.white_kingside_file,
            black_queenside=self.white_queenside,
            black_queenside_file=7 - self.white_queenside_file,
        )

    def to_fen(self) -> str:
        symbols = []
        if self.white_kingside:
            symbols.append(_castle_symbol(self.white_kingside_file, True, True))
        if self.white_queenside:
            symbols.append(_castle_symbol(self.white_queenside_file, False, True))
        if self.black_kingside:
            symbols.append(_castle_symbol(self.black_kingside_file, True, False))
        if self.black_queenside:
            symbols.append(_castle_symbol(self.black_queenside_file, False, False))
        return ''.join(symbols) or '-'


def _castle_symbol(file_idx: int, is_kingside: bool, is_white: bool) -> str:
    if is_white:
        if file_idx == 7:
            return 'K'
        if file_idx == 0:
            return 'Q'
        return chr(ord('A') + file_idx)
    else:
        if file_idx == 7:
            return 'k'
        if file_idx == 0:
            return 'q'
        return chr(ord('a') + file_idx)


def _bit_index(file_idx: int, rank_idx: int) -> int:
    return rank_idx * 8 + file_idx


def _mask_has(mask: int, file_idx: int, rank_idx: int) -> bool:
    return bool((mask >> _bit_index(file_idx, rank_idx)) & 1)


def _square_name(file_idx: int, rank_idx: int) -> str:
    return f"{chr(file_idx + ord('a'))}{rank_idx + 1}"


def _mirror_mask(mask: int) -> int:
    return reverse_bytes_in_bytes(mask)


def _bit_count(mask: int) -> int:
    return bin(mask & FULL_MASK).count('1')


def _single_square(mask: int) -> tuple[int, int]:
    idx = _lowest_bit_index(mask & FULL_MASK)
    return idx % 8, idx // 8


def reverse_bits_in_bytes(value: int) -> int:
    value &= FULL_MASK
    value = ((value >> 1) & 0x5555555555555555) | ((value & 0x5555555555555555) << 1)
    value = ((value >> 2) & 0x3333333333333333) | ((value & 0x3333333333333333) << 2)
    value = ((value >> 4) & 0x0F0F0F0F0F0F0F0F) | ((value & 0x0F0F0F0F0F0F0F0F) << 4)
    return value & FULL_MASK


def reverse_bytes_in_bytes(value: int) -> int:
    value &= FULL_MASK
    value = ((value & 0x00000000FFFFFFFF) << 32) | ((value & 0xFFFFFFFF00000000) >> 32)
    value = ((value & 0x0000FFFF0000FFFF) << 16) | ((value & 0xFFFF0000FFFF0000) >> 16)
    value = ((value & 0x00FF00FF00FF00FF) << 8) | ((value & 0xFF00FF00FF00FF00) >> 8)
    return value & FULL_MASK


def transpose_bits_in_bytes(value: int) -> int:
    value &= FULL_MASK
    value = (
        ((value & 0xAA00AA00AA00AA00) >> 9)
        | ((value & 0x0055005500550055) << 9)
        | (value & 0x55AA55AA55AA55AA)
    )
    value = (
        ((value & 0xCCCC0000CCCC0000) >> 18)
        | ((value & 0x0000333300003333) << 18)
        | (value & 0x3333CCCC3333CCCC)
    )
    value = (
        ((value & 0xF0F0F0F000000000) >> 36)
        | ((value & 0x000000000F0F0F0F) << 36)
        | (value & 0x0F0F0F0FF0F0F0F0)
    )
    return value & FULL_MASK


def _lowest_bit_index(value: int) -> int:
    return (value & -value).bit_length() - 1


def _is_canonical(input_format: int) -> bool:
    return input_format in {
        INPUT_112_WITH_CANONICALIZATION,
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
        INPUT_112_WITH_CANONICALIZATION_V2,
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    }


def _is_hectoplies(input_format: int) -> bool:
    return input_format in {
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
        INPUT_112_WITH_CANONICALIZATION_V2,
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    }


def _is_armageddon(input_format: int) -> bool:
    return input_format in {
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    }


def _is_castling_plane_format(input_format: int) -> bool:
    return input_format >= INPUT_112_WITH_CASTLING_PLANE


def parse_v6_record(chunk: bytes) -> V6Record:
    if len(chunk) != RECORD_SIZE:
        raise ValueError(f"Invalid record size {len(chunk)} (expected {RECORD_SIZE})")
    unpacked = STRUCT_V6.unpack(chunk)
    idx = 0
    version = unpacked[idx]
    idx += 1
    input_format = unpacked[idx]
    idx += 1
    probabilities = list(unpacked[idx : idx + 1858])
    idx += 1858
    planes = [int(x & FULL_MASK) for x in unpacked[idx : idx + 104]]
    idx += 104
    (
        castling_us_ooo,
        castling_us_oo,
        castling_them_ooo,
        castling_them_oo,
        side_to_move_or_enpassant,
        rule50_count,
        invariance_info,
        dummy,
    ) = unpacked[idx : idx + 8]
    idx += 8
    floats = list(unpacked[idx : idx + 15])
    idx += 15
    (
        root_q,
        best_q,
        root_d,
        best_d,
        root_m,
        best_m,
        plies_left,
        result_q,
        result_d,
        played_q,
        played_d,
        played_m,
        orig_q,
        orig_d,
        orig_m,
    ) = floats
    visits = unpacked[idx]
    idx += 1
    played_idx = unpacked[idx]
    idx += 1
    best_idx = unpacked[idx]
    idx += 1
    policy_kld = unpacked[idx]
    idx += 1
    reserved = unpacked[idx]

    return V6Record(
        version=version,
        input_format=input_format,
        probabilities=probabilities,
        planes=planes,
        castling_us_ooo=castling_us_ooo,
        castling_us_oo=castling_us_oo,
        castling_them_ooo=castling_them_ooo,
        castling_them_oo=castling_them_oo,
        side_to_move_or_enpassant=side_to_move_or_enpassant,
        rule50_count=rule50_count,
        invariance_info=invariance_info,
        dummy=dummy,
        root_q=root_q,
        best_q=best_q,
        root_d=root_d,
        best_d=best_d,
        root_m=root_m,
        best_m=best_m,
        plies_left=plies_left,
        result_q=result_q,
        result_d=result_d,
        played_q=played_q,
        played_d=played_d,
        played_m=played_m,
        orig_q=orig_q,
        orig_d=orig_d,
        orig_m=orig_m,
        visits=visits,
        played_idx=played_idx,
        best_idx=best_idx,
        policy_kld=policy_kld,
        reserved=reserved,
    )


def iter_v6_records(stream: io.BufferedReader) -> Iterator[V6Record]:
    while True:
        chunk = stream.read(RECORD_SIZE)
        if not chunk:
            break
        if len(chunk) != RECORD_SIZE:
            raise ValueError("Truncated V6 record encountered")
        yield parse_v6_record(chunk)


def open_v6_file(path: Path) -> io.BufferedReader:
    raw = path.open('rb')
    magic = raw.read(2)
    if magic == b'\x1f\x8b':
        raw.close()
        return gzip.open(path, 'rb')
    raw.seek(0)
    return raw


def _plane_from_mask(mask: int) -> Plane:
    return Plane(mask=mask & FULL_MASK)


def decode_planes(record: V6Record) -> List[Plane]:
    planes = [_plane_from_mask(reverse_bits_in_bytes(mask)) for mask in record.planes]
    fmt = record.input_format
    if fmt == INPUT_CLASSICAL_112_PLANE:
        planes.extend(
            [
                _plane_from_mask(FULL_MASK if record.castling_us_ooo else 0),
                _plane_from_mask(FULL_MASK if record.castling_us_oo else 0),
                _plane_from_mask(FULL_MASK if record.castling_them_ooo else 0),
                _plane_from_mask(FULL_MASK if record.castling_them_oo else 0),
            ]
        )
    elif fmt in {
        INPUT_112_WITH_CASTLING_PLANE,
        INPUT_112_WITH_CANONICALIZATION,
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
        INPUT_112_WITH_CANONICALIZATION_V2,
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    }:
        mask0 = record.castling_us_ooo | (record.castling_them_ooo << 56)
        mask1 = record.castling_us_oo | (record.castling_them_oo << 56)
        planes.append(_plane_from_mask(mask0))
        planes.append(_plane_from_mask(mask1))
        planes.append(Plane())
        planes.append(Plane())
    else:
        raise ValueError(f"Unsupported input_format {fmt}")

    if _is_canonical(fmt):
        sm_mask = (record.side_to_move_or_enpassant & 0xFF) << 56
        planes.append(_plane_from_mask(sm_mask))
    else:
        planes.append(_plane_from_mask(FULL_MASK if record.side_to_move_or_enpassant else 0))

    rule_plane = Plane()
    if _is_hectoplies(fmt):
        rule_plane.value = record.rule50_count / 100.0
    else:
        rule_plane.value = float(record.rule50_count)
    planes.append(rule_plane)

    armageddon_plane = Plane()
    if _is_armageddon(fmt) and record.invariance_info >= 128:
        armageddon_plane.mask = FULL_MASK
    planes.append(armageddon_plane)

    planes.append(_plane_from_mask(FULL_MASK))

    if _is_canonical(fmt) and (record.invariance_info & 0x7):
        transform = record.invariance_info & 0x7
        for plane in planes:
            mask = plane.mask & FULL_MASK
            if mask == 0 or mask == FULL_MASK:
                continue
            if transform & TRANSPOSE_TRANSFORM:
                mask = transpose_bits_in_bytes(mask)
            if transform & MIRROR_TRANSFORM:
                mask = reverse_bytes_in_bytes(mask)
            if transform & FLIP_TRANSFORM:
                mask = reverse_bits_in_bytes(mask)
            plane.mask = mask
    return planes


@lru_cache(maxsize=1)
def _load_policy_moves() -> List[str]:
    data_path = Path(__file__).resolve().parent / "resources" / "lc0_policy_moves.json"
    with data_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _transform_square(file_idx: int, rank_idx: int, transform: int) -> Tuple[int, int]:
    if transform & (MIRROR_TRANSFORM | TRANSPOSE_TRANSFORM):
        rank_idx = 7 - rank_idx
    if transform & (FLIP_TRANSFORM | TRANSPOSE_TRANSFORM):
        file_idx = 7 - file_idx
    return file_idx, rank_idx


def _inverse_transform(transform: int) -> int:
    if transform & TRANSPOSE_TRANSFORM:
        inv = TRANSPOSE_TRANSFORM
        if transform & FLIP_TRANSFORM:
            inv |= MIRROR_TRANSFORM
        if transform & MIRROR_TRANSFORM:
            inv |= FLIP_TRANSFORM
        return inv
    return transform


def move_from_nn_index(index: int, transform: int) -> str:
    moves = _load_policy_moves()
    move_str = moves[index]
    file_from = ord(move_str[0]) - ord("a")
    rank_from = int(move_str[1]) - 1
    file_to = ord(move_str[2]) - ord("a")
    rank_to = int(move_str[3]) - 1
    promo = move_str[4] if len(move_str) == 5 else ""
    if transform:
        inv = _inverse_transform(transform)
        file_from, rank_from = _transform_square(file_from, rank_from, inv)
        file_to, rank_to = _transform_square(file_to, rank_to, inv)
    result = f"{chr(file_from + ord('a'))}{rank_from + 1}{chr(file_to + ord('a'))}{rank_to + 1}"
    if promo:
        result += promo
    return result


def flip_move_uci(move: str) -> str:
    file_from = ord(move[0]) - ord("a")
    rank_from = int(move[1]) - 1
    file_to = ord(move[2]) - ord("a")
    rank_to = int(move[3]) - 1
    promo = move[4:] if len(move) > 4 else ""
    file_from = 7 - file_from
    file_to = 7 - file_to
    rank_from = 7 - rank_from
    rank_to = 7 - rank_to
    result = f"{chr(file_from + ord('a'))}{rank_from + 1}{chr(file_to + ord('a'))}{rank_to + 1}"
    if promo:
        result += promo
    return result


def extract_policy_top(record: V6Record, top_n: int = 8, transform: int = 0) -> List[Tuple[str, float]]:
    items = [
        (idx, prob)
        for idx, prob in enumerate(record.probabilities)
        if prob >= 0.0
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    moves: List[Tuple[str, float]] = []
    for idx, prob in items[:top_n]:
        moves.append((move_from_nn_index(idx, transform), prob))
    return moves


def make_record_stream(path: Path) -> Iterator[V6Record]:
    with open_v6_file(path) as stream:
        yield from iter_v6_records(stream)


def _decode_castlings(fmt: int, planes: Sequence[Plane]) -> CastlingInfo:
    info = CastlingInfo()
    mask0 = planes[K_AUX_PLANE_BASE + 0].mask & FULL_MASK if len(planes) > K_AUX_PLANE_BASE else 0
    mask1 = planes[K_AUX_PLANE_BASE + 1].mask & FULL_MASK if len(planes) > K_AUX_PLANE_BASE + 1 else 0
    if fmt == INPUT_CLASSICAL_112_PLANE:
        if mask0:
            info.white_queenside = True
        if mask1:
            info.white_kingside = True
        if planes[K_AUX_PLANE_BASE + 2].mask:
            info.black_queenside = True
        if planes[K_AUX_PLANE_BASE + 3].mask:
            info.black_kingside = True
    else:
        lower0 = mask0 & 0xFF
        upper0 = (mask0 >> 56) & 0xFF
        lower1 = mask1 & 0xFF
        upper1 = (mask1 >> 56) & 0xFF
        if lower0:
            info.white_queenside = True
            info.white_queenside_file = _lowest_bit_index(lower0)
        if lower1:
            info.white_kingside = True
            info.white_kingside_file = _lowest_bit_index(lower1)
        if upper0:
            info.black_queenside = True
            info.black_queenside_file = _lowest_bit_index(upper0)
        if upper1:
            info.black_kingside = True
            info.black_kingside_file = _lowest_bit_index(upper1)
    return info


def _piece_char(file_idx: int, rank_idx: int, masks: dict[str, int]) -> str | None:
    if _mask_has(masks['pawns_us'], file_idx, rank_idx):
        return 'P'
    if _mask_has(masks['pawns_them'], file_idx, rank_idx):
        return 'p'
    if _mask_has(masks['knights_us'], file_idx, rank_idx):
        return 'N'
    if _mask_has(masks['knights_them'], file_idx, rank_idx):
        return 'n'
    if _mask_has(masks['bishops_us'], file_idx, rank_idx):
        return 'B'
    if _mask_has(masks['bishops_them'], file_idx, rank_idx):
        return 'b'
    if _mask_has(masks['rooks_us'], file_idx, rank_idx):
        return 'R'
    if _mask_has(masks['rooks_them'], file_idx, rank_idx):
        return 'r'
    if _mask_has(masks['queens_us'], file_idx, rank_idx):
        return 'Q'
    if _mask_has(masks['queens_them'], file_idx, rank_idx):
        return 'q'
    if _mask_has(masks['kings_us'], file_idx, rank_idx):
        return 'K'
    if _mask_has(masks['kings_them'], file_idx, rank_idx):
        return 'k'
    return None


def build_fen_from_planes(record: V6Record, planes: Sequence[Plane]) -> dict:
    fmt = record.input_format
    is_canonical = _is_canonical(fmt)
    masks = {
        'pawns_us': planes[0].mask & FULL_MASK,
        'knights_us': planes[1].mask & FULL_MASK,
        'bishops_us': planes[2].mask & FULL_MASK,
        'rooks_us': planes[3].mask & FULL_MASK,
        'queens_us': planes[4].mask & FULL_MASK,
        'kings_us': planes[5].mask & FULL_MASK,
        'pawns_them': planes[6].mask & FULL_MASK,
        'knights_them': planes[7].mask & FULL_MASK,
        'bishops_them': planes[8].mask & FULL_MASK,
        'rooks_them': planes[9].mask & FULL_MASK,
        'queens_them': planes[10].mask & FULL_MASK,
        'kings_them': planes[11].mask & FULL_MASK,
    }
    castlings = _decode_castlings(fmt, planes)
    black_to_move_flag = (not is_canonical) and bool(planes[K_AUX_PLANE_BASE + 4].mask)
    if black_to_move_flag:
        masks['pawns_us'], masks['pawns_them'] = masks['pawns_them'], masks['pawns_us']
        masks['knights_us'], masks['knights_them'] = masks['knights_them'], masks['knights_us']
        masks['bishops_us'], masks['bishops_them'] = masks['bishops_them'], masks['bishops_us']
        masks['rooks_us'], masks['rooks_them'] = masks['rooks_them'], masks['rooks_us']
        masks['queens_us'], masks['queens_them'] = masks['queens_them'], masks['queens_us']
        masks['kings_us'], masks['kings_them'] = masks['kings_them'], masks['kings_us']
        for key in list(masks.keys()):
            masks[key] = _mirror_mask(masks[key])
        castlings = castlings.mirror()

    rows: list[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        row_chars: list[str] = []
        for file in range(8):
            piece = _piece_char(file, rank, masks)
            if piece:
                if empty:
                    row_chars.append(str(empty))
                    empty = 0
                row_chars.append(piece)
            else:
                empty += 1
        if empty:
            row_chars.append(str(empty))
        rows.append(''.join(row_chars))
    board_part = '/'.join(rows)
    side_char = 'b' if black_to_move_flag else 'w'
    castling_str = castlings.to_fen()
    if is_canonical:
        mask = planes[K_AUX_PLANE_BASE + 4].mask & FULL_MASK if len(planes) > K_AUX_PLANE_BASE + 4 else 0
        if mask == 0:
            en_passant = '-'
        else:
            file_idx = _lowest_bit_index(((mask >> 56) & 0xFF) or mask)
            en_passant = _square_name(file_idx, 5)
    else:
        en_passant = '-'
        if len(planes) > K_PLANES_PER_BOARD + 6:
            pawndiff = (planes[6].mask ^ planes[K_PLANES_PER_BOARD + 6].mask) & FULL_MASK
            prev_mask = planes[K_PLANES_PER_BOARD + 6].mask & FULL_MASK
            if _bit_count(pawndiff) == 2 and prev_mask:
                from_file, from_rank = _single_square(prev_mask & pawndiff)
                to_file, to_rank = _single_square(planes[6].mask & pawndiff)
                if from_file != to_file or abs(from_rank - to_rank) != 2:
                    en_passant = '-'
                else:
                    target_rank = 2 if planes[K_AUX_PLANE_BASE + 4].mask else 5
                    en_passant = _square_name(to_file, target_rank)
    rule50 = int(record.rule50_count)
    fen = f"{board_part} {side_char} {castling_str} {en_passant} {rule50} {rule50}"
    return {
        'fen': fen,
        'black_to_move': black_to_move_flag,
        'castling': castlings,
        'en_passant': en_passant,
    }


def record_to_sample(record: V6Record, top_policy: int = 8) -> dict:
    planes = decode_planes(record)
    fmt = record.input_format
    is_canonical = _is_canonical(fmt)
    fen_info = build_fen_from_planes(record, planes)
    transform = record.invariance_info & 0x7 if is_canonical else 0
    canonical_best = move_from_nn_index(record.best_idx, transform)
    canonical_played = move_from_nn_index(record.played_idx, transform)
    actual_black = bool(record.invariance_info & (1 << 7)) if is_canonical else fen_info['black_to_move']
    best_move = canonical_best if not actual_black else flip_move_uci(canonical_best)
    played_move = canonical_played if not actual_black else flip_move_uci(canonical_played)
    policy_top_canonical = extract_policy_top(record, top_policy, transform)
    if actual_black:
        policy_top = [(flip_move_uci(move), prob) for move, prob in policy_top_canonical]
    else:
        policy_top = policy_top_canonical
    if record.result_q > 1e-6:
        result = 1
    elif record.result_q < -1e-6:
        result = -1
    else:
        result = 0
    sample = {
        'fen': fen_info['fen'],
        'fen_side_to_move': 'b' if fen_info['black_to_move'] else 'w',
        'original_side_to_move': 'b' if actual_black else 'w',
        'is_canonical': is_canonical,
        'best_move': best_move,
        'played_move': played_move,
        'best_move_canonical': canonical_best,
        'played_move_canonical': canonical_played,
        'policy_top': policy_top,
        'policy_top_canonical': policy_top_canonical,
        'result': result,
        'result_q': record.result_q,
        'result_d': record.result_d,
        'best_q': record.best_q,
        'root_q': record.root_q,
        'visits': record.visits,
        'best_idx': record.best_idx,
        'played_idx': record.played_idx,
        'policy_kld': record.policy_kld,
        'rule50_count': record.rule50_count,
        'plies_left': record.plies_left,
        'invariance_info': record.invariance_info,
        'side_to_move_or_enpassant': record.side_to_move_or_enpassant,
        'en_passant': fen_info['en_passant'],
        'castling': fen_info['castling'].to_fen(),
    }
    return sample
