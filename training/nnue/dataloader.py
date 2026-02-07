"""Utility helpers for reading NNUE training JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass
class TrainingRecord:
    """Structured representation of a single training sample."""

    fen: str
    result: int
    result_q: float = 0.0
    value_cp: Optional[float] = None
    ply: Optional[int] = None
    best_move: Optional[str] = None
    policy_top: List[tuple[str, float]] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def read_jsonl_dir(path: str) -> Iterator[dict]:
    """Iterate over JSONL records from a directory or file."""

    root = Path(path)
    if root.is_file():
        yield from _read_jsonl_file(root)
        return
    files = sorted(p for p in root.glob('*.jsonl'))
    for file_path in files:
        yield from _read_jsonl_file(file_path)


def _read_jsonl_file(file_path: Path) -> Iterator[dict]:
    with file_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _coerce_policy(policy_field: object) -> List[tuple[str, float]]:
    moves: List[tuple[str, float]] = []
    if isinstance(policy_field, list):
        for entry in policy_field:
            if isinstance(entry, dict):
                move = entry.get('move')
                prob = entry.get('p', entry.get('prob', 0.0))
                if isinstance(move, str):
                    moves.append((move, float(prob)))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[0], str):
                moves.append((entry[0], float(entry[1])))
    return moves


def _coerce_result(record: dict) -> int:
    if 'result' in record:
        try:
            return int(record['result'])
        except Exception:
            pass
    result_q = record.get('result_q')
    if isinstance(result_q, (int, float)):
        if result_q > 1e-6:
            return 1
        if result_q < -1e-6:
            return -1
    return 0


def _coerce_result_q(record: dict, result: int) -> float:
    result_q = record.get('result_q')
    if isinstance(result_q, (int, float)):
        return float(result_q)
    return float(result)


def _coerce_value_cp(record: dict) -> Optional[float]:
    for key in ('value_cp', 'eval_cp', 'score_cp'):
        v = record.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _coerce_ply(record: dict) -> Optional[int]:
    v = record.get('ply')
    if isinstance(v, int):
        return int(v)
    return None


def jsonl_to_training_samples(records: Iterable[dict]) -> Iterator[TrainingRecord]:
    for record in records:
        fen = record.get('fen')
        if not isinstance(fen, str):
            continue
        result = _coerce_result(record)
        result_q = _coerce_result_q(record, result)
        value_cp = _coerce_value_cp(record)
        ply = _coerce_ply(record)
        best_move = None
        for key in ('target_best_move', 'best_move', 'best_move_canonical', 'played_move'):
            move = record.get(key)
            if isinstance(move, str):
                best_move = move
                break
        policy_top = _coerce_policy(record.get('policy_top'))
        yield TrainingRecord(
            fen=fen,
            result=result,
            result_q=result_q,
            value_cp=value_cp,
            ply=ply,
            best_move=best_move,
            policy_top=policy_top,
            raw=record,
        )
