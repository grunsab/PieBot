"""Utility helpers for reading NNUE training JSONL files."""

from __future__ import annotations

import gzip
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
    outcome_valid: bool = True
    value_cp: Optional[float] = None
    teacher_depth: Optional[int] = None
    run_id: Optional[str] = None
    game_id: Optional[str] = None
    ply: Optional[int] = None
    best_move: Optional[str] = None
    policy_top: List[tuple[str, float]] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def read_jsonl_dir(path: str) -> Iterator[dict]:
    """Iterate over JSONL records from a directory or file.

    Accepts both ``*.jsonl`` and ``*.jsonl.gz``. Self-play rows measure
    374.5 bytes raw and compress 10.35x, so a gzipped corpus costs ~36
    bytes/row. That is what makes accumulation affordable: 1e9 rows is
    ~36 GB compressed against ~319 GB raw, and the box has 150 GB total.
    Without this the replay window has to keep discarding history.

    A shard and its gzipped twin MUST yield byte-identical records --
    see test_dataloader.py::test_gzipped_shard_reads_identically.
    """

    root = Path(path)
    if root.is_file():
        yield from _read_jsonl_file(root)
        return
    # Sort on the stem so shard_000001.jsonl and shard_000001.jsonl.gz
    # occupy the same position in the ordering regardless of suffix.
    files = sorted(
        (p for p in root.iterdir() if _is_jsonl_shard(p)),
        key=lambda p: (p.name[: -len('.gz')] if p.name.endswith('.gz') else p.name),
    )
    for file_path in files:
        yield from _read_jsonl_file(file_path)


def _is_jsonl_shard(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.name.endswith('.jsonl') or path.name.endswith('.jsonl.gz')


def _read_jsonl_file(file_path: Path) -> Iterator[dict]:
    opener = (
        (lambda: gzip.open(file_path, 'rt', encoding='utf-8'))
        if file_path.name.endswith('.gz')
        else (lambda: file_path.open('r', encoding='utf-8'))
    )
    with opener() as handle:
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


def _coerce_outcome_valid(record: dict) -> bool:
    value = record.get('outcome_valid')
    if isinstance(value, bool):
        return value
    return True


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


def _coerce_teacher_depth(record: dict) -> Optional[int]:
    value = record.get('teacher_depth')
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _coerce_optional_string(record: dict, key: str) -> Optional[str]:
    value = record.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def jsonl_to_training_samples(records: Iterable[dict]) -> Iterator[TrainingRecord]:
    for record in records:
        fen = record.get('fen')
        if not isinstance(fen, str):
            continue
        result = _coerce_result(record)
        result_q = _coerce_result_q(record, result)
        outcome_valid = _coerce_outcome_valid(record)
        value_cp = _coerce_value_cp(record)
        teacher_depth = _coerce_teacher_depth(record)
        run_id = _coerce_optional_string(record, 'run_id')
        game_id = _coerce_optional_string(record, 'game_id')
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
            outcome_valid=outcome_valid,
            value_cp=value_cp,
            teacher_depth=teacher_depth,
            run_id=run_id,
            game_id=game_id,
            ply=ply,
            best_move=best_move,
            policy_top=policy_top,
            raw=record,
        )
