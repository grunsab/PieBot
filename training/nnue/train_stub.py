#!/usr/bin/env python3
"""
Tiny NNUE trainer (bootstrap implementation).

Reads JSONL training data with {"fen": str, "result": int}, computes HalfKP-style
active feature indices, and trains a small one-hidden-layer scalar network
(ReLU + linear head) with minibatch SGD on MSE targets.

Usage:
  python training/nnue/train_stub.py \
    --jsonl-dir data/nnue_jsonl/test80 --batch-size 4096 --max-samples 500000 \
    --epochs 8 --val-split 0.1 --learning-rate 0.05 --out out/nnue_stub_train

"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from .dataloader import TrainingRecord, jsonl_to_training_samples, read_jsonl_dir
except Exception:
    from dataloader import TrainingRecord, jsonl_to_training_samples, read_jsonl_dir  # type: ignore


PIECE_ORDER = "PNBRQ"
FEATURE_PIECE_ORDER = "PNBRQpnbrq"
COUNT_ORDER = "PNBRQKpnbrqk"
_DIRECT_CHECKPOINT_FORMATS = {
    "piebot-halfkp-mse-v2",
    "piebot-halfkp-mse-v2-torch",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_jsonl_source(path: Path) -> str:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    digest = hashlib.sha256()
    for file_path in files:
        name = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _count_jsonl_source_records(path: Path) -> int:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    records = 0
    for file_path in files:
        with file_path.open("rb") as handle:
            records += sum(1 for line in handle if line.strip())
    return records


def _load_initial_checkpoint(
    checkpoint_path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    objective: Dict[str, Any],
) -> Tuple[List[float], List[float], List[float], float, Dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"initial checkpoint does not exist: {path}")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid initial checkpoint JSON: {path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a JSON object")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in _DIRECT_CHECKPOINT_FORMATS:
        raise ValueError(f"unsupported initial checkpoint format: {checkpoint_format!r}")
    if int(checkpoint.get("input_dim", 0)) != input_dim:
        raise ValueError("initial checkpoint input_dim mismatch")
    if int(checkpoint.get("hidden_dim", 0)) != hidden_dim:
        raise ValueError("initial checkpoint hidden_dim mismatch")
    checkpoint_feature_set = checkpoint.get("feature_set")
    if checkpoint_feature_set != FEATURE_SET:
        raise ValueError("initial checkpoint feature_set mismatch")
    checkpoint_target_schema = checkpoint.get("target_schema")
    if checkpoint_target_schema != TARGET_SCHEMA:
        raise ValueError("initial checkpoint target_schema mismatch")
    if checkpoint.get("objective") != objective:
        raise ValueError("initial checkpoint objective mismatch")

    values: Dict[str, List[float]] = {}
    for key, expected_len in (
        ("w1", input_dim * hidden_dim),
        ("b1", hidden_dim),
        ("w2", hidden_dim),
    ):
        raw = checkpoint.get(key)
        if not isinstance(raw, list) or len(raw) != expected_len:
            raise ValueError(f"initial checkpoint {key} size mismatch")
        try:
            converted = [float(value) for value in raw]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"initial checkpoint {key} contains non-numeric values") from exc
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"initial checkpoint {key} contains non-finite values")
        values[key] = converted
    try:
        b2 = float(checkpoint["b2"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("initial checkpoint b2 is missing or non-numeric") from exc
    if not math.isfinite(b2):
        raise ValueError("initial checkpoint b2 contains a non-finite value")
    metadata = {
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "format": str(checkpoint_format),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "feature_set": checkpoint_feature_set,
        "target_schema": checkpoint_target_schema,
        "objective": objective,
    }
    return values["w1"], values["b1"], values["w2"], b2, metadata


def _parse_board_fen(fen: str) -> Tuple[List[Tuple[str, int]], int | None, int | None]:
    board_part = fen.split()[0]
    ranks = board_part.split("/")
    if len(ranks) != 8:
        raise ValueError(f"invalid FEN board part: {board_part}")

    pieces: List[Tuple[str, int]] = []
    white_king = None
    black_king = None
    for fen_rank, rank_str in enumerate(ranks):
        rank_idx = 7 - fen_rank
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            if file_idx >= 8:
                raise ValueError(f"invalid FEN rank overflow: {rank_str}")
            sq = rank_idx * 8 + file_idx
            pieces.append((ch, sq))
            if ch == "K":
                white_king = sq
            elif ch == "k":
                black_king = sq
            file_idx += 1
        if file_idx != 8:
            raise ValueError(f"invalid FEN rank width: {rank_str}")
    return pieces, white_king, black_king


def featureize_fen_counts(fen: str) -> List[int]:
    """Simple placeholder features: counts of 12 piece types (white/black x 6).
    Order: [P,N,B,R,Q,K, p,n,b,r,q,k]
    """
    pieces, _wk, _bk = _parse_board_fen(fen)
    counts = {ch: 0 for ch in COUNT_ORDER}
    for piece, _sq in pieces:
        if piece in counts:
            counts[piece] += 1
    return [counts[ch] for ch in COUNT_ORDER]


LEGACY_HALFKP_DIM = 2 * 64 * len(PIECE_ORDER) * 64
HALFKP_DIM = 2 * 64 * len(FEATURE_PIECE_ORDER) * 64
FEATURE_SET = "halfkp-all-pieces-v2"
TARGET_SCHEMA = "soft-cp-wdl-v2"
OBJECTIVE_SCHEMA = "nnue-objective-v1"
SAMPLING_SCHEMA = "source-teacher-stratified-v3-no-internal-leakage"
FIXED_VALIDATION_SAMPLING_SCHEMA = "uniform-reservoir-v1"
PRIMARY_VALIDATION_SAMPLING_SCHEMA = "game-hash-disjoint-training-mixture-v1"
CHECKPOINT_SELECTION_SCHEMA = "reference-selected-primary-guarded-v2"
REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION = 0.01
# The primary validation split is drawn from the current training mixture, which is
# mostly outcome-only rows; its per-epoch loss carries between-cycle noise of order
# 1e-3 relative. Selecting on a strict improvement in that metric discarded four
# consecutive campaign_v6 cycles whose teacher-labeled reference loss had genuinely
# improved (deltas as small as 6e-5 relative). The primary split is therefore a
# divergence guard with a noise-band tolerance, not the selection signal.
PRIMARY_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION = 0.002
PRIMARY_VALIDATION_HASH_NAMESPACE = "piebot-primary-validation-partition-v1"


def is_better_checkpoint(
    *,
    val_loss: float,
    best_val_loss: float,
    reference_val_loss: Optional[float],
    best_reference_val_loss: Optional[float],
    initial_reference_val_loss: Optional[float],
) -> bool:
    """Decide whether this epoch's weights should replace the incumbent best.

    An epoch is selected when it improves *either* the noisy primary split or the
    frozen teacher-labeled reference split, provided the reference loss stays
    inside its absolute envelope and the primary loss stays inside its noise band.
    Requiring the primary split specifically to improve is what stalled
    campaign_v6. Runs without a reference split keep the original
    strict-primary-improvement behavior.
    """
    if val_loss is None or not math.isfinite(float(val_loss)):
        return False

    if reference_val_loss is None or best_reference_val_loss is None:
        return float(val_loss) < float(best_val_loss)

    if not math.isfinite(float(reference_val_loss)):
        return False

    # Absolute envelope against the weights this cycle started from, so neither
    # metric can license drifting away from the incoming model.
    if initial_reference_val_loss is not None and math.isfinite(
        float(initial_reference_val_loss)
    ):
        reference_limit = float(initial_reference_val_loss) * (
            1.0 + REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        )
        if float(reference_val_loss) > reference_limit + 1e-12:
            return False

    # Either split may supply the evidence of progress. Requiring the primary
    # split specifically to improve is what stalled campaign_v6: its loss is
    # noise-dominated, so real gains visible on the reference split were vetoed.
    primary_improves = float(val_loss) < float(best_val_loss)
    reference_improves = float(reference_val_loss) < float(best_reference_val_loss)
    if not (primary_improves or reference_improves):
        return False

    # Divergence guard: the primary split may sit inside its noise band, but it
    # must not blow past it, which would indicate genuine overfitting.
    primary_limit = float(best_val_loss) * (
        1.0 + PRIMARY_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
    )
    return float(val_loss) <= primary_limit + 1e-12


def _active_halfkp_indices(fen: str) -> List[int]:
    pieces, wk, bk = _parse_board_fen(fen)
    if wk is None or bk is None:
        return []
    out: List[int] = []
    for perspective_off, ksq in enumerate((wk, bk)):
        for actual, sq in pieces:
            if actual in ("K", "k"):
                continue
            try:
                piece_plane = FEATURE_PIECE_ORDER.index(actual)
            except ValueError:
                continue
            idx = (
                (
                    (perspective_off * 64 + ksq) * len(FEATURE_PIECE_ORDER)
                    + piece_plane
                )
                * 64
            ) + sq
            out.append(idx)
    return out


def _teacher_available(record: TrainingRecord, min_teacher_depth: int = 0) -> bool:
    if record.value_cp is None or not math.isfinite(float(record.value_cp)):
        return False
    minimum = max(0, int(min_teacher_depth))
    if minimum <= 0:
        return True
    return record.teacher_depth is not None and int(record.teacher_depth) >= minimum


def _has_usable_target(record: TrainingRecord, min_teacher_depth: int = 0) -> bool:
    if record.outcome_valid:
        return True
    return _teacher_available(record, min_teacher_depth)


def _iter_usable_records(
    paths: Sequence[Path],
    min_teacher_depth: int = 0,
    require_teacher: bool = False,
) -> Iterator[TrainingRecord]:
    for path in paths:
        for record in jsonl_to_training_samples(read_jsonl_dir(str(path))):
            if _has_usable_target(record, min_teacher_depth) and (
                not require_teacher or _teacher_available(record, min_teacher_depth)
            ):
                yield record


def _training_source_groups(path: Path) -> List[List[Path]]:
    root = Path(path)
    if root.is_file():
        return [[root]]
    files = sorted(root.glob("*.jsonl"))
    if not files:
        return []

    grouped: Dict[int, List[Path]] = {}
    unmatched: List[Path] = []
    for file_path in files:
        match = re.match(r"src(\d+)_", file_path.name)
        if match is None:
            unmatched.append(file_path)
            continue
        grouped.setdefault(int(match.group(1)), []).append(file_path)
    if not grouped:
        return [files]

    groups = [grouped[key] for key in sorted(grouped)]
    if unmatched:
        groups.append(unmatched)
    return groups


def _jsonl_files(path: Path) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.jsonl"))


def _record_identity(record: TrainingRecord) -> str:
    """Return a stable identity used to keep duplicate samples together."""
    if record.run_id is not None and record.game_id is not None and record.ply is not None:
        return f"position\0{record.run_id}\0{record.game_id}\0{record.ply}"
    # Legacy rows have no reliable provenance. Hash exactly the parsed fields
    # that determine features and scalar targets, not the raw JSON object, so
    # irrelevant metadata cannot disguise a copied training example.
    encoded = json.dumps(
        (
            "piebot-training-record-v1",
            record.fen,
            record.result,
            record.result_q,
            record.outcome_valid,
            record.value_cp,
            record.teacher_depth,
            record.ply,
        ),
        separators=(",", ":"),
    ).encode("utf-8")
    return f"record\0{hashlib.sha256(encoded).hexdigest()}"


def _sample_identity_sha256(records: Sequence[TrainingRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_record_identity(record).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validation_group_identity(record: TrainingRecord) -> str:
    """Group an entire provenance-aware game into one validation partition."""
    if record.run_id is not None and record.game_id is not None:
        return f"game\0{record.run_id}\0{record.game_id}"
    return _record_identity(record)


def _internal_validation_partition(
    records: Sequence[TrainingRecord],
    val_split: float,
    *,
    min_teacher_depth: int = 0,
    validation_seed: int = 20_260_802,
) -> Tuple[List[int], List[int]]:
    """Partition samples without placing a game or duplicate on both sides.

    Provenance-aware records are grouped by game. Legacy teacher balancing can
    intentionally repeat a scarce labeled position, so all exact copies of
    that position remain together as the strongest available fallback.
    """
    return _internal_validation_partition_from_metadata(
        [_validation_group_identity(record) for record in records],
        [_teacher_available(record, min_teacher_depth) for record in records],
        val_split,
        validation_seed=validation_seed,
    )


def _internal_validation_partition_from_metadata(
    identities: Sequence[str],
    teacher_available: Sequence[bool],
    val_split: float,
    *,
    validation_seed: int = 20_260_802,
) -> Tuple[List[int], List[int]]:
    """Partition compact metadata with a stable, whole-group hash assignment.

    The assignment depends only on the namespace, validation seed, and group
    identity. A game therefore remains held out when it is encountered through
    replay or when input ordering/training seeds change. Tiny datasets receive
    a deterministic non-empty fallback when the hash threshold puts every
    group on the same side.
    """
    if len(identities) != len(teacher_available):
        raise ValueError("validation partition metadata length mismatch")
    count = len(identities)
    split = min(0.9, max(0.0, float(val_split)))
    if split <= 0.0 or count <= 1:
        return list(range(count)), []

    # Materialize unique identities in sorted order so even the tiny-set
    # fallback is independent of source order.
    unique_identities = sorted(set(identities))
    if len(unique_identities) <= 1:
        return list(range(count)), []

    def group_hash(identity: str) -> int:
        digest = hashlib.sha256()
        digest.update(PRIMARY_VALIDATION_HASH_NAMESPACE.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(validation_seed)).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity.encode("utf-8"))
        return int.from_bytes(digest.digest(), "big")

    group_hashes = {
        identity: group_hash(identity) for identity in unique_identities
    }
    split_numerator, split_denominator = split.as_integer_ratio()
    threshold = ((1 << 256) * split_numerator) // split_denominator
    validation_identities = {
        identity
        for identity, identity_hash in group_hashes.items()
        if identity_hash < threshold
    }
    if not validation_identities:
        validation_identities.add(
            min(unique_identities, key=lambda identity: group_hashes[identity])
        )
    elif len(validation_identities) == len(unique_identities):
        validation_identities.remove(
            max(unique_identities, key=lambda identity: group_hashes[identity])
        )

    train_indices = [
        idx for idx, identity in enumerate(identities)
        if identity not in validation_identities
    ]
    validation_indices = [
        idx for idx, identity in enumerate(identities)
        if identity in validation_identities
    ]
    return train_indices, validation_indices


def _assert_validation_source_disjoint(
    validation_jsonl_dir: Path,
    training_jsonl_dir: Path,
) -> None:
    """Fail closed when fixed validation aliases any training data."""
    validation_root = Path(validation_jsonl_dir)
    training_root = Path(training_jsonl_dir)
    if validation_root.resolve() == training_root.resolve():
        raise ValueError("fixed validation source must be separate from training data")

    validation_files = _jsonl_files(validation_root)
    training_files = _jsonl_files(training_root)

    def file_identities(paths: Sequence[Path]) -> set[Tuple[int, int]]:
        identities: set[Tuple[int, int]] = set()
        for path in paths:
            stat = path.stat()
            identities.add((int(stat.st_dev), int(stat.st_ino)))
        return identities

    if file_identities(validation_files).intersection(file_identities(training_files)):
        raise ValueError("fixed validation source overlaps training data by file identity")

    validation_hashes = {_sha256_file(path) for path in validation_files}
    training_hashes = {_sha256_file(path) for path in training_files}
    if validation_hashes.intersection(training_hashes):
        raise ValueError("fixed validation source contains a copied shard from training data")

    def provenance_keys(
        paths: Sequence[Path],
    ) -> Tuple[set[Tuple[str, str]], set[str]]:
        games: set[Tuple[str, str]] = set()
        legacy_records: set[str] = set()
        for path in paths:
            for record in jsonl_to_training_samples(read_jsonl_dir(str(path))):
                if record.run_id is not None and record.game_id is not None:
                    games.add((record.run_id, record.game_id))
                else:
                    # Legacy data has no game provenance. Exact canonical
                    # record identity is the strongest safe isolation signal
                    # available; fail closed instead of silently validating on
                    # a copied row from training.
                    legacy_records.add(_record_identity(record))
        return games, legacy_records

    validation_games, validation_legacy_records = provenance_keys(validation_files)
    training_games, training_legacy_records = provenance_keys(training_files)
    if validation_games and validation_games.intersection(training_games):
        raise ValueError("fixed validation source overlaps training game provenance")
    if validation_legacy_records.intersection(training_legacy_records):
        raise ValueError("fixed validation source overlaps training record identity")


def _balanced_sample_quotas(counts: Sequence[int], limit: int) -> List[int]:
    quotas = [0 for _ in counts]
    remaining = min(max(0, int(limit)), sum(max(0, int(v)) for v in counts))
    active = [idx for idx, count in enumerate(counts) if count > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        progressed = 0
        next_active: List[int] = []
        for position, idx in enumerate(active):
            requested = share + (1 if position < extra else 0)
            available = counts[idx] - quotas[idx]
            taken = min(requested, available)
            quotas[idx] += taken
            remaining -= taken
            progressed += taken
            if quotas[idx] < counts[idx]:
                next_active.append(idx)
        if progressed == 0:
            break
        active = next_active
    return quotas


def _primary_weighted_sample_quotas(
    counts: Sequence[int],
    limit: int,
    primary_sample_fraction: float,
) -> List[int]:
    total = min(max(0, int(limit)), sum(max(0, int(value)) for value in counts))
    if not counts or total <= 0:
        return [0 for _ in counts]
    if len(counts) == 1:
        return [min(max(0, int(counts[0])), total)]

    fraction = _clamp(float(primary_sample_fraction), 0.0, 1.0)
    primary_target = int(round(total * fraction))
    quotas = [0 for _ in counts]
    quotas[0] = min(max(0, int(counts[0])), primary_target)

    replay_target = total - primary_target
    replay_quotas = _balanced_sample_quotas(counts[1:], replay_target)
    for idx, quota in enumerate(replay_quotas, start=1):
        quotas[idx] = quota

    remaining = total - sum(quotas)
    if remaining > 0:
        capacities = [max(0, int(count)) - quotas[idx] for idx, count in enumerate(counts)]
        extra = _balanced_sample_quotas(capacities, remaining)
        quotas = [quota + extra[idx] for idx, quota in enumerate(quotas)]
    return quotas


def _reservoir_sample_records(
    paths: Sequence[Path],
    limit: int,
    seed: int,
    min_teacher_depth: int = 0,
    teacher_required: Optional[bool] = None,
    require_teacher: bool = False,
) -> List[TrainingRecord]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    reservoir: List[TrainingRecord] = []
    seen = 0
    for record in _iter_usable_records(
        paths,
        min_teacher_depth,
        require_teacher=require_teacher,
    ):
        if (
            teacher_required is not None
            and _teacher_available(record, min_teacher_depth) != teacher_required
        ):
            continue
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(record)
            continue
        replacement = rng.randrange(seen)
        if replacement < limit:
            reservoir[replacement] = record
    return reservoir


def iterate_fixed_validation_samples(
    jsonl_dir: Path,
    max_samples: int,
    *,
    seed: int,
    min_teacher_depth: int = 0,
    require_teacher: bool = False,
) -> Iterator[Tuple[List[int], TrainingRecord]]:
    """Uniformly sample fixed validation, independent of training mix knobs."""
    paths = _jsonl_files(Path(jsonl_dir))
    if max_samples <= 0:
        records = list(
            _iter_usable_records(
                paths,
                min_teacher_depth,
                require_teacher=require_teacher,
            )
        )
    else:
        records = _reservoir_sample_records(
            paths,
            max_samples,
            seed,
            min_teacher_depth,
            require_teacher=require_teacher,
        )
    for record in records:
        yield _active_halfkp_indices(record.fen), record


def _reservoir_sample_record_strata(
    paths: Sequence[Path],
    teacher_limit: int,
    outcome_limit: int,
    seed: int,
    min_teacher_depth: int = 0,
    require_teacher: bool = False,
) -> Tuple[List[TrainingRecord], List[TrainingRecord]]:
    """Reservoir-sample both strata, then deterministically balance by cycling."""
    teacher_limit = max(0, int(teacher_limit))
    outcome_limit = max(0, int(outcome_limit))
    teacher_rng = random.Random(int(seed) + 101)
    outcome_rng = random.Random(int(seed) + 211)
    teachers: List[TrainingRecord] = []
    outcomes: List[TrainingRecord] = []
    teachers_seen = 0
    outcomes_seen = 0
    for record in _iter_usable_records(
        paths,
        min_teacher_depth,
        require_teacher=require_teacher,
    ):
        is_teacher = _teacher_available(record, min_teacher_depth)
        if is_teacher:
            teachers_seen += 1
            if len(teachers) < teacher_limit:
                teachers.append(record)
            elif teacher_limit > 0:
                replacement = teacher_rng.randrange(teachers_seen)
                if replacement < teacher_limit:
                    teachers[replacement] = record
        else:
            outcomes_seen += 1
            if len(outcomes) < outcome_limit:
                outcomes.append(record)
            elif outcome_limit > 0:
                replacement = outcome_rng.randrange(outcomes_seen)
                if replacement < outcome_limit:
                    outcomes[replacement] = record

    def expand_to_limit(
        records: List[TrainingRecord],
        limit: int,
        expansion_seed: int,
    ) -> List[TrainingRecord]:
        if not records or len(records) >= limit:
            return records[:limit]
        expanded = list(records)
        rng = random.Random(expansion_seed)
        while len(expanded) < limit:
            cycle = list(records)
            rng.shuffle(cycle)
            expanded.extend(cycle[: limit - len(expanded)])
        return expanded

    return (
        expand_to_limit(teachers, teacher_limit, int(seed) + 401),
        expand_to_limit(outcomes, outcome_limit, int(seed) + 503),
    )


def _teacher_stratified_sample_quotas(
    group_quotas: Sequence[int],
    teacher_counts: Sequence[int],
    total_counts: Sequence[int],
    teacher_sample_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Split fixed source quotas into teacher/outcome quotas when feasible.

    Source quotas are never changed, so ``primary_sample_fraction`` remains an
    independent constraint. The requested teacher count is clamped only when
    the source quotas make it impossible (for example, a source has no deep
    labels and its quota cannot be transferred without violating the source
    mix). A non-empty stratum may be deterministically oversampled, which keeps
    the requested mix exact without changing source quotas or the total cap.
    """
    if not (
        len(group_quotas) == len(teacher_counts) == len(total_counts)
    ):
        raise ValueError("teacher quota inputs must have matching lengths")
    lower: List[int] = []
    upper: List[int] = []
    for quota, teacher_count, total_count in zip(
        group_quotas,
        teacher_counts,
        total_counts,
    ):
        q = max(0, int(quota))
        teachers = max(0, int(teacher_count))
        outcomes = max(0, int(total_count) - teachers)
        lower.append(q if outcomes == 0 else 0)
        upper.append(q if teachers > 0 else 0)

    total = sum(max(0, int(quota)) for quota in group_quotas)
    requested = int(round(total * _clamp(float(teacher_sample_fraction), 0.0, 1.0)))
    desired = min(max(requested, sum(lower)), sum(upper))
    capacities = [hi - lo for lo, hi in zip(lower, upper)]
    extra = _balanced_sample_quotas(capacities, desired - sum(lower))
    teacher_quotas = [lo + add for lo, add in zip(lower, extra)]
    outcome_quotas = [
        max(0, int(quota)) - teacher_quota
        for quota, teacher_quota in zip(group_quotas, teacher_quotas)
    ]
    return teacher_quotas, outcome_quotas


def iterate_samples(
    jsonl_dir: Path,
    max_samples: int,
    seed: int = 1,
    primary_sample_fraction: float = 0.5,
    teacher_sample_fraction: float = 0.5,
    min_teacher_depth: int = 0,
    require_teacher: bool = False,
) -> Iterator[Tuple[List[int], TrainingRecord]]:
    groups = _training_source_groups(Path(jsonl_dir))
    if max_samples <= 0:
        for group in groups:
            for record in _iter_usable_records(
                group,
                min_teacher_depth,
                require_teacher=require_teacher,
            ):
                yield _active_halfkp_indices(record.fen), record
        return

    counts: List[int] = []
    teacher_counts: List[int] = []
    for group in groups:
        total = 0
        teachers = 0
        for record in _iter_usable_records(
            group,
            min_teacher_depth,
            require_teacher=require_teacher,
        ):
            total += 1
            teachers += int(_teacher_available(record, min_teacher_depth))
        counts.append(total)
        teacher_counts.append(teachers)
    quotas = _primary_weighted_sample_quotas(
        counts,
        max_samples,
        primary_sample_fraction,
    )
    teacher_quotas, outcome_quotas = _teacher_stratified_sample_quotas(
        quotas,
        teacher_counts,
        counts,
        1.0 if require_teacher else teacher_sample_fraction,
    )
    samples: List[List[TrainingRecord]] = []
    for group_idx, (group, teacher_quota, outcome_quota) in enumerate(
        zip(groups, teacher_quotas, outcome_quotas)
    ):
        group_seed = int(seed) + (group_idx + 1) * 1_000_003
        teachers, outcomes = _reservoir_sample_record_strata(
            group,
            teacher_quota,
            outcome_quota,
            group_seed,
            min_teacher_depth,
            require_teacher=require_teacher,
        )
        sampled = teachers + outcomes
        random.Random(group_seed + 307).shuffle(sampled)
        samples.append(sampled)
    max_group_len = max((len(group) for group in samples), default=0)
    for sample_idx in range(max_group_len):
        for group in samples:
            if sample_idx < len(group):
                record = group[sample_idx]
                yield _active_halfkp_indices(record.fen), record


def _result_to_target_cp(result: int, target_cp: float) -> float:
    if result > 0:
        return float(target_cp)
    if result < 0:
        return -float(target_cp)
    return 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _target_cp_for_record(
    record: TrainingRecord,
    *,
    target_cp: float,
    teacher_mix: float,
    max_teacher_cp: float,
    outcome_decay: float = 1.0,
    min_teacher_depth: int = 0,
) -> float:
    teacher_available = _teacher_available(record, min_teacher_depth)
    if not record.outcome_valid:
        if not teacher_available:
            return 0.0
        return _clamp(float(record.value_cp), -float(max_teacher_cp), float(max_teacher_cp))

    result_q = float(record.result_q)
    if not math.isfinite(result_q):
        result_q = float(record.result)
    result_q = _clamp(result_q, -1.0, 1.0)
    outcome_cp = result_q * float(target_cp)
    if record.ply is not None and record.ply > 0 and outcome_decay < 0.999999:
        outcome_cp *= float(outcome_decay) ** int(record.ply)

    if not teacher_available:
        return outcome_cp

    teacher_cp = _clamp(float(record.value_cp), -float(max_teacher_cp), float(max_teacher_cp))
    mix = _clamp(float(teacher_mix), 0.0, 1.0)
    return mix * teacher_cp + (1.0 - mix) * outcome_cp


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _wdl_probability_to_cp(probability: float, wdl_scale_cp: float) -> float:
    """Return the finite CP value represented by a soft WDL probability."""
    bounded = _clamp(float(probability), 1e-6, 1.0 - 1e-6)
    return float(wdl_scale_cp) * math.log(bounded / (1.0 - bounded))


def _wdl_loss_gradient_cp(
    prediction_cp: float,
    target_probability: float,
    wdl_scale_cp: float,
) -> float:
    """Derivative of the reported BCE objective with respect to CP output."""
    scale = max(1e-6, float(wdl_scale_cp))
    target = _clamp(float(target_probability), 0.0, 1.0)
    return (_sigmoid(float(prediction_cp) / scale) - target) / scale


def objective_metadata(
    *,
    loss_kind: str,
    target_cp: float,
    teacher_mix: float,
    max_teacher_cp: float,
    outcome_decay: float,
    min_teacher_depth: int,
    huber_delta_cp: float,
    wdl_scale_cp: float,
) -> Dict[str, Any]:
    """Stable identity for target semantics carried by model/optimizer state."""
    return {
        "schema": OBJECTIVE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "loss_kind": str(loss_kind),
        "target_cp": float(target_cp),
        "teacher_mix": float(teacher_mix),
        "max_teacher_cp": float(max_teacher_cp),
        "outcome_decay": float(outcome_decay),
        "min_teacher_depth": int(min_teacher_depth),
        "huber_delta_cp": float(huber_delta_cp),
        "wdl_scale_cp": float(wdl_scale_cp),
    }


def _target_wdl_probability_for_record(
    record: TrainingRecord,
    *,
    teacher_mix: float,
    max_teacher_cp: float,
    wdl_scale_cp: float,
    min_teacher_depth: int = 0,
    outcome_decay: float = 1.0,
    target_cp: float = 100.0,
) -> float:
    teacher_available = _teacher_available(record, min_teacher_depth)
    scale = max(1e-6, float(wdl_scale_cp))
    if not record.outcome_valid:
        if not teacher_available:
            return 0.5
        teacher_cp = _clamp(
            float(record.value_cp),
            -float(max_teacher_cp),
            float(max_teacher_cp),
        )
        return _sigmoid(teacher_cp / scale)

    result_q = float(record.result_q)
    if not math.isfinite(result_q):
        result_q = float(record.result)
    result_q = _clamp(result_q, -1.0, 1.0)
    outcome_cp = result_q * float(target_cp)
    if record.ply is not None and record.ply > 0 and outcome_decay < 0.999999:
        outcome_cp *= float(outcome_decay) ** int(record.ply)
    # A decisive result is evidence worth +/-target_cp, not a hard 0/1 label.
    # Hard labels have no finite optimum when the network output is consumed
    # directly as centipawns by alpha-beta search.
    outcome_probability = _sigmoid(outcome_cp / scale)
    if not teacher_available:
        return outcome_probability

    teacher_cp = _clamp(
        float(record.value_cp),
        -float(max_teacher_cp),
        float(max_teacher_cp),
    )
    teacher_probability = _sigmoid(teacher_cp / scale)
    mix = _clamp(float(teacher_mix), 0.0, 1.0)
    return mix * teacher_probability + (1.0 - mix) * outcome_probability


def _targets_for_record(
    record: TrainingRecord,
    *,
    loss_kind: str,
    target_cp: float,
    teacher_mix: float,
    max_teacher_cp: float,
    outcome_decay: float,
    min_teacher_depth: int,
    wdl_scale_cp: float,
) -> Tuple[float, float]:
    """Build mutually consistent CP diagnostics and WDL objective targets."""
    probability = _target_wdl_probability_for_record(
        record,
        target_cp=target_cp,
        teacher_mix=teacher_mix,
        max_teacher_cp=max_teacher_cp,
        outcome_decay=outcome_decay,
        min_teacher_depth=min_teacher_depth,
        wdl_scale_cp=wdl_scale_cp,
    )
    if loss_kind == "wdl":
        cp = _wdl_probability_to_cp(probability, wdl_scale_cp)
    else:
        cp = _target_cp_for_record(
            record,
            target_cp=target_cp,
            teacher_mix=teacher_mix,
            max_teacher_cp=max_teacher_cp,
            outcome_decay=outcome_decay,
            min_teacher_depth=min_teacher_depth,
        )
    return cp, probability


def _eval_split(
    w1: List[float],
    b1: List[float],
    w2: List[float],
    b2: float,
    input_dim: int,
    xs: Sequence[Sequence[int]],
    cp_targets: Sequence[float],
    wdl_targets: Sequence[float],
    *,
    loss_kind: str = "mse",
    huber_delta_cp: float = 100.0,
    wdl_scale_cp: float = 400.0,
) -> Tuple[float, float, float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    loss_sum = 0.0
    cp_mse_sum = 0.0
    correct = 0
    prediction_abs_sum = 0.0
    prediction_max_abs = 0.0
    hidden_dim = len(b1)
    for i in range(len(xs)):
        act = xs[i]
        y = cp_targets[i]
        hpre = [0.0] * hidden_dim
        for j in range(hidden_dim):
            off = j * input_dim
            s = b1[j]
            for idx in act:
                s += w1[off + idx]
            hpre[j] = s
        h = [v if v > 0.0 else 0.0 for v in hpre]
        pred = b2
        for j in range(hidden_dim):
            pred += w2[j] * h[j]
        prediction_abs = abs(pred)
        prediction_abs_sum += prediction_abs
        prediction_max_abs = max(prediction_max_abs, prediction_abs)
        diff = pred - y
        cp_mse_sum += diff * diff
        if loss_kind == "mse":
            loss_sum += diff * diff
        elif loss_kind == "huber":
            delta = max(1e-6, float(huber_delta_cp))
            abs_diff = abs(diff)
            if abs_diff <= delta:
                loss_sum += 0.5 * diff * diff
            else:
                loss_sum += delta * (abs_diff - 0.5 * delta)
        else:
            probability = _clamp(float(wdl_targets[i]), 0.0, 1.0)
            logit = pred / max(1e-6, float(wdl_scale_cp))
            loss_sum += max(logit, 0.0) - logit * probability + math.log1p(
                math.exp(-abs(logit))
            )
        pred_label = 1 if pred > 1e-6 else (-1 if pred < -1e-6 else 0)
        true_label = 1 if y > 1e-6 else (-1 if y < -1e-6 else 0)
        if pred_label == true_label:
            correct += 1
    n = float(len(xs))
    return (
        loss_sum / n,
        cp_mse_sum / n,
        correct / n,
        prediction_abs_sum / n,
        prediction_max_abs,
    )


def train_model(
    *,
    jsonl_dir: Path,
    batch_size: int = 4096,
    max_samples: int = 200000,
    epochs: int = 8,
    val_split: float = 0.1,
    learning_rate: float = 0.05,
    hidden_dim: int = 16,
    target_cp: float = 100.0,
    teacher_mix: float = 0.7,
    max_teacher_cp: float = 1500.0,
    outcome_decay: float = 1.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 5.0,
    primary_sample_fraction: float = 0.5,
    teacher_sample_fraction: float = 0.5,
    min_teacher_depth: int = 0,
    loss_kind: str = "mse",
    huber_delta_cp: float = 100.0,
    wdl_scale_cp: float = 400.0,
    validation_jsonl_dir: Optional[Path] = None,
    max_validation_samples: int = 100_000,
    validation_seed: int = 20_260_802,
    validation_require_teacher: bool = False,
    seed: int = 1,
    out_dir: Path,
    initial_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    val_split = min(0.9, max(0.0, float(val_split)))
    lr = max(0.0, float(learning_rate))
    hidden_dim = max(1, int(hidden_dim))
    target_cp = max(1.0, float(target_cp))
    teacher_mix = _clamp(float(teacher_mix), 0.0, 1.0)
    max_teacher_cp = max(1.0, float(max_teacher_cp))
    outcome_decay = _clamp(float(outcome_decay), 0.0, 1.0)
    adam_beta1 = _clamp(float(adam_beta1), 0.0, 0.9999)
    adam_beta2 = _clamp(float(adam_beta2), 0.0, 0.99999)
    adam_eps = max(1e-12, float(adam_eps))
    grad_clip = max(0.0, float(grad_clip))
    primary_sample_fraction = _clamp(float(primary_sample_fraction), 0.0, 1.0)
    teacher_sample_fraction = _clamp(float(teacher_sample_fraction), 0.0, 1.0)
    min_teacher_depth = max(0, int(min_teacher_depth))
    loss_kind = str(loss_kind).strip().lower()
    if loss_kind not in {"mse", "huber", "wdl"}:
        raise ValueError("loss_kind must be one of: mse, huber, wdl")
    huber_delta_cp = max(1e-6, float(huber_delta_cp))
    wdl_scale_cp = max(1e-6, float(wdl_scale_cp))
    max_validation_samples = max(0, int(max_validation_samples))
    validation_seed = int(validation_seed)
    validation_require_teacher = bool(validation_require_teacher)
    if validation_jsonl_dir is not None:
        _assert_validation_source_disjoint(
            Path(validation_jsonl_dir),
            Path(jsonl_dir),
        )
    objective = objective_metadata(
        loss_kind=loss_kind,
        target_cp=target_cp,
        teacher_mix=teacher_mix,
        max_teacher_cp=max_teacher_cp,
        outcome_decay=outcome_decay,
        min_teacher_depth=min_teacher_depth,
        huber_delta_cp=huber_delta_cp,
        wdl_scale_cp=wdl_scale_cp,
    )

    xs: List[List[int]] = []
    cp_targets: List[float] = []
    wdl_targets: List[float] = []
    # Compact metadata is sufficient for the stable game/duplicate partition;
    # retaining 700k complete TrainingRecord objects costs several GB.
    validation_group_identities: List[str] = []
    validation_teacher_flags: List[bool] = []
    best_move_available = 0
    teacher_value_available = 0
    raw_teacher_value_available = 0
    for feats, record in iterate_samples(
        jsonl_dir,
        max_samples,
        seed=seed,
        primary_sample_fraction=primary_sample_fraction,
        teacher_sample_fraction=teacher_sample_fraction,
        min_teacher_depth=min_teacher_depth,
    ):
        xs.append(feats)
        validation_group_identities.append(_validation_group_identity(record))
        validation_teacher_flags.append(
            _teacher_available(record, min_teacher_depth)
        )
        cp, probability = _targets_for_record(
            record,
            loss_kind=loss_kind,
            target_cp=target_cp,
            teacher_mix=teacher_mix,
            max_teacher_cp=max_teacher_cp,
            outcome_decay=outcome_decay,
            min_teacher_depth=min_teacher_depth,
            wdl_scale_cp=wdl_scale_cp,
        )
        cp_targets.append(cp)
        wdl_targets.append(probability)
        if record.best_move:
            best_move_available += 1
        if record.value_cp is not None:
            raw_teacher_value_available += 1
        if _teacher_available(record, min_teacher_depth):
            teacher_value_available += 1

    if not xs:
        raise ValueError("no training samples were loaded")
    requested_teacher_samples = int(round(len(xs) * teacher_sample_fraction))
    teacher_sampling_satisfied = teacher_value_available == requested_teacher_samples
    if (
        max_samples > 0
        and 0 < teacher_value_available < len(xs)
        and not teacher_sampling_satisfied
    ):
        raise ValueError(
            "unable to satisfy teacher_sample_fraction while preserving source quotas: "
            f"requested {requested_teacher_samples}/{len(xs)}, "
            f"selected {teacher_value_available}/{len(xs)}"
        )

    dim = HALFKP_DIM
    rng = random.Random(seed)
    order = list(range(len(xs)))
    rng.shuffle(order)
    xs = [xs[i] for i in order]
    cp_targets = [cp_targets[i] for i in order]
    wdl_targets = [wdl_targets[i] for i in order]
    validation_group_identities = [
        validation_group_identities[i] for i in order
    ]
    validation_teacher_flags = [validation_teacher_flags[i] for i in order]

    fixed_validation = validation_jsonl_dir is not None
    validation_teacher_value_available: Optional[int] = None
    validation_raw_teacher_value_available: Optional[int] = None
    validation_sample_sha256: Optional[str] = None
    validation_source: Optional[Dict[str, Any]] = None
    train_indices, validation_indices = _internal_validation_partition_from_metadata(
        validation_group_identities,
        validation_teacher_flags,
        val_split,
        validation_seed=validation_seed,
    )
    train_x = [xs[idx] for idx in train_indices]
    train_cp = [cp_targets[idx] for idx in train_indices]
    train_wdl = [wdl_targets[idx] for idx in train_indices]
    val_x = [xs[idx] for idx in validation_indices]
    val_cp = [cp_targets[idx] for idx in validation_indices]
    val_wdl = [wdl_targets[idx] for idx in validation_indices]
    train_count = len(train_x)
    val_count = len(val_x)
    train_teacher_count = sum(
        validation_teacher_flags[idx] for idx in train_indices
    )
    primary_validation_teacher_count = sum(
        validation_teacher_flags[idx] for idx in validation_indices
    )
    train_groups = {
        validation_group_identities[idx] for idx in train_indices
    }
    validation_groups = {
        validation_group_identities[idx] for idx in validation_indices
    }
    internal_validation_record_overlap = len(
        train_groups.intersection(validation_groups)
    )

    reference_val_x: List[List[int]] = []
    reference_val_cp: List[float] = []
    reference_val_wdl: List[float] = []
    if fixed_validation:
        validation_teacher_value_available = 0
        validation_raw_teacher_value_available = 0
        validation_path = Path(validation_jsonl_dir)  # type: ignore[arg-type]
        validation_source_before = {
            "path": validation_path.resolve().as_posix(),
            "sha256": _sha256_jsonl_source(validation_path),
            "records": _count_jsonl_source_records(validation_path),
            "max_samples": max_validation_samples,
            "seed": validation_seed,
        }
        validation_digest = hashlib.sha256()
        for feats, record in iterate_fixed_validation_samples(
            validation_path,
            max_validation_samples,
            seed=validation_seed,
            min_teacher_depth=min_teacher_depth,
            require_teacher=validation_require_teacher,
        ):
            validation_digest.update(_record_identity(record).encode("utf-8"))
            validation_digest.update(b"\0")
            if record.value_cp is not None:
                validation_raw_teacher_value_available += 1
            if _teacher_available(record, min_teacher_depth):
                validation_teacher_value_available += 1
            cp, probability = _targets_for_record(
                record,
                loss_kind=loss_kind,
                target_cp=target_cp,
                teacher_mix=teacher_mix,
                max_teacher_cp=max_teacher_cp,
                outcome_decay=outcome_decay,
                min_teacher_depth=min_teacher_depth,
                wdl_scale_cp=wdl_scale_cp,
            )
            reference_val_x.append(feats)
            reference_val_cp.append(cp)
            reference_val_wdl.append(probability)
        if not reference_val_x:
            raise ValueError("external reference validation dataset contains no usable samples")
        validation_sample_sha256 = validation_digest.hexdigest()
        validation_source = {
            "path": validation_path.resolve().as_posix(),
            "sha256": _sha256_jsonl_source(validation_path),
            "records": _count_jsonl_source_records(validation_path),
            "max_samples": max_validation_samples,
            "seed": validation_seed,
        }
        if validation_source != validation_source_before:
            raise ValueError("fixed validation source changed while trainer was reading it")
    reference_val_count = len(reference_val_x)

    # Small random init.
    w1 = [(rng.random() - 0.5) * 0.01 for _ in range(hidden_dim * dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [(rng.random() - 0.5) * 0.01 for _ in range(hidden_dim)]
    b2 = 0.0
    initialized_from = None
    if initial_checkpoint is not None:
        w1, b1, w2, b2, initialized_from = _load_initial_checkpoint(
            Path(initial_checkpoint),
            input_dim=dim,
            hidden_dim=hidden_dim,
            objective=objective,
        )
    best_w1 = list(w1)
    best_b1 = list(b1)
    best_w2 = list(w2)
    best_b2 = b2

    # Adam moments.
    m_w1 = [0.0 for _ in range(hidden_dim * dim)]
    v_w1 = [0.0 for _ in range(hidden_dim * dim)]
    m_b1 = [0.0 for _ in range(hidden_dim)]
    v_b1 = [0.0 for _ in range(hidden_dim)]
    m_w2 = [0.0 for _ in range(hidden_dim)]
    v_w2 = [0.0 for _ in range(hidden_dim)]
    m_b2 = 0.0
    v_b2 = 0.0
    adam_t = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_cp_mse_history: List[float] = []
    val_cp_mse_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []
    train_prediction_mean_abs_history: List[float] = []
    val_prediction_mean_abs_history: List[float] = []
    train_prediction_max_abs_history: List[float] = []
    val_prediction_max_abs_history: List[float] = []
    reference_val_loss_history: List[float] = []
    reference_val_cp_mse_history: List[float] = []
    reference_val_acc_history: List[float] = []
    reference_val_prediction_mean_abs_history: List[float] = []
    reference_val_prediction_max_abs_history: List[float] = []
    reference_val_checkpoint_eligible_history: List[bool] = []
    best_val_loss = float("inf")
    best_epoch = 0
    initial_train_loss = None
    initial_train_acc = None
    initial_val_loss = None
    initial_val_acc = None
    initial_train_cp_mse = None
    initial_val_cp_mse = None
    initial_train_prediction_mean_abs = None
    initial_val_prediction_mean_abs = None
    initial_train_prediction_max_abs = None
    initial_val_prediction_max_abs = None
    initial_reference_val_loss = None
    initial_reference_val_cp_mse = None
    initial_reference_val_acc = None
    initial_reference_val_prediction_mean_abs = None
    initial_reference_val_prediction_max_abs = None
    best_reference_val_loss = None
    best_reference_val_cp_mse = None
    best_reference_val_acc = None
    best_reference_val_prediction_mean_abs = None
    best_reference_val_prediction_max_abs = None
    if initialized_from is not None:
        (
            initial_train_loss,
            initial_train_cp_mse,
            initial_train_acc,
            initial_train_prediction_mean_abs,
            initial_train_prediction_max_abs,
        ) = _eval_split(
            w1,
            b1,
            w2,
            b2,
            dim,
            train_x,
            train_cp,
            train_wdl,
            loss_kind=loss_kind,
            huber_delta_cp=huber_delta_cp,
            wdl_scale_cp=wdl_scale_cp,
        )
        if val_count > 0:
            (
                initial_val_loss,
                initial_val_cp_mse,
                initial_val_acc,
                initial_val_prediction_mean_abs,
                initial_val_prediction_max_abs,
            ) = _eval_split(
                w1,
                b1,
                w2,
                b2,
                dim,
                val_x,
                val_cp,
                val_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            initial_val_loss, initial_val_acc = initial_train_loss, initial_train_acc
            initial_val_cp_mse = initial_train_cp_mse
            initial_val_prediction_mean_abs = initial_train_prediction_mean_abs
            initial_val_prediction_max_abs = initial_train_prediction_max_abs
        if reference_val_count > 0:
            (
                initial_reference_val_loss,
                initial_reference_val_cp_mse,
                initial_reference_val_acc,
                initial_reference_val_prediction_mean_abs,
                initial_reference_val_prediction_max_abs,
            ) = _eval_split(
                w1,
                b1,
                w2,
                b2,
                dim,
                reference_val_x,
                reference_val_cp,
                reference_val_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
            best_reference_val_loss = initial_reference_val_loss
            best_reference_val_cp_mse = initial_reference_val_cp_mse
            best_reference_val_acc = initial_reference_val_acc
            best_reference_val_prediction_mean_abs = (
                initial_reference_val_prediction_mean_abs
            )
            best_reference_val_prediction_max_abs = (
                initial_reference_val_prediction_max_abs
            )
        best_val_loss = float(initial_val_loss)

    for epoch in range(epochs):
        idx = list(range(train_count))
        rng.shuffle(idx)

        for start in range(0, train_count, batch_size):
            batch_idx = idx[start:start + batch_size]
            if not batch_idx:
                continue

            gw1 = [0.0 for _ in range(hidden_dim * dim)]
            gb1 = [0.0 for _ in range(hidden_dim)]
            gw2 = [0.0 for _ in range(hidden_dim)]
            gb2 = 0.0

            for i in batch_idx:
                act = train_x[i]
                target = train_cp[i]
                hpre = [0.0 for _ in range(hidden_dim)]
                for j in range(hidden_dim):
                    off = j * dim
                    s = b1[j]
                    for f in act:
                        s += w1[off + f]
                    hpre[j] = s
                h = [v if v > 0.0 else 0.0 for v in hpre]
                pred = b2
                for j in range(hidden_dim):
                    pred += w2[j] * h[j]
                diff = pred - target
                if loss_kind == "mse":
                    dloss_dpred = 2.0 * diff
                elif loss_kind == "huber":
                    if abs(diff) <= huber_delta_cp:
                        dloss_dpred = diff
                    else:
                        dloss_dpred = math.copysign(huber_delta_cp, diff)
                else:
                    dloss_dpred = _wdl_loss_gradient_cp(
                        pred,
                        train_wdl[i],
                        wdl_scale_cp,
                    )

                gb2 += dloss_dpred
                for j in range(hidden_dim):
                    gw2[j] += dloss_dpred * h[j]
                for j in range(hidden_dim):
                    if hpre[j] <= 0.0:
                        continue
                    dpre = dloss_dpred * w2[j]
                    gb1[j] += dpre
                    off = j * dim
                    for f in act:
                        gw1[off + f] += dpre

            scale = 1.0 / float(len(batch_idx))
            if scale != 1.0:
                for j in range(hidden_dim * dim):
                    gw1[j] *= scale
                for j in range(hidden_dim):
                    gb1[j] *= scale
                    gw2[j] *= scale
                gb2 *= scale

            if grad_clip > 0.0:
                norm2 = gb2 * gb2
                for j in range(hidden_dim):
                    norm2 += gb1[j] * gb1[j]
                    norm2 += gw2[j] * gw2[j]
                for j in range(hidden_dim * dim):
                    norm2 += gw1[j] * gw1[j]
                norm = math.sqrt(norm2)
                if norm > grad_clip:
                    gscale = grad_clip / (norm + 1e-12)
                    for j in range(hidden_dim * dim):
                        gw1[j] *= gscale
                    for j in range(hidden_dim):
                        gb1[j] *= gscale
                        gw2[j] *= gscale
                    gb2 *= gscale

            adam_t += 1
            bc1 = 1.0 - (adam_beta1 ** adam_t)
            bc2 = 1.0 - (adam_beta2 ** adam_t)

            for j in range(hidden_dim * dim):
                g = gw1[j]
                m_w1[j] = adam_beta1 * m_w1[j] + (1.0 - adam_beta1) * g
                v_w1[j] = adam_beta2 * v_w1[j] + (1.0 - adam_beta2) * g * g
                mhat = m_w1[j] / bc1
                vhat = v_w1[j] / bc2
                w1[j] -= lr * mhat / (math.sqrt(vhat) + adam_eps)

            for j in range(hidden_dim):
                g1 = gb1[j]
                m_b1[j] = adam_beta1 * m_b1[j] + (1.0 - adam_beta1) * g1
                v_b1[j] = adam_beta2 * v_b1[j] + (1.0 - adam_beta2) * g1 * g1
                mhat1 = m_b1[j] / bc1
                vhat1 = v_b1[j] / bc2
                b1[j] -= lr * mhat1 / (math.sqrt(vhat1) + adam_eps)

                g2 = gw2[j]
                m_w2[j] = adam_beta1 * m_w2[j] + (1.0 - adam_beta1) * g2
                v_w2[j] = adam_beta2 * v_w2[j] + (1.0 - adam_beta2) * g2 * g2
                mhat2 = m_w2[j] / bc1
                vhat2 = v_w2[j] / bc2
                w2[j] -= lr * mhat2 / (math.sqrt(vhat2) + adam_eps)

            m_b2 = adam_beta1 * m_b2 + (1.0 - adam_beta1) * gb2
            v_b2 = adam_beta2 * v_b2 + (1.0 - adam_beta2) * gb2 * gb2
            b2 -= lr * (m_b2 / bc1) / (math.sqrt(v_b2 / bc2) + adam_eps)

        (
            train_loss,
            train_cp_mse,
            train_acc,
            train_prediction_mean_abs,
            train_prediction_max_abs,
        ) = _eval_split(
            w1,
            b1,
            w2,
            b2,
            dim,
            train_x,
            train_cp,
            train_wdl,
            loss_kind=loss_kind,
            huber_delta_cp=huber_delta_cp,
            wdl_scale_cp=wdl_scale_cp,
        )
        if val_count > 0:
            (
                val_loss,
                val_cp_mse,
                val_acc,
                val_prediction_mean_abs,
                val_prediction_max_abs,
            ) = _eval_split(
                w1,
                b1,
                w2,
                b2,
                dim,
                val_x,
                val_cp,
                val_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            val_loss, val_acc = train_loss, train_acc
            val_cp_mse = train_cp_mse
            val_prediction_mean_abs = train_prediction_mean_abs
            val_prediction_max_abs = train_prediction_max_abs
        if reference_val_count > 0:
            (
                reference_val_loss,
                reference_val_cp_mse,
                reference_val_acc,
                reference_val_prediction_mean_abs,
                reference_val_prediction_max_abs,
            ) = _eval_split(
                w1,
                b1,
                w2,
                b2,
                dim,
                reference_val_x,
                reference_val_cp,
                reference_val_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            reference_val_loss = None
            reference_val_cp_mse = None
            reference_val_acc = None
            reference_val_prediction_mean_abs = None
            reference_val_prediction_max_abs = None
        reference_checkpoint_eligible = True
        if (
            reference_val_loss is not None
            and initial_reference_val_loss is not None
        ):
            reference_limit = float(initial_reference_val_loss) * (
                1.0 + REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
            )
            reference_checkpoint_eligible = (
                math.isfinite(float(reference_val_loss))
                and float(reference_val_loss) <= reference_limit + 1e-12
            )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_cp_mse_history.append(train_cp_mse)
        val_cp_mse_history.append(val_cp_mse)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_prediction_mean_abs_history.append(train_prediction_mean_abs)
        val_prediction_mean_abs_history.append(val_prediction_mean_abs)
        train_prediction_max_abs_history.append(train_prediction_max_abs)
        val_prediction_max_abs_history.append(val_prediction_max_abs)
        if reference_val_loss is not None:
            reference_val_loss_history.append(reference_val_loss)
            reference_val_cp_mse_history.append(reference_val_cp_mse)
            reference_val_acc_history.append(reference_val_acc)
            reference_val_prediction_mean_abs_history.append(
                reference_val_prediction_mean_abs
            )
            reference_val_prediction_max_abs_history.append(
                reference_val_prediction_max_abs
            )
        reference_val_checkpoint_eligible_history.append(
            reference_checkpoint_eligible
        )

        if is_better_checkpoint(
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            reference_val_loss=reference_val_loss,
            best_reference_val_loss=best_reference_val_loss,
            initial_reference_val_loss=initial_reference_val_loss,
        ):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_w1 = list(w1)
            best_b1 = list(b1)
            best_w2 = list(w2)
            best_b2 = b2
            best_reference_val_loss = reference_val_loss
            best_reference_val_cp_mse = reference_val_cp_mse
            best_reference_val_acc = reference_val_acc
            best_reference_val_prediction_mean_abs = (
                reference_val_prediction_mean_abs
            )
            best_reference_val_prediction_max_abs = (
                reference_val_prediction_max_abs
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "piebot-halfkp-mse-v2",
        "target_schema": TARGET_SCHEMA,
        "objective": objective,
        "input_dim": dim,
        "hidden_dim": hidden_dim,
        "w1": best_w1,
        "b1": best_b1,
        "w2": best_w2,
        "b2": best_b2,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "feature_set": FEATURE_SET,
        "loss_kind": loss_kind,
        "huber_delta_cp": huber_delta_cp,
        "wdl_scale_cp": wdl_scale_cp,
        "min_teacher_depth": min_teacher_depth,
        "primary_sample_fraction": primary_sample_fraction,
        "teacher_sample_fraction": teacher_sample_fraction,
        "sampling_schema": SAMPLING_SCHEMA,
        "validation_sampling_schema": PRIMARY_VALIDATION_SAMPLING_SCHEMA,
        "reference_validation_sampling_schema": FIXED_VALIDATION_SAMPLING_SCHEMA,
        "checkpoint_selection_schema": CHECKPOINT_SELECTION_SCHEMA,
        "reference_validation_max_relative_loss_regression": (
            REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_max_relative_loss_regression": (
            PRIMARY_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_hash_namespace": PRIMARY_VALIDATION_HASH_NAMESPACE,
        "validation_seed": validation_seed,
        "validation_require_teacher": validation_require_teacher,
        "validation_source": validation_source,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "initialized_from": initialized_from,
        "optimizer_state_restored": False,
    }
    metrics = {
        "train_samples": train_count,
        "val_samples": val_count,
        "input_dim": dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "feature_set": FEATURE_SET,
        "target_schema": TARGET_SCHEMA,
        "objective": objective,
        "primary_sample_fraction": primary_sample_fraction,
        "teacher_sample_fraction": teacher_sample_fraction,
        "sampling_schema": SAMPLING_SCHEMA,
        "validation_sampling_schema": PRIMARY_VALIDATION_SAMPLING_SCHEMA,
        "reference_validation_sampling_schema": FIXED_VALIDATION_SAMPLING_SCHEMA,
        "checkpoint_selection_schema": CHECKPOINT_SELECTION_SCHEMA,
        "reference_validation_max_relative_loss_regression": (
            REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_max_relative_loss_regression": (
            PRIMARY_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_hash_namespace": PRIMARY_VALIDATION_HASH_NAMESPACE,
        "min_teacher_depth": min_teacher_depth,
        "loss_kind": loss_kind,
        "huber_delta_cp": huber_delta_cp,
        "wdl_scale_cp": wdl_scale_cp,
        "validation_jsonl_dir": (
            Path(validation_jsonl_dir).resolve().as_posix()
            if validation_jsonl_dir is not None
            else None
        ),
        "max_validation_samples": max_validation_samples,
        "validation_seed": validation_seed,
        "validation_require_teacher": validation_require_teacher,
        "validation_source": validation_source,
        "fixed_validation": fixed_validation,
        "reference_val_samples": reference_val_count,
        "train_records_with_teacher_value": train_teacher_count,
        "primary_validation_records_with_teacher_value": (
            primary_validation_teacher_count
        ),
        "primary_validation_teacher_sample_fraction": (
            float(primary_validation_teacher_count) / float(val_count)
            if val_count
            else 0.0
        ),
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_eps": adam_eps,
        "grad_clip": grad_clip,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_reference_val_loss": best_reference_val_loss,
        "best_reference_val_cp_mse": best_reference_val_cp_mse,
        "best_reference_val_acc": best_reference_val_acc,
        "best_reference_val_prediction_mean_abs": (
            best_reference_val_prediction_mean_abs
        ),
        "best_reference_val_prediction_max_abs": (
            best_reference_val_prediction_max_abs
        ),
        "initial_train_loss": initial_train_loss,
        "initial_train_acc": initial_train_acc,
        "initial_val_loss": initial_val_loss,
        "initial_val_acc": initial_val_acc,
        "initial_train_cp_mse": initial_train_cp_mse,
        "initial_val_cp_mse": initial_val_cp_mse,
        "initial_train_prediction_mean_abs": initial_train_prediction_mean_abs,
        "initial_val_prediction_mean_abs": initial_val_prediction_mean_abs,
        "initial_train_prediction_max_abs": initial_train_prediction_max_abs,
        "initial_val_prediction_max_abs": initial_val_prediction_max_abs,
        "initial_reference_val_loss": initial_reference_val_loss,
        "initial_reference_val_cp_mse": initial_reference_val_cp_mse,
        "initial_reference_val_acc": initial_reference_val_acc,
        "initial_reference_val_prediction_mean_abs": (
            initial_reference_val_prediction_mean_abs
        ),
        "initial_reference_val_prediction_max_abs": (
            initial_reference_val_prediction_max_abs
        ),
        "initialized_from": initialized_from,
        "optimizer_state_restored": False,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_cp_mse_history": train_cp_mse_history,
        "val_cp_mse_history": val_cp_mse_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "train_prediction_mean_abs_history": train_prediction_mean_abs_history,
        "val_prediction_mean_abs_history": val_prediction_mean_abs_history,
        "train_prediction_max_abs_history": train_prediction_max_abs_history,
        "val_prediction_max_abs_history": val_prediction_max_abs_history,
        "reference_val_loss_history": reference_val_loss_history,
        "reference_val_cp_mse_history": reference_val_cp_mse_history,
        "reference_val_acc_history": reference_val_acc_history,
        "reference_val_prediction_mean_abs_history": (
            reference_val_prediction_mean_abs_history
        ),
        "reference_val_prediction_max_abs_history": (
            reference_val_prediction_max_abs_history
        ),
        "reference_val_checkpoint_eligible_history": (
            reference_val_checkpoint_eligible_history
        ),
        "records_with_best_move": best_move_available,
        "records_with_teacher_value": teacher_value_available,
        "records_with_raw_teacher_value": raw_teacher_value_available,
        "records_total": len(xs),
        "validation_records_with_teacher_value": validation_teacher_value_available,
        "validation_records_with_raw_teacher_value": validation_raw_teacher_value_available,
        "validation_sample_sha256": validation_sample_sha256,
        "internal_validation_record_overlap": internal_validation_record_overlap,
        "actual_teacher_sample_fraction": (
            float(teacher_value_available) / float(len(xs)) if xs else 0.0
        ),
        "requested_teacher_samples": requested_teacher_samples,
        "teacher_sampling_satisfied": teacher_sampling_satisfied,
        "train_target_cp_mean_abs": (
            sum(abs(value) for value in train_cp) / float(len(train_cp))
            if train_cp
            else 0.0
        ),
        "train_target_cp_max_abs": max((abs(value) for value in train_cp), default=0.0),
        "val_target_cp_mean_abs": (
            sum(abs(value) for value in val_cp) / float(len(val_cp))
            if val_cp
            else 0.0
        ),
        "val_target_cp_max_abs": max((abs(value) for value in val_cp), default=0.0),
        "reference_val_target_cp_mean_abs": (
            sum(abs(value) for value in reference_val_cp)
            / float(len(reference_val_cp))
            if reference_val_cp
            else 0.0
        ),
        "reference_val_target_cp_max_abs": max(
            (abs(value) for value in reference_val_cp),
            default=0.0,
        ),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--initial-checkpoint", type=Path, default=None)
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--primary-sample-fraction", type=float, default=0.5)
    ap.add_argument("--teacher-sample-fraction", type=float, default=0.5)
    ap.add_argument("--min-teacher-depth", type=int, default=0)
    ap.add_argument("--loss-kind", choices=["mse", "huber", "wdl"], default="mse")
    ap.add_argument("--huber-delta-cp", type=float, default=100.0)
    ap.add_argument("--wdl-scale-cp", type=float, default=400.0)
    ap.add_argument("--validation-jsonl-dir", type=Path, default=None)
    ap.add_argument("--max-validation-samples", type=int, default=100_000)
    ap.add_argument("--validation-seed", type=int, default=20_260_802)
    ap.add_argument("--validation-require-teacher", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("out/nnue_stub_train"))
    args = ap.parse_args(argv)

    metrics = train_model(
        jsonl_dir=args.jsonl_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        target_cp=args.target_cp,
        teacher_mix=args.teacher_mix,
        max_teacher_cp=args.max_teacher_cp,
        outcome_decay=args.outcome_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        primary_sample_fraction=args.primary_sample_fraction,
        teacher_sample_fraction=args.teacher_sample_fraction,
        min_teacher_depth=args.min_teacher_depth,
        loss_kind=args.loss_kind,
        huber_delta_cp=args.huber_delta_cp,
        wdl_scale_cp=args.wdl_scale_cp,
        validation_jsonl_dir=args.validation_jsonl_dir,
        max_validation_samples=args.max_validation_samples,
        validation_seed=args.validation_seed,
        validation_require_teacher=args.validation_require_teacher,
        seed=args.seed,
        out_dir=args.out,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Val samples: {metrics['val_samples']}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val loss: {metrics['best_val_loss']:.6f}")
    print(f"Wrote: {(args.out / 'checkpoint.json').as_posix()}")
    print(f"Wrote: {(args.out / 'metrics.json').as_posix()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
