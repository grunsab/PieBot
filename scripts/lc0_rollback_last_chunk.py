#!/usr/bin/env python3
"""Step a stopped LCZero lineage back one completed chunk.

For a lineage whose newest committed chunk cannot be resumed from: its saved
weights and Adam state disagree, so the trainer refuses it on every start. The
trainer has no way back on its own, because it re-adopts an existing
complete.json. This utility moves that chunk's directory out of chunks/ and
rewrites lc0_state.json to the boundary before it. The next start then resumes
from the previous chunk and trains the discarded one again with the same seed,
corpus chunk and learning rate.

The utility intentionally does not stop the trainer or control supervisor. The
operator attests the supervisor stop, and the launcher and trainer locks are
held throughout. The directory is moved before the state is rewritten: a crash
between the two leaves a state whose checkpoint is missing, which the trainer
refuses loudly, and a rerun with the same arguments finishes the job.

Standard library only, so a copy can run outside the pinned trainer checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - the production Vast host is Linux.
    fcntl = None  # type: ignore[assignment]


STATE_SCHEMA = "piebot-lc0-autopilot-v1"
AUDIT_SCHEMA = "piebot-lc0-chunk-rollback-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RollbackError(RuntimeError):
    """Raised when the rollback cannot be proven safe."""


@dataclass(frozen=True)
class RollbackResult:
    status: str
    completed_chunks: int
    audit_path: Path
    discarded_directory: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode(obj: Any) -> bytes:
    # Exactly how training.nnue.autopilot._atomic_write_json serialises state.
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        os.fchmod(descriptor, mode)
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _held_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise RollbackError("fcntl locking is required for an LC0 chunk rollback")
    if path.is_symlink() or not path.is_file():
        raise RollbackError(f"lock file is missing: {path}")
    with path.open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RollbackError(f"trainer appears to be running; {path.name} is held") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_json(path: Path, *, label: str) -> tuple[Any, bytes]:
    if path.is_symlink() or not path.is_file():
        raise RollbackError(f"{label} is missing: {path}")
    raw = path.read_bytes()
    try:
        return json.loads(raw), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RollbackError(f"{label} is not valid JSON: {exc}") from exc


def _verify_resume_artifacts(state: dict[str, Any], latest: dict[str, Any]) -> None:
    # Everything training.nnue.lc0_autopilot checks before resuming, plus the
    # exported net of the chunk being resumed from, which the backups require.
    directory = Path(latest["directory"])
    checks = [(state["active_model_path"], state["active_model_sha256"]),
              (state["training_checkpoint_path"], latest["checkpoint_sha256"]),
              (state["training_optimizer_path"], latest["optimizer_sha256"]),
              (directory / "candidate.nnue", latest["quant_sha256"])]
    if state.get("best_quant_path"):
        checks.extend([(state["best_quant_path"], state["best_quant_sha256"]),
                       (state["best_checkpoint_path"], state["best_checkpoint_sha256"])])
    for raw, sha in checks:
        path = Path(raw)
        if not path.is_file() or _sha256_file(path) != sha:
            raise RollbackError(f"committed artifact checksum mismatch: {path}")


def _rollback(*, out_root: Path, expected_completed_chunks: int, expected_checkpoint_sha256: str,
              dry_run: bool) -> RollbackResult:
    training = out_root / "training"
    state_path = training / "lc0_state.json"
    state, raw = _load_json(state_path, label="LC0 state")
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise RollbackError("unsupported LC0 state schema")
    if _encode(state) != raw:
        raise RollbackError("LC0 state is not in the trainer's serialisation; refusing to rewrite it")
    history = state.get("history")
    if not isinstance(history, list) or state.get("completed_chunks") != len(history):
        raise RollbackError("LC0 history/completed count mismatch")

    number = expected_completed_chunks - 1
    home = training / "chunks" / f"chunk_{number:08d}"
    slot = training / "rollbacks" / f"{home.name}_{expected_checkpoint_sha256[:12]}"
    kept = slot / home.name
    audit_path = slot / "rollback.json"

    if len(history) == number and audit_path.is_file() and kept.is_dir():
        audit, _ = _load_json(audit_path, label="rollback audit")
        if (audit.get("schema") == AUDIT_SCHEMA and history and history[-1] == audit.get("resumes_from")
                and audit.get("discarded", {}).get("checkpoint_sha256") == expected_checkpoint_sha256):
            return RollbackResult("already-rolled-back", number, audit_path, kept)
    if len(history) != expected_completed_chunks:
        raise RollbackError(f"completed_chunks is {len(history)}, expected {expected_completed_chunks}")
    if number < 1:
        raise RollbackError("no earlier chunk to resume from")
    discarded, previous = history[-1], history[-2]
    if discarded["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise RollbackError(
            f"newest checkpoint is {discarded['checkpoint_sha256']}, expected {expected_checkpoint_sha256}")
    resume = home.with_name(f"chunk_{number - 1:08d}")
    if (discarded["number"], previous["number"]) != (number, number - 1):
        raise RollbackError("history numbers do not match the completed count")
    if (Path(discarded["directory"]), Path(previous["directory"])) != (home, resume):
        raise RollbackError("history directories are not this lineage's chunk directories")
    if (state["training_checkpoint_path"] != str(home / "train/checkpoint.json")
            or state["training_optimizer_path"] != str(home / "train/optimizer.pt")):
        raise RollbackError("learner paths do not point at the newest chunk")
    for key in ("best_checkpoint_path", "best_quant_path", "active_model_path"):
        if state.get(key) and home in Path(state[key]).parents:
            raise RollbackError(f"the newest chunk is the lineage's {key.split('_')[0]} model; "
                                "found a successor lineage from an earlier checkpoint instead")
    position = (state["pass_number"], state["cursor"])
    if position not in ((discarded["pass_number"], discarded["cursor"] + 1), (discarded["pass_number"] + 1, 0)):
        raise RollbackError("cursor does not follow the newest chunk")
    pending = state.get("in_progress")
    if pending is not None and (pending.get("number"), pending.get("pass_number"),
                                pending.get("cursor")) != (len(history), *position):
        raise RollbackError("in_progress is not the chunk after the newest one")

    moved = not home.exists() and kept.is_dir()
    current = kept if moved else home
    if current.is_symlink() or not current.is_dir() or (not moved and kept.exists()):
        raise RollbackError(f"newest chunk directory is not where it is expected: {home}")
    if _load_json(current / "complete.json", label="chunk completion record")[0] != discarded:
        raise RollbackError(f"complete.json does not match the newest history record: {current}")

    after = dict(state)
    after.update(
        history=history[:-1], completed_chunks=number, pass_number=discarded["pass_number"],
        cursor=discarded["cursor"], in_progress=None, evaluation_pending=False, status="paused",
        last_error=None, training_checkpoint_path=str(resume / "train/checkpoint.json"),
        training_optimizer_path=str(resume / "train/optimizer.pt"))
    _verify_resume_artifacts(after, previous)
    encoded = _encode(after)
    if dry_run:
        return RollbackResult("dry-run", number, audit_path, kept)

    before_sha = hashlib.sha256(raw).hexdigest()
    before_copy = slot / f"lc0_state.before.{before_sha[:12]}.json"
    mode = stat.S_IMODE(state_path.stat().st_mode)
    slot.mkdir(parents=True, exist_ok=True)
    _fsync_directory(slot.parent)
    _fsync_directory(training)
    _atomic_write(before_copy, raw, mode)
    _atomic_write(audit_path, _encode({
        "schema": AUDIT_SCHEMA,
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "state_path": str(state_path), "state_before_sha256": before_sha,
        "state_before_copy": str(before_copy), "state_after_sha256": hashlib.sha256(encoded).hexdigest(),
        "discarded": discarded, "resumes_from": previous, "discarded_directory": str(kept),
        "fields": {key: {"before": state.get(key), "after": value}
                   for key, value in after.items() if key != "history" and state.get(key) != value},
    }) + b"\n", mode)
    if not moved:
        os.rename(home, kept)
        _fsync_directory(home.parent)
        _fsync_directory(slot)
    _atomic_write(state_path, encoded, mode)
    return RollbackResult("rolled-back", number, audit_path, kept)


def rollback_last_chunk(*, out_root: Path, expected_completed_chunks: int, expected_checkpoint_sha256: str,
                        supervisor_stop_verified: bool, dry_run: bool = False) -> RollbackResult:
    """Discard the newest completed chunk of the lineage under out_root."""
    if not supervisor_stop_verified:
        raise RollbackError("the supervisor stop of the trainer must be verified first")
    if isinstance(expected_completed_chunks, bool) or not isinstance(expected_completed_chunks, int):
        raise RollbackError("expected completed chunks must be an integer")
    if not _SHA256_RE.fullmatch(str(expected_checkpoint_sha256)):
        raise RollbackError("expected checkpoint sha256 must be 64 lowercase hexadecimal characters")
    out_root = Path(out_root).resolve()
    with _held_lock(out_root / "launcher.lock"), _held_lock(out_root / "training" / "lc0.lock"):
        try:
            return _rollback(out_root=out_root, expected_completed_chunks=expected_completed_chunks,
                             expected_checkpoint_sha256=expected_checkpoint_sha256, dry_run=dry_run)
        except (KeyError, TypeError, AttributeError) as exc:
            raise RollbackError(f"LC0 state is missing an expected field: {exc!r}") from exc


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="lineage root (the launcher's OUT_ROOT, holding launcher.lock and training/)")
    parser.add_argument("--expected-completed-chunks", type=int, required=True,
                        help="completed_chunks in the state now, before the rollback")
    parser.add_argument("--expected-checkpoint-sha256", required=True,
                        help="recorded sha256 of the newest chunk's checkpoint, the one to discard")
    parser.add_argument("--supervisor-stop-verified", action="store_true",
                        help="attest supervisor STOPPED and all descendants exited; externally verified, not tool-checked")
    parser.add_argument("--dry-run", action="store_true", help="run every check and change nothing")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        result = rollback_last_chunk(
            out_root=args.out_root, expected_completed_chunks=args.expected_completed_chunks,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            supervisor_stop_verified=args.supervisor_stop_verified, dry_run=args.dry_run)
    except RollbackError as exc:
        print(f"LC0 chunk rollback refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result.status, "completed_chunks": result.completed_chunks,
                      "audit_path": str(result.audit_path),
                      "discarded_directory": str(result.discarded_directory)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
