#!/usr/bin/env python3
"""Audit and advance the source pin for a stopped restart-safe Vast.ai run.

The utility intentionally does not stop the trainer.  It acquires the existing
autopilot lock nonblocking and refuses to continue if another process owns it.
The prepared audit is durable before the source pin is replaced, which makes a
retry safe after a crash at either side of the atomic pin update.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - the production Vast host is Linux.
    fcntl = None  # type: ignore[assignment]


AUDIT_SCHEMA = "piebot-source-commit-migration-v1"
AUDIT_PHASE = "prepared"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MigrationError(RuntimeError):
    """Raised when a source transition cannot be proven safe."""


@dataclass(frozen=True)
class MigrationResult:
    status: str
    audit_path: Path
    old_commit: str
    new_commit: str


def audit_path_for(out_root: Path, old_commit: str, new_commit: str) -> Path:
    """Return the immutable prepared-audit path for one commit transition."""
    return (
        Path(out_root)
        / "source_commit_migrations"
        / f"{old_commit}_to_{new_commit}.prepared.json"
    )


def _require_commit(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise MigrationError(f"{label} must be a lowercase 40-character Git SHA-1")
    return value


def _require_directory(path: Path, *, label: str) -> Path:
    path = Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise MigrationError(f"{label} must be a real directory, not a symlink: {path}")
    return path.resolve()


def _require_regular_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise MigrationError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=check,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MigrationError("git is unavailable") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Git command failed").strip()
        raise MigrationError(f"git {' '.join(args)} failed: {detail}") from exc


def _validate_repository(
    repo_root: Path,
    *,
    old_commit: str,
    new_commit: str,
) -> None:
    top_level = _git(repo_root, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(top_level).resolve() != repo_root:
        raise MigrationError(
            f"repo root is not the Git top level: expected {repo_root}, got {top_level}"
        )
    dirty = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if dirty:
        raise MigrationError(
            "repository is not clean; commit or remove all tracked/untracked changes"
        )
    head = _git(repo_root, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if head != new_commit:
        raise MigrationError(f"new commit does not match repository HEAD: {head}")
    for label, commit in (("old", old_commit), ("new", new_commit)):
        exists = _git(
            repo_root,
            "cat-file",
            "-e",
            f"{commit}^{{commit}}",
            check=False,
        )
        if exists.returncode != 0:
            raise MigrationError(f"{label} commit is not present in the repository: {commit}")
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        old_commit,
        new_commit,
        check=False,
    )
    if ancestry.returncode == 1:
        raise MigrationError(
            f"source transition is not a fast-forward: {old_commit} -> {new_commit}"
        )
    if ancestry.returncode != 0:
        detail = (ancestry.stderr or ancestry.stdout or "unknown error").strip()
        raise MigrationError(f"could not verify fast-forward ancestry: {detail}")


def _read_source_pin(path: Path) -> str:
    path = _require_regular_file(path, label="source commit pin")
    try:
        raw = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise MigrationError("source commit pin is not ASCII") from exc
    lines = raw.splitlines()
    if len(lines) != 1 or raw not in {lines[0], f"{lines[0]}\n"}:
        raise MigrationError("source commit pin must contain exactly one SHA and optional newline")
    return _require_commit(lines[0], label="stored source commit pin")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MigrationError(f"autopilot state contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_state(path: Path) -> tuple[dict[str, Any], str]:
    path = _require_regular_file(path, label="autopilot state")
    raw = path.read_bytes()
    try:
        state = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"autopilot state is not valid JSON: {exc}") from exc
    if not isinstance(state, dict):
        raise MigrationError("autopilot state must be a JSON object")
    return state, hashlib.sha256(raw).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MigrationError(f"{label} must be a positive integer")
    return value


def _state_position(state: Mapping[str, Any]) -> dict[str, Any]:
    completed = state.get("completed_cycles")
    if not isinstance(completed, list):
        raise MigrationError("autopilot state completed_cycles must be a list")
    last_cycle = 0
    for index, entry in enumerate(completed):
        if not isinstance(entry, dict):
            raise MigrationError(f"completed_cycles[{index}] must be an object")
        cycle = _positive_int(entry.get("cycle"), label=f"completed_cycles[{index}].cycle")
        if cycle <= last_cycle:
            raise MigrationError("completed cycle numbers must be strictly increasing")
        if entry.get("status") != "completed":
            raise MigrationError(f"completed cycle {cycle} does not have completed status")
        last_cycle = cycle

    next_cycle = _positive_int(state.get("next_cycle"), label="next_cycle")
    if last_cycle and next_cycle != last_cycle + 1:
        raise MigrationError(
            f"next_cycle {next_cycle} does not follow durable completed cycle {last_cycle}"
        )
    if not last_cycle and next_cycle != 1:
        raise MigrationError("next_cycle must be 1 when no durable cycle is completed")

    status = state.get("status")
    if not isinstance(status, str) or not status:
        raise MigrationError("autopilot state status must be a non-empty string")
    current = state.get("current_cycle")
    if not isinstance(current, dict):
        raise MigrationError("autopilot state current_cycle must be an object")
    current_cycle = _positive_int(current.get("cycle"), label="current_cycle.cycle")
    current_status = current.get("status")
    if not isinstance(current_status, str) or not current_status:
        raise MigrationError("current_cycle.status must be a non-empty string")
    expected_current = last_cycle if current_status == "completed" else next_cycle
    if current_cycle != expected_current:
        raise MigrationError(
            f"current cycle {current_cycle} is inconsistent with status {current_status}"
        )
    return {
        "last_durable_completed_cycle": last_cycle if last_cycle else None,
        "next_cycle": next_cycle,
        "status": status,
        "current_cycle": {"cycle": current_cycle, "status": current_status},
    }


def _verified_state_artifact(
    state: Mapping[str, Any],
    *,
    out_root: Path,
    path_key: str,
    sha_key: str,
    label: str,
) -> dict[str, Any]:
    raw_path = state.get(path_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise MigrationError(f"stored {label} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        raise MigrationError(f"stored {label} path must be absolute: {path}")
    path = _require_regular_file(path, label=label)
    try:
        path.relative_to(out_root)
    except ValueError as exc:
        raise MigrationError(f"stored {label} is outside Vast output root: {path}") from exc
    expected = state.get(sha_key)
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise MigrationError(f"stored {label} SHA-256 is missing or malformed")
    actual = _sha256_file(path)
    if actual != expected:
        raise MigrationError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return {
        "path": str(path),
        "sha256": actual,
        "stored_sha256": expected,
        "verified": True,
    }


def _snapshot(
    *,
    repo_root: Path,
    out_root: Path,
    lock_path: Path,
    state_path: Path,
    old_commit: str,
    new_commit: str,
) -> dict[str, Any]:
    state, state_sha256 = _load_state(state_path)
    position = _state_position(state)
    checkpoint = _verified_state_artifact(
        state,
        out_root=out_root,
        path_key="training_checkpoint_path",
        sha_key="training_checkpoint_sha256",
        label="training checkpoint",
    )
    active_model = _verified_state_artifact(
        state,
        out_root=out_root,
        path_key="active_model_path",
        sha_key="active_model_sha256",
        label="active model",
    )
    return {
        "schema": AUDIT_SCHEMA,
        "phase": AUDIT_PHASE,
        "repo_root": str(repo_root),
        "out_root": str(out_root),
        "source_commit": {
            "old": old_commit,
            "new": new_commit,
            "fast_forward_verified": True,
        },
        "lock": {
            "path": str(lock_path),
            "mode": "exclusive-nonblocking",
            "acquired": True,
        },
        "state": {
            "path": str(state_path),
            "sha256": state_sha256,
            **position,
        },
        "training_checkpoint": checkpoint,
        "active_model": active_model,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    audit_directory_created = False
    try:
        path.parent.mkdir(mode=0o700)
        audit_directory_created = True
    except FileExistsError:
        pass
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MigrationError(f"audit directory must be a real directory: {path.parent}")
    if audit_directory_created:
        _fsync_directory(path.parent.parent)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_raw)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(encoded)
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


def _atomic_replace_pin(path: Path, commit: str) -> None:
    metadata = path.stat()
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_raw)
    try:
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(f"{commit}\n".encode("ascii"))
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


def _load_audit(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path, label="prepared audit")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"prepared audit is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MigrationError("prepared audit must be a JSON object")
    timestamp = payload.get("prepared_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise MigrationError("prepared audit has an invalid UTC timestamp")
    try:
        datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise MigrationError("prepared audit has an invalid UTC timestamp") from exc
    if payload.get("schema") != AUDIT_SCHEMA or payload.get("phase") != AUDIT_PHASE:
        raise MigrationError("prepared audit has an unsupported schema or phase")
    return payload


def _without_timestamp(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("prepared_at_utc", None)
    return result


def _validate_recovery_audit(
    audit: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    require_current_snapshot: bool,
) -> None:
    if require_current_snapshot:
        if _without_timestamp(audit) != dict(snapshot):
            raise MigrationError(
                "prepared audit no longer matches current state/artifacts; refusing pin update"
            )
        return
    for key in ("schema", "phase", "repo_root", "out_root", "source_commit", "lock"):
        if audit.get(key) != snapshot.get(key):
            raise MigrationError(f"prepared audit does not match the requested transition: {key}")
    for section in ("state", "training_checkpoint", "active_model"):
        value = audit.get(section)
        if not isinstance(value, dict):
            raise MigrationError(f"prepared audit is missing {section} evidence")
        digest = value.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise MigrationError(f"prepared audit has invalid {section} SHA-256 evidence")
        if not isinstance(value.get("path"), str) or not value.get("path"):
            raise MigrationError(f"prepared audit has invalid {section} path evidence")
        if section == "state" and value.get("path") != snapshot["state"]["path"]:
            raise MigrationError("prepared audit references a different autopilot state path")
        if section != "state" and (
            value.get("verified") is not True or value.get("stored_sha256") != digest
        ):
            raise MigrationError(f"prepared audit has unverified {section} evidence")


@contextmanager
def _existing_nonblocking_lock(lock_path: Path) -> Iterator[TextIO]:
    if fcntl is None:
        raise MigrationError("fcntl locking is required for Vast source migration")
    lock_path = _require_regular_file(lock_path, label="autopilot lock")
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags)
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise MigrationError(
                    "trainer appears to be running; autopilot.lock is held"
                ) from exc
            raise MigrationError(f"could not acquire autopilot lock: {exc}") from exc
        yield handle
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def migrate_source_commit(
    *,
    repo_root: Path,
    out_root: Path,
    expected_old_commit: str,
    expected_new_commit: str,
    before_pin_replace: Optional[Callable[[Path, Path], None]] = None,
) -> MigrationResult:
    """Prepare an immutable audit, then atomically advance the source pin."""
    old_commit = _require_commit(expected_old_commit, label="expected old commit")
    new_commit = _require_commit(expected_new_commit, label="expected new commit")
    if old_commit == new_commit:
        raise MigrationError("old and new source commits must differ")
    repo_root = _require_directory(repo_root, label="repository root")
    out_root = _require_directory(out_root, label="Vast output root")
    lock_path = out_root / "autopilot.lock"
    state_path = out_root / "autopilot_state.json"
    pin_path = out_root / "source_git_commit"
    audit_path = audit_path_for(out_root, old_commit, new_commit)

    with _existing_nonblocking_lock(lock_path):
        _validate_repository(repo_root, old_commit=old_commit, new_commit=new_commit)
        stored_pin = _read_source_pin(pin_path)
        if stored_pin not in {old_commit, new_commit}:
            raise MigrationError(
                f"stored source pin {stored_pin} matches neither expected old nor new commit"
            )
        snapshot = _snapshot(
            repo_root=repo_root,
            out_root=out_root,
            lock_path=lock_path,
            state_path=state_path,
            old_commit=old_commit,
            new_commit=new_commit,
        )

        if audit_path.exists() or audit_path.is_symlink():
            audit = _load_audit(audit_path)
            _validate_recovery_audit(
                audit,
                snapshot,
                require_current_snapshot=stored_pin == old_commit,
            )
        elif stored_pin == new_commit:
            raise MigrationError(
                "source pin already names the new commit but no prepared audit exists"
            )
        else:
            audit = {**snapshot, "prepared_at_utc": _utc_now()}
            _atomic_write_json(audit_path, audit)
            audit = _load_audit(audit_path)
            _validate_recovery_audit(audit, snapshot, require_current_snapshot=True)

        if stored_pin == new_commit:
            return MigrationResult(
                status="already-applied",
                audit_path=audit_path,
                old_commit=old_commit,
                new_commit=new_commit,
            )

        if before_pin_replace is not None:
            before_pin_replace(audit_path, pin_path)
        _atomic_replace_pin(pin_path, new_commit)
        if _read_source_pin(pin_path) != new_commit:
            raise MigrationError("source pin verification failed after atomic replacement")
        return MigrationResult(
            status="migrated",
            audit_path=audit_path,
            old_commit=old_commit,
            new_commit=new_commit,
        )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-old-commit", required=True)
    parser.add_argument("--expected-new-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        result = migrate_source_commit(
            repo_root=args.repo_root,
            out_root=args.out_root,
            expected_old_commit=args.expected_old_commit,
            expected_new_commit=args.expected_new_commit,
        )
    except MigrationError as exc:
        print(f"source commit migration refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "old_commit": result.old_commit,
                "new_commit": result.new_commit,
                "audit_path": str(result.audit_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
