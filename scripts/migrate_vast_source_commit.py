#!/usr/bin/env python3
"""Audit and advance the source pin for a stopped restart-safe Vast.ai run.

The utility intentionally does not stop the trainer or control supervisor.
Self-play uses its existing autopilot lock. The opt-in LC0-pretraining mode
requires external supervisor-stop verification and the real launcher lock,
and refuses once any training state or learner output exists.
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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - the production Vast host is Linux.
    fcntl = None  # type: ignore[assignment]


AUDIT_SCHEMA = "piebot-source-commit-migration-v1"
LC0_AUDIT_SCHEMA = "piebot-lc0-pretraining-source-commit-migration-v1"
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
    completed_cycles = state.get("completed_cycles") or []
    raw_checkpoint = state.get("training_checkpoint_path")
    if raw_checkpoint in (None, "") and not completed_cycles:
        # A fresh lineage before its first completed cycle legitimately has
        # no learner checkpoint yet: there is nothing to preserve or verify.
        # A mature state missing its checkpoint is still a hard refusal.
        checkpoint: dict[str, Any] = {
            "path": None,
            "sha256": None,
            "stored_sha256": None,
            "verified": True,
            "fresh_lineage": True,
        }
    else:
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


def _load_audit(path: Path, *, expected_schema: str = AUDIT_SCHEMA) -> dict[str, Any]:
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
    if payload.get("schema") != expected_schema or payload.get("phase") != AUDIT_PHASE:
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
        if section == "training_checkpoint" and value.get("fresh_lineage") is True:
            if (
                value.get("verified") is not True
                or value.get("path") is not None
                or value.get("sha256") is not None
            ):
                raise MigrationError(
                    "prepared audit has invalid fresh-lineage checkpoint evidence"
                )
            continue
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


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _within(path: Path, root: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise MigrationError(f"{label} path must be absolute: {path}")
    if ".." in path.parts:
        raise MigrationError(f"{label} path must not contain parent traversal: {path}")
    # Approved roots are canonical. Permit a system alias before that root
    # (e.g. macOS /var), but reject every alias at or below the root itself.
    anchor = next((parent for parent in reversed((path, *path.parents))
                   if parent.resolve() == root), None)
    if anchor is None:
        raise MigrationError(f"{label} is outside its approved root: {path}")
    component = anchor
    for part in (None, *path.relative_to(anchor).parts):
        if part is not None:
            component /= part
        if component.is_symlink():
            raise MigrationError(f"{label} must not contain a symlink: {component}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise MigrationError(f"{label} is outside its approved root: {path}")
    return resolved


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MigrationError(f"{label} must be a lowercase SHA-256")
    return value


def _file_commitment(path: Path, *, label: str) -> dict[str, Any]:
    path = _require_regular_file(path, label=label)
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _json_commitment(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _require_regular_file(path, label=label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"{label} must be a JSON object")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _require_lc0_pretraining(out_root: Path) -> None:
    for relative in ("training/lc0_state.json", "lc0_state.json", "autopilot_state.json"):
        if _present(out_root / relative):
            raise MigrationError(f"training or self-play state exists; pretraining migration refused: {relative}")
    # Even a zero-chunk state owns an identity and deadline. Refuse orphaned
    # output too; never infer that deleting state made a lineage fresh again.
    for name in ("chunks", "cycles", "train", "checkpoint.json", "optimizer.pt", "accepted", "baseline"):
        if _present(out_root / name):
            raise MigrationError(f"learner artifact exists; pretraining migration refused: {name}")
    training = out_root / "training"
    if _present(training):
        training = _require_directory(training, label="LC0 training directory")
        if any(path.name != "lc0.lock" for path in training.iterdir()):
            raise MigrationError("training artifacts exist; only an unused lc0.lock may precede training")


def _raw_snapshot(out_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_root = _within(out_root / "data/raw", out_root, label="LC0 raw data root")
    raw_root = _require_directory(raw_root, label="LC0 raw data root")
    raw, commitment = _json_commitment(raw_root / "manifest.json", label="frozen raw manifest")
    if raw.get("schema") != "piebot-lc0-raw-v1" or not isinstance(raw.get("files"), list) or not raw["files"]:
        raise MigrationError("raw manifest must be a non-empty frozen LC0 snapshot")
    bounds = []
    for key in ("since", "until"):
        try:
            bound = datetime.fromisoformat(raw[key].replace("Z", "+00:00"))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise MigrationError(f"invalid frozen raw manifest {key}") from exc
        if bound.tzinfo is None or bound.utcoffset() is None:
            raise MigrationError(f"frozen raw manifest {key} must include a UTC offset")
        bounds.append(bound)
    if bounds[0] >= bounds[1]:
        raise MigrationError("invalid frozen raw manifest date bounds")
    urls, destinations = set(), set()
    for entry in raw["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str) or not entry["url"]:
            raise MigrationError("invalid raw archive URL commitment")
        _positive_int(entry.get("size"), label="raw archive size")
        if not isinstance(entry.get("dest"), str):
            raise MigrationError("raw archive destination is missing")
        dest = _within(Path(entry["dest"]), raw_root, label="raw archive")
        if entry["url"] in urls or dest in destinations:
            raise MigrationError("duplicate raw archive URL or destination")
        urls.add(entry["url"])
        destinations.add(dest)
        if entry.get("sha256") is not None:
            _require_sha256(entry["sha256"], label="raw archive checksum commitment")
        elif entry.get("status") in {"downloaded", "verified"}:
            raise MigrationError("completed raw archive lacks its checksum commitment")
    commitment.update(schema=raw["schema"], since=raw["since"], until=raw["until"],
                      archive_count=len(raw["files"]), payloads_rehashed=False)
    return raw, commitment


def _corpus_snapshot(out_root: Path, raw: dict[str, Any], raw_sha: str) -> dict[str, Any] | None:
    corpus = _within(out_root / "data/corpus", out_root, label="LC0 corpus root")
    if not _present(corpus):
        return None
    corpus = _require_directory(corpus, label="LC0 corpus root")
    files: dict[str, Any] = {}
    result: dict[str, Any] = {"path": str(corpus), "files": files,
                              "chunk_commitments": [], "payloads_rehashed": False}
    identity = None
    identity_path = corpus / "identity.json"
    if _present(identity_path):
        identity, files["identity.json"] = _json_commitment(identity_path, label="corpus identity")
        config = {key: value for key, value in identity.items() if key != "corpus_id"}
        actual_id = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        expected_sources = [{key: row.get(key) for key in ("url", "sha256", "size", "suite")}
                            for row in raw["files"]]
        if (identity.get("schema") != "piebot-lc0-corpus-v1" or identity.get("corpus_id") != actual_id
                or identity.get("since") != raw["since"] or identity.get("until") != raw["until"]
                or identity.get("sources") != expected_sources):
            raise MigrationError("corpus identity does not match frozen raw manifest")
        result["corpus_id"] = actual_id
    for name in ("progress.sqlite3", "progress.sqlite3-journal", "progress.sqlite3-wal", "progress.sqlite3-shm"):
        if _present(corpus / name):
            files[name] = _file_commitment(corpus / name, label=f"corpus {name}")

    chunks: dict[str, dict[str, Any]] = {}
    def chunk_commitment(path: Path, entry: dict[str, Any]) -> None:
        path = _within(path, corpus, label="committed corpus chunk")
        path = _require_regular_file(path, label="committed corpus chunk")
        checksum = _require_sha256(entry.get("sha256"), label="corpus chunk checksum commitment")
        count = _positive_int(entry.get("positions"), label="corpus chunk positions")
        value = {"path": str(path), "sha256_commitment": checksum,
                 "positions": count, "bytes": path.stat().st_size}
        if str(path) in chunks and chunks[str(path)] != value:
            raise MigrationError("conflicting corpus chunk commitments")
        chunks[str(path)] = value

    archives = corpus / "archives"
    if _present(archives):
        _require_directory(archives, label="corpus archives directory")
        for directory in sorted(archives.iterdir()):
            _require_directory(directory, label="corpus archive directory")
            if directory.name.endswith(".part"):
                # This cache is outside a committed archive transaction.
                continue
            info, commitment = _json_commitment(directory / "archive.json", label="corpus archive metadata")
            files[str((directory / "archive.json").relative_to(corpus))] = commitment
            for key in ("chunks", "holdout"):
                if not isinstance(info.get(key), list):
                    raise MigrationError(f"invalid corpus archive {key}")
                for entry in info[key]:
                    name = entry.get("name") if isinstance(entry, dict) else None
                    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                        raise MigrationError("invalid committed corpus chunk name")
                    chunk_commitment(directory / name, entry)
    manifest_path = corpus / "corpus_manifest.json"
    if _present(manifest_path):
        manifest, files["corpus_manifest.json"] = _json_commitment(manifest_path, label="corpus manifest")
        if (identity is None or manifest.get("complete") is not True
                or manifest.get("schema") != "piebot-lc0-corpus-v1"
                or manifest.get("corpus_id") != identity["corpus_id"]
                or manifest.get("since") != identity["since"]
                or manifest.get("until") != identity["until"]
                or manifest.get("raw_manifest_sha256") != raw_sha):
            raise MigrationError("corpus manifest does not match frozen raw manifest/identity")
        if not isinstance(manifest.get("chunks"), list) or not manifest["chunks"]:
            raise MigrationError("complete corpus manifest has no chunks")
        for entry in manifest["chunks"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise MigrationError("invalid corpus manifest chunk")
            chunk_commitment(Path(entry["path"]), entry)
        validation = manifest.get("validation")
        if not isinstance(validation, dict) or not isinstance(validation.get("path"), str):
            raise MigrationError("invalid corpus validation commitment")
        validation_path = _within(Path(validation["path"]), corpus, label="fixed validation")
        files["validation.jsonl"] = _file_commitment(validation_path, label="fixed validation")
        if files["validation.jsonl"]["sha256"] != validation.get("sha256"):
            raise MigrationError("fixed validation SHA-256 mismatch")
    if identity is None and (files or chunks):
        raise MigrationError("corpus cache commitments exist without an identity")
    result["chunk_commitments"] = [chunks[key] for key in sorted(chunks)]
    return result


def migrate_lc0_pretraining_source_commit(
    *, repo_root: Path, out_root: Path, expected_old_commit: str, expected_new_commit: str,
    bootstrap_root: Path, bootstrap_checkpoint: Path, expected_bootstrap_sha256: str,
    active_model: Path, expected_active_sha256: str, supervisor_stop_verified: bool = False,
    before_pin_replace: Optional[Callable[[Path, Path], None]] = None,
) -> MigrationResult:
    """Audit a pin-only LC0 transition before a training identity/clock exists.

    The caller must externally verify supervisor is stopped and its descendants
    have exited. The explicit attestation is recorded, not checked by this tool.
    Raw/chunk payloads are not rehashed; their frozen metadata commitments are
    audited, while bootstrap, validation and corpus metadata bytes are hashed.
    """
    if supervisor_stop_verified is not True:
        raise MigrationError("LC0 migration requires an externally verified supervisor stop attestation")
    old = _require_commit(expected_old_commit, label="expected old commit")
    new = _require_commit(expected_new_commit, label="expected new commit")
    if old == new:
        raise MigrationError("old and new source commits must differ")
    repo_root = _require_directory(repo_root, label="repository root")
    out_root = _require_directory(out_root, label="LC0 output root")
    bootstrap_root = _require_directory(bootstrap_root, label="bootstrap root")
    if bootstrap_root.is_relative_to(out_root) or out_root.is_relative_to(bootstrap_root):
        raise MigrationError("bootstrap root must be separate from LC0 output root")
    bootstrap_checkpoint = _within(Path(bootstrap_checkpoint), bootstrap_root, label="bootstrap checkpoint")
    active_model = _within(Path(active_model), bootstrap_root, label="active model")
    expected_bootstrap_sha256 = _require_sha256(expected_bootstrap_sha256, label="bootstrap checkpoint checksum")
    expected_active_sha256 = _require_sha256(expected_active_sha256, label="active model checksum")
    pin = out_root / "source_git_commit"
    lock = out_root / "launcher.lock"
    audit_path = audit_path_for(out_root, old, new)
    with ExitStack() as stack:
        held_locks = {lock: stack.enter_context(_existing_nonblocking_lock(lock))}
        _require_lc0_pretraining(out_root)
        optional_locks = (out_root / "data/corpus/prepare.lock", out_root / "training/lc0.lock")
        for path in optional_locks:
            _within(path, out_root, label="optional LC0 lock")
            if _present(path):
                held_locks[path] = stack.enter_context(_existing_nonblocking_lock(path))
        acquired_optional_locks = {path for path in held_locks if path != lock}

        def snapshot() -> dict[str, Any]:
            _validate_repository(repo_root, old_commit=old, new_commit=new)
            _require_lc0_pretraining(out_root)
            if {path for path in optional_locks if _present(path)} != acquired_optional_locks:
                raise MigrationError("optional LC0 lock presence changed after acquisition")
            for path, handle in held_locks.items():
                _within(path, out_root, label="held LC0 lock")
                current_lock = _require_regular_file(path, label="held LC0 lock").stat()
                held_lock = os.fstat(handle.fileno())
                if (current_lock.st_dev, current_lock.st_ino) != (held_lock.st_dev, held_lock.st_ino):
                    raise MigrationError(f"held LC0 lock path changed: {path}")
            _within(bootstrap_checkpoint, bootstrap_root, label="bootstrap checkpoint")
            _within(active_model, bootstrap_root, label="active model")
            checkpoint = _file_commitment(bootstrap_checkpoint, label="bootstrap checkpoint")
            incumbent = _file_commitment(active_model, label="active model")
            if checkpoint["sha256"] != expected_bootstrap_sha256 or incumbent["sha256"] != expected_active_sha256:
                raise MigrationError("bootstrap checkpoint or incumbent SHA-256 mismatch")
            raw, raw_commitment = _raw_snapshot(out_root)
            delta = _git(repo_root, "diff", "--binary", "--full-index", old, new).stdout
            return {"schema": LC0_AUDIT_SCHEMA, "phase": AUDIT_PHASE, "mode": "lc0-pretraining",
                "repo_root": str(repo_root), "out_root": str(out_root),
                "source_commit": {"old": old, "new": new, "fast_forward_verified": True},
                "source_delta": {"sha256": hashlib.sha256(delta.encode()).hexdigest(),
                                 "name_status": _git(repo_root, "diff", "--name-status", old, new).stdout},
                "lock": {"path": str(lock), "mode": "exclusive-nonblocking", "acquired": True},
                "additional_locks": [str(path) for path in optional_locks if path in acquired_optional_locks],
                "supervisor_stop": {"externally_verified_by_caller": True, "verified_by_tool": False},
                "training": {"state_path": str(out_root / "training/lc0_state.json"),
                             "clock_started": False, "required_budget_hours": 720,
                             "live_budget_verified_by_tool": False,
                             "budget_basis": "required unchanged launch setting; supervisor configuration is external",
                             "state_created_or_modified": False},
                "bootstrap_root": str(bootstrap_root), "bootstrap_checkpoint": checkpoint,
                "active_model": incumbent, "raw_manifest": raw_commitment,
                "corpus": _corpus_snapshot(out_root, raw, raw_commitment["sha256"])}

        stored = _read_source_pin(pin)
        if stored not in {old, new}:
            raise MigrationError("stored source pin matches neither expected old nor new commit")
        current = snapshot()
        _within(audit_path, out_root, label="prepared LC0 audit")
        if _present(audit_path):
            audit = _load_audit(audit_path, expected_schema=LC0_AUDIT_SCHEMA)
            if _without_timestamp(audit) != current:
                raise MigrationError("prepared LC0 audit no longer matches current data/artifacts")
        elif stored == new:
            raise MigrationError("new LC0 source pin has no prepared audit")
        else:
            _atomic_write_json(audit_path, {**current, "prepared_at_utc": _utc_now()})
            audit = _load_audit(audit_path, expected_schema=LC0_AUDIT_SCHEMA)
            if _without_timestamp(audit) != current:
                raise MigrationError("prepared LC0 audit does not match the verified snapshot")
        if stored == new:
            return MigrationResult("already-applied", audit_path, old, new)
        if before_pin_replace is not None:
            before_pin_replace(audit_path, pin)
        if _read_source_pin(pin) != old or snapshot() != current:
            raise MigrationError("LC0 data/source pin changed before replacement")
        _within(audit_path, out_root, label="prepared LC0 audit")
        if _load_audit(audit_path, expected_schema=LC0_AUDIT_SCHEMA) != audit:
            raise MigrationError("prepared LC0 audit changed before replacement")
        _atomic_replace_pin(pin, new)
        if _read_source_pin(pin) != new:
            raise MigrationError("LC0 source pin verification failed after replacement")
        return MigrationResult("migrated", audit_path, old, new)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-old-commit", required=True)
    parser.add_argument("--expected-new-commit", required=True)
    parser.add_argument("--mode", choices=("selfplay", "lc0-pretraining"), default="selfplay")
    parser.add_argument("--bootstrap-root", type=Path)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-checkpoint-sha256")
    parser.add_argument("--active-model", type=Path)
    parser.add_argument("--active-model-sha256")
    parser.add_argument("--supervisor-stop-verified", action="store_true",
                        help="LC0 only: attest supervisor STOPPED and all descendants exited; externally verified, not tool-checked")
    args = parser.parse_args(argv)
    lc0_values = (args.bootstrap_root, args.bootstrap_checkpoint, args.bootstrap_checkpoint_sha256,
                  args.active_model, args.active_model_sha256)
    if args.mode == "lc0-pretraining" and (not all(lc0_values) or not args.supervisor_stop_verified):
        parser.error("lc0-pretraining requires bootstrap root/checkpoint/hash, active model/hash and --supervisor-stop-verified")
    if args.mode == "selfplay" and (any(lc0_values) or args.supervisor_stop_verified):
        parser.error("LC0 options require --mode lc0-pretraining")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        common = dict(repo_root=args.repo_root, out_root=args.out_root,
                      expected_old_commit=args.expected_old_commit, expected_new_commit=args.expected_new_commit)
        if args.mode == "lc0-pretraining":
            result = migrate_lc0_pretraining_source_commit(**common,
                bootstrap_root=args.bootstrap_root, bootstrap_checkpoint=args.bootstrap_checkpoint,
                expected_bootstrap_sha256=args.bootstrap_checkpoint_sha256,
                active_model=args.active_model, expected_active_sha256=args.active_model_sha256,
                supervisor_stop_verified=args.supervisor_stop_verified)
        else:
            result = migrate_source_commit(**common)
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
