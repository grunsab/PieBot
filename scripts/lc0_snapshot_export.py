#!/usr/bin/env python3
"""Read-only, bounded capture of one LC0 state's model recovery artifacts.

This standalone module uses no trainer lock and never modifies campaign files.
Captured descriptors pin immutable artifacts across retention unlink operations.
Missing or changed artifacts fail the entire attempt; callers may retry using a
new state. Corpus payloads are excluded, so strict campaign resume is not claimed.
"""

from __future__ import annotations

import hashlib
import argparse
import io
import json
import os
import re
import stat
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable


SCHEMA = "piebot-lc0-model-snapshot-v1"
STATE_SCHEMA = "piebot-lc0-autopilot-v1"
BLOCK_BYTES = 1024 * 1024
METADATA_LIMIT = 64 * 1024 * 1024
STATE_LIMIT = 32 * 1024 * 1024
HEADER_LIMIT = 8 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class SnapshotError(RuntimeError):
    """The requested state cannot currently be captured coherently."""


def _checksum(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise SnapshotError(f"invalid {label} SHA-256")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError) as exc:
        raise SnapshotError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} must be a JSON object")
    return value


def _root(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise SnapshotError(f"{label} must be an absolute, real directory")
    return path.resolve()


def _open_descriptor(path: Path, root: Path) -> int:
    directory = None
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = path.relative_to(root).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        if directory is not None:
            os.close(directory)


@dataclass
class PinnedFile:
    path: Path
    handle: BinaryIO
    initial_stat: os.stat_result
    expected_sha256: str | None
    roles: list[str] = field(default_factory=list)
    sha256: str | None = None

    def unchanged(self) -> None:
        current = os.fstat(self.handle.fileno())
        # Unlink changes ctime/link count; it does not change immutable bytes.
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(current, key) != getattr(self.initial_stat, key) for key in keys):
            raise SnapshotError(f"pinned file changed: {self.path}")


class CapturePlan:
    def __init__(self, campaign_root: Path, bootstrap_root: Path, max_bytes: int, max_files: int):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise SnapshotError("max_bytes must be a positive integer")
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            raise SnapshotError("max_files must be a positive integer")
        self.campaign_root = _root(campaign_root, "campaign root")
        self.bootstrap_root = _root(bootstrap_root, "bootstrap root")
        if (self.campaign_root.is_relative_to(self.bootstrap_root)
                or self.bootstrap_root.is_relative_to(self.campaign_root)):
            raise SnapshotError("campaign and bootstrap roots must be separate")
        self.max_bytes, self.max_files = max_bytes, max_files
        self.files: dict[str, PinnedFile] = {}
        self.manifest: dict = {}
        self.state_bytes = b""
        self.total_bytes = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for file in self.files.values():
            file.handle.close()
        self.closed = True

    def _path(self, raw: object, approved_root: Path | None = None) -> tuple[Path, Path]:
        if not isinstance(raw, (str, Path)):
            raise SnapshotError("artifact path must be an absolute path")
        path = Path(raw)
        if not path.is_absolute() or ".." in path.parts:
            raise SnapshotError(f"artifact path is not absolute/canonical: {path}")
        roots = (approved_root,) if approved_root else (self.campaign_root, self.bootstrap_root)
        for root in roots:
            if path.is_relative_to(root) and path != root:
                return path, root
        raise SnapshotError(f"artifact escapes approved roots: {path}")

    def add(self, raw: object, role: str, expected: str | None = None,
            approved_root: Path | None = None) -> PinnedFile:
        path, root = self._path(raw, approved_root)
        if expected is not None:
            expected = _checksum(expected, role)
        key = str(path)
        if key in self.files:
            file = self.files[key]
            if expected and file.expected_sha256 not in (None, expected):
                raise SnapshotError(f"conflicting file checksum commitments: {path}")
            file.expected_sha256 = expected or file.expected_sha256
            if role not in file.roles:
                file.roles.append(role)
            return file
        if len(self.files) >= self.max_files:
            raise SnapshotError("snapshot file cap exceeded")
        descriptor = None
        try:
            descriptor = _open_descriptor(path, root)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise SnapshotError(f"artifact must be a regular file: {path}")
            if self.total_bytes + metadata.st_size > self.max_bytes:
                raise SnapshotError("snapshot byte cap exceeded")
            handle = os.fdopen(descriptor, "rb")
            descriptor = None
            file = PinnedFile(path, handle, metadata, expected, [role])
            self.files[key] = file
            self.total_bytes += metadata.st_size
            return file
        except OSError as exc:
            raise SnapshotError(f"cannot pin artifact {path}: {exc.strerror}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def metadata(self, file: PinnedFile) -> bytes:
        if file.initial_stat.st_size > METADATA_LIMIT:
            raise SnapshotError(f"metadata size limit exceeded: {file.path}")
        file.unchanged()
        file.handle.seek(0)
        raw = file.handle.read(METADATA_LIMIT + 1)
        file.unchanged()
        if len(raw) != file.initial_stat.st_size:
            raise SnapshotError(f"metadata read length changed: {file.path}")
        digest = hashlib.sha256(raw).hexdigest()
        if file.expected_sha256 and file.expected_sha256 != digest:
            raise SnapshotError(f"metadata SHA-256 mismatch: {file.path}")
        file.sha256 = digest
        return raw

    def copy_file(self, source_path: str, output: BinaryIO | None) -> str:
        if self.closed:
            raise SnapshotError("capture plan is closed")
        file = self.files[source_path]
        file.unchanged()
        file.handle.seek(0)
        digest = hashlib.sha256()
        copied = 0
        while block := file.handle.read(BLOCK_BYTES):
            copied += len(block)
            if copied > file.initial_stat.st_size:
                raise SnapshotError(f"file grew during capture: {file.path}")
            digest.update(block)
            if output is not None:
                output.write(block)
        file.unchanged()
        actual = digest.hexdigest()
        if copied != file.initial_stat.st_size or actual != (file.expected_sha256 or file.sha256 or actual):
            raise SnapshotError(f"artifact SHA-256/size mismatch: {file.path}")
        file.sha256 = actual
        return actual

    def verify(self):
        for path in self.files:
            self.copy_file(path, None)


def capture_plan(*, campaign_root: Path, bootstrap_root: Path, max_bytes: int,
                 max_files: int = 128, after_state_read: Callable | None = None,
                 after_open: Callable | None = None) -> CapturePlan:
    """Pin and verify one state's committed model/Adam recovery references.

    Hooks support deterministic race tests. A failed attempt closes all retained
    descriptors. The caller must close successful plans, preferably using with.
    """
    plan = CapturePlan(campaign_root, bootstrap_root, max_bytes, max_files)
    campaign, bootstrap = plan.campaign_root, plan.bootstrap_root
    try:
        state_file = plan.add(campaign / "training/lc0_state.json", "state")
        if state_file.initial_stat.st_size > STATE_LIMIT:
            raise SnapshotError("LC0 state exceeds 32 MiB cap")
        plan.state_bytes = plan.metadata(state_file)
        state = _json(plan.state_bytes, "LC0 state")
        if state.get("schema") != STATE_SCHEMA:
            raise SnapshotError("unsupported LC0 state schema")
        if after_state_read:
            after_state_read(plan.state_bytes)
        identity = state["identity"]
        initial, incumbent = identity["initial_checkpoint"], identity["initial_active_model"]
        plan.add(initial["path"], "initial_checkpoint", initial["sha256"], bootstrap)
        plan.add(incumbent["path"], "initial_quant", incumbent["sha256"], bootstrap)
        history = state["history"]
        if not isinstance(history, list) or state.get("completed_chunks") != len(history):
            raise SnapshotError("LC0 history/completed count mismatch")
        records: dict[str, tuple[dict, list[str]]] = {}

        def add_chunk(record, role):
            directory, _ = plan._path(record["directory"], campaign / "training/chunks")
            if directory.name != f"chunk_{record['number']:08d}":
                raise SnapshotError("unexpected committed chunk directory")
            records.setdefault(str(directory), (record, []))[1].append(role)
            plan.add(directory / "train/checkpoint.json", role + "_checkpoint", record["checkpoint_sha256"], campaign)
            plan.add(directory / "train/optimizer.pt", role + "_optimizer", record["optimizer_sha256"], campaign)
            plan.add(directory / "candidate.nnue", role + "_quant", record["quant_sha256"], campaign)

        if history:
            latest = history[-1]
            directory = Path(latest["directory"])
            if (state["training_checkpoint_path"] != str(directory / "train/checkpoint.json")
                    or state["training_optimizer_path"] != str(directory / "train/optimizer.pt")):
                raise SnapshotError("latest learner paths do not match state history")
            add_chunk(latest, "latest")
        elif state.get("training_checkpoint_path") or state.get("training_optimizer_path"):
            raise SnapshotError("learner artifacts lack committed history")
        if state.get("best_checkpoint_path"):
            if state["best_checkpoint_path"] == initial["path"]:
                if state["best_checkpoint_sha256"] != initial["sha256"]:
                    raise SnapshotError("baseline best checkpoint checksum mismatch")
                plan.add(initial["path"], "best_checkpoint", initial["sha256"], bootstrap)
                plan.add(state["best_quant_path"], "best_quant", state["best_quant_sha256"])
            else:
                best = next((record for record in reversed(history)
                             if str(Path(record["directory"]) / "train/checkpoint.json") == state["best_checkpoint_path"]), None)
                if (best is None or best["checkpoint_sha256"] != state["best_checkpoint_sha256"]
                        or best["quant_sha256"] != state["best_quant_sha256"]
                        or str(Path(best["directory"]) / "candidate.nnue") != state["best_quant_path"]):
                    raise SnapshotError("best learner does not match a committed history record")
                add_chunk(best, "best")
        plan.add(state["active_model_path"], "active_quant", state["active_model_sha256"])
        # All retention-sensitive model files are now open. Metadata reads and
        # payload hashing happen afterwards, without consulting a newer state.
        pin = plan.add(campaign / "source_git_commit", "source_pin")
        corpus_file = plan.add(campaign / "data/corpus/corpus_manifest.json", "corpus_manifest",
                               identity["manifest_sha256"], campaign)
        corpus = _json(plan.metadata(corpus_file), "corpus manifest")
        if (corpus.get("schema") != "piebot-lc0-corpus-v1" or corpus.get("complete") is not True
                or corpus.get("corpus_id") != identity["corpus_id"]):
            raise SnapshotError("corpus identity does not match state")
        raw = plan.add(campaign / "data/raw/manifest.json", "raw_manifest", corpus["raw_manifest_sha256"], campaign)
        corpus_identity = plan.add(campaign / "data/corpus/identity.json", "corpus_identity", approved_root=campaign)
        validation = corpus["validation"]
        if validation["sha256"] != identity["validation_sha256"]:
            raise SnapshotError("fixed validation checksum does not match state")
        plan.add(validation["path"], "fixed_validation", validation["sha256"], campaign)
        for directory, (record, roles) in records.items():
            for role in roles:
                plan.add(Path(directory) / "complete.json", role + "_completion", approved_root=campaign)
                plan.add(Path(directory) / "train/metrics.json", role + "_metrics", approved_root=campaign)
        if after_open:
            after_open(plan)
        source_commit = identity["source"]["commit"]
        if not isinstance(source_commit, str) or COMMIT.fullmatch(source_commit) is None:
            raise SnapshotError("invalid source commit")
        if plan.metadata(pin) not in (source_commit.encode(), (source_commit + "\n").encode()):
            raise SnapshotError("source pin does not match captured state")
        _json(plan.metadata(raw), "raw manifest")
        if _json(plan.metadata(corpus_identity), "corpus identity").get("corpus_id") != identity["corpus_id"]:
            raise SnapshotError("corpus identity file does not match state")
        for directory, (record, _) in records.items():
            complete = _json(plan.metadata(plan.files[str(Path(directory) / "complete.json")]), "completion metadata")
            metrics = _json(plan.metadata(plan.files[str(Path(directory) / "train/metrics.json")]), "training metrics")
            if complete != record:
                raise SnapshotError("completion metadata does not match captured history")
            if metrics.get("optimizer_state", {}).get("sha256") != record["optimizer_sha256"]:
                raise SnapshotError("optimizer metrics binding does not match captured history")
        plan.verify()
        blobs = {}
        roles = set()
        for file in plan.files.values():
            entry = blobs.setdefault(file.sha256, {"sha256": file.sha256, "size": file.initial_stat.st_size,
                "member": "blobs/" + file.sha256, "sources": []})
            if entry["size"] != file.initial_stat.st_size:
                raise SnapshotError("identical checksum has conflicting file size")
            for role in file.roles:
                if role in roles:
                    raise SnapshotError(f"duplicate snapshot role: {role}")
                roles.add(role)
                entry["sources"].append({"role": role, "path": str(file.path)})
        plan.manifest = {"schema": SCHEMA,
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "capabilities": {"model_recovery": True, "strict_campaign_resume": False,
                             "corpus_payloads_included": False},
            "state_sha256": hashlib.sha256(plan.state_bytes).hexdigest(),
            "completed_chunks": state["completed_chunks"], "deadline_at": state.get("deadline_at"),
            "source": identity["source"], "campaign_root": str(campaign), "bootstrap_root": str(bootstrap),
            "files": [blobs[key] for key in sorted(blobs)]}
        summary = _state_summary(plan.state_bytes)
        plan.manifest.update(status=summary["status"], model_identity_key=summary["model_identity_key"])
        return plan
    except (KeyError, TypeError, ValueError, OSError) as exc:
        plan.close()
        raise SnapshotError(f"invalid or unavailable capture reference: {exc}") from exc
    except BaseException:
        plan.close()
        raise


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class _VerifiedReader:
    def __init__(self, file: PinnedFile):
        self.file = file
        self.file.unchanged()
        self.file.handle.seek(0)
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int) -> bytes:
        block = self.file.handle.read(size)
        self.count += len(block)
        self.digest.update(block)
        return block

    def finish(self):
        self.file.unchanged()
        if (self.count != self.file.initial_stat.st_size or self.file.handle.read(1)
                or self.digest.hexdigest() != self.file.sha256):
            raise SnapshotError(f"artifact changed while streaming: {self.file.path}")


def write_tar(plan: CapturePlan, output: BinaryIO, *, before_complete: Callable | None = None) -> None:
    """Stream plain USTAR; success requires a completion marker and clean exit.

    A receiver must discard an incomplete stream, and must not commit merely
    because it saw the marker: exporter/transport success is also required.
    """
    if plan.closed:
        raise SnapshotError("capture plan is closed")
    header = _encoded(plan.manifest)
    if len(header) > HEADER_LIMIT:
        raise SnapshotError("snapshot header exceeds 8 MiB cap")
    def add_bytes(archive, name, payload):
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(payload), 0o600, 0
        archive.addfile(info, io.BytesIO(payload))
    try:
        with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            add_bytes(archive, "snapshot.json", header)
            streamed = set()
            for entry in plan.manifest["files"]:
                path = entry["sources"][0]["path"]
                file = plan.files[path]
                info = tarfile.TarInfo(entry["member"])
                info.size, info.mode, info.mtime = entry["size"], 0o600, 0
                reader = _VerifiedReader(file)
                archive.addfile(info, reader)
                reader.finish()
                streamed.add(path)
            if before_complete:
                before_complete(plan)
            # Aliases share a digest but can have independent retained inodes.
            for path, file in plan.files.items():
                if path not in streamed:
                    plan.copy_file(path, None)
                file.unchanged()
            add_bytes(archive, "complete.json", _encoded({
                "schema": "piebot-lc0-model-snapshot-complete-v1",
                "snapshot_sha256": hashlib.sha256(header).hexdigest(),
                "blob_count": len(plan.manifest["files"]),
                "total_blob_bytes": sum(entry["size"] for entry in plan.manifest["files"])}))
    except (OSError, tarfile.TarError) as exc:
        raise SnapshotError(f"snapshot stream failed: {exc}") from exc


export_tar = write_tar


def probe(campaign_root: Path) -> dict:
    """Read only bounded state metadata; discovery does not verify payloads."""
    campaign = _root(campaign_root, "campaign root")
    path = campaign / "training/lc0_state.json"
    try:
        descriptor = _open_descriptor(path, campaign)
    except FileNotFoundError:
        return {"ready": False, "reason": "state-not-created", "status": "preparing",
                "completed_chunks": 0, "state_sha256": None, "model_identity_key": None,
                "hashes_verified": False}
    except OSError as exc:
        raise SnapshotError(f"cannot read LC0 state: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        file = PinnedFile(path, handle, os.fstat(handle.fileno()), None)
        if not stat.S_ISREG(file.initial_stat.st_mode) or file.initial_stat.st_size > STATE_LIMIT:
            raise SnapshotError("LC0 probe state must be a regular file below 32 MiB")
        raw = handle.read(STATE_LIMIT + 1)
        file.unchanged()
    return _state_summary(raw)


def _state_summary(raw: bytes) -> dict:
    state = _json(raw, "LC0 state")
    if state.get("schema") != STATE_SCHEMA:
        raise SnapshotError("unsupported LC0 state schema")
    count = state.get("completed_chunks")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SnapshotError("invalid LC0 completed count")
    result = {"ready": count > 0, "reason": "ready" if count else "awaiting-first-completed-chunk",
              "status": state.get("status"), "completed_chunks": count,
              "state_sha256": hashlib.sha256(raw).hexdigest(), "model_identity_key": None,
              "deadline_at": state.get("deadline_at"), "hashes_verified": False}
    if count:
        try:
            if len(state["history"]) != count:
                raise SnapshotError("LC0 history/completed count mismatch")
            latest = state["history"][-1]
            result["latest"] = {key: _checksum(latest[key], key) for key in
                                ("checkpoint_sha256", "optimizer_sha256", "quant_sha256")}
            result["best"] = ({"checkpoint_sha256": _checksum(state["best_checkpoint_sha256"], "best checkpoint"),
                               "quant_sha256": _checksum(state["best_quant_sha256"], "best quant")}
                              if state.get("best_checkpoint_path") else None)
            result["active"] = {"sha256": _checksum(state["active_model_sha256"], "active quant")}
            key = {"identity": state["identity"], "latest": result["latest"],
                   "best": result["best"], "active": result["active"]}
            result["model_identity_key"] = hashlib.sha256(_encoded(key)).hexdigest()
        except (KeyError, TypeError) as exc:
            raise SnapshotError("probe state lacks committed model identities") from exc
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--bootstrap-root", type=Path)
    parser.add_argument("--max-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--max-files", type=int, default=128)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if not args.probe and args.bootstrap_root is None:
        parser.error("capture requires --bootstrap-root")
    try:
        if args.probe:
            print(json.dumps(probe(args.campaign_root), sort_keys=True))
        else:
            with capture_plan(campaign_root=args.campaign_root, bootstrap_root=args.bootstrap_root,
                              max_bytes=args.max_bytes, max_files=args.max_files) as plan:
                write_tar(plan, sys.stdout.buffer)
                sys.stdout.buffer.flush()
        return 0
    except (SnapshotError, OSError) as exc:
        print(f"LC0 snapshot refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
