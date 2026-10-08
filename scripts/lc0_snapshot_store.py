"""Receive verified LC0 model snapshots into a bounded local SHA256 store.

``catalog.json`` is the publication authority. A receipt or an unreferenced
blob alone is never a committed snapshot. Source paths are metadata only.
This module neither starts transports nor changes the source campaign.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import uuid
from typing import BinaryIO, Callable


GiB = 1024 ** 3
MAX_HEADER_BYTES = 8 * 1024 ** 2
MAX_COMPLETE_BYTES = 64 * 1024
MAX_FILES = 128
CHUNK_BYTES = 1024 ** 2
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
HEADER_SCHEMA = "piebot-lc0-model-snapshot-v1"
COMPLETE_SCHEMA = "piebot-lc0-model-snapshot-complete-v1"
POOL_SCHEMA = "piebot-lc0-snapshot-pool-v1"
CATALOG_SCHEMA = "piebot-lc0-snapshot-catalog-v1"
RECEIPT_SCHEMA = "piebot-lc0-snapshot-receipt-v1"
STAGING_SCHEMA = "piebot-lc0-snapshot-staging-v1"


class SnapshotError(RuntimeError):
    """The receive failed before publication, or the pool cannot be trusted."""


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise SnapshotError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def constant(value):
        raise SnapshotError(f"non-finite JSON value: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SnapshotError(f"invalid snapshot JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError("snapshot JSON must be an object")
    return value


def _uint(value) -> bool:
    return type(value) is int and value >= 0


def _read_small(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise SnapshotError(f"invalid or oversized pool metadata: {path.name}")
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise SnapshotError(f"pool metadata grew beyond its bound: {path.name}")
    return raw


def _manifest(raw: bytes, max_snapshot_bytes: int) -> dict:
    if not raw or len(raw) > MAX_HEADER_BYTES:
        raise SnapshotError("snapshot.json exceeds its size bound")
    header = _json(raw)
    if header.get("schema") != HEADER_SCHEMA:
        raise SnapshotError("unsupported snapshot schema")
    capabilities = header.get("capabilities")
    if not isinstance(capabilities, dict) or any(
        capabilities.get(key) is not value for key, value in {
            "model_recovery": True, "strict_campaign_resume": False,
            "corpus_payloads_included": False,
        }.items()
    ):
        raise SnapshotError("snapshot capabilities differ from model-recovery contract")
    if not _uint(header.get("completed_chunks")):
        raise SnapshotError("completed_chunks must be a nonnegative integer")
    state_sha = header.get("state_sha256")
    if not isinstance(state_sha, str) or not SHA_RE.fullmatch(state_sha):
        raise SnapshotError("invalid state SHA256")
    files = header.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise SnapshotError("snapshot must contain 1..128 unique blobs")
    roles = {}
    hashes = set()
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise SnapshotError("invalid snapshot file entry")
        sha = item.get("sha256")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha) or sha in hashes:
            raise SnapshotError("invalid or repeated blob SHA256")
        hashes.add(sha)
        if not _uint(item.get("size")) or item.get("member") != f"blobs/{sha}":
            raise SnapshotError("invalid blob member or size")
        total += item["size"]
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            raise SnapshotError("each blob needs source roles")
        for source in sources:
            if not isinstance(source, dict):
                raise SnapshotError("invalid source role")
            role, path = source.get("role"), source.get("path")
            if (not isinstance(role, str) or not role.strip() or role in roles
                    or not isinstance(path, str) or not path.startswith("/") or "\0" in path):
                raise SnapshotError("source roles must be unique and paths absolute metadata")
            roles[role] = sha
    if roles.get("state") != state_sha:
        raise SnapshotError("state role does not match state_sha256")
    if total > max_snapshot_bytes:
        raise SnapshotError("snapshot exceeds max_snapshot_bytes")
    return header


class _TarStream:
    """Strict plain USTAR reader; unlike tarfile stream iteration, require EOF."""
    def __init__(self, stream: BinaryIO, limit: int):
        self.stream, self.limit, self.bytes_read = stream, limit, 0

    def read(self, size: int, *, eof_ok=False) -> bytes:
        pieces = []
        left = size
        while left:
            part = self.stream.read(left)
            if not isinstance(part, (bytes, bytearray)) or len(part) > left:
                raise SnapshotError("transport returned an invalid read")
            if not part:
                if eof_ok and left == size:
                    return b""
                raise SnapshotError("truncated TAR stream")
            self.bytes_read += len(part)
            if self.bytes_read > self.limit:
                raise SnapshotError("TAR transport exceeds its declared size bound")
            pieces.append(part)
            left -= len(part)
        return b"".join(pieces)

    def member(self, name: str, size: int | None = None, *, maximum: int | None = None):
        block = self.read(512)
        if block[257:263] != b"ustar\0" or block[263:265] != b"00":
            raise SnapshotError("only plain USTAR members are accepted")
        try:
            info = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
        except (tarfile.TarError, ValueError, UnicodeError) as exc:
            raise SnapshotError(f"invalid TAR header: {exc}") from exc
        if (info.name != name or info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                or info.linkname or info.size < 0):
            raise SnapshotError(f"unexpected TAR member: {info.name!r}")
        if size is not None and info.size != size:
            raise SnapshotError(f"TAR size mismatch for {name}")
        if maximum is not None and info.size > maximum:
            raise SnapshotError(f"TAR member too large: {name}")
        return info.size

    def padding(self, size: int):
        padding = (-size) % 512
        if padding and any(self.read(padding)):
            raise SnapshotError("nonzero TAR member padding")

    def finish(self):
        if self.read(1024) != bytes(1024):
            raise SnapshotError("TAR is missing its two zero terminator blocks")
        # Python tarfile pads to a 10240-byte record. No concatenated archive,
        # arbitrary trailer or unbounded zero stream is accepted.
        remaining_padding = 10240 - 1024
        while True:
            block = self.read(512, eof_ok=True)
            if not block:
                return
            remaining_padding -= len(block)
            if remaining_padding < 0 or any(block):
                raise SnapshotError("unexpected data after TAR terminator")


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(out, raw: bytes):
    view = memoryview(raw)
    while view:
        count = out.write(view)
        if type(count) is not int or not 0 < count <= len(view):
            raise SnapshotError("backup file write made no progress")
        view = view[count:]


class _Pool:
    def __init__(self, root: Path, max_bytes: int, min_free: int):
        self.root, self.max_bytes, self.min_free = root, max_bytes, min_free

    def usage(self) -> int:
        total = 0
        for base, directories, files in os.walk(self.root, followlinks=False):
            for name in directories + files:
                path = Path(base) / name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise SnapshotError("symlinks are forbidden inside the backup pool")
                if stat.S_ISREG(metadata.st_mode):
                    total += max(metadata.st_size, getattr(metadata, "st_blocks", 0) * 512)
                elif not stat.S_ISDIR(metadata.st_mode):
                    raise SnapshotError("nonregular entry in backup pool")
        return total

    def space(self, extra: int):
        # Account for allocated file blocks as well as logical bytes. Reserve
        # one filesystem block for each pending write's final partial block.
        reserve = extra + (4096 if extra else 0)
        if self.usage() + reserve > self.max_bytes:
            raise SnapshotError("backup pool capacity exceeded (including staging)")
        if shutil.disk_usage(self.root).free - reserve < self.min_free:
            raise SnapshotError("backup free-space floor would be violated")

    def write(self, path: Path, raw: bytes):
        self.space(len(raw))
        with path.open("xb", buffering=0) as out:
            _write_all(out, raw)
            os.fsync(out.fileno())

    def verify_blob(self, sha: str, size: int | None = None) -> bool:
        path = self.root / "blobs" / sha
        if not path.exists():
            return False
        if path.is_symlink() or not path.is_file():
            raise SnapshotError("invalid existing CAS entry")
        if size is not None and path.stat().st_size != size:
            raise SnapshotError("existing CAS blob has the wrong size")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != sha:
            raise SnapshotError("existing CAS blob failed SHA256 verification")
        return True

    def initialize(self):
        allowed = {"pool.lock", "pool.json", "catalog.json", "blobs", "snapshots", ".staging"}
        if any(path.name not in allowed for path in self.root.iterdir()):
            raise SnapshotError("refusing an unowned or unexpected pool entry")
        marker = self.root / "pool.json"
        if marker.exists():
            if marker.is_symlink() or _json(_read_small(marker, 4096)) != {"schema": POOL_SCHEMA}:
                raise SnapshotError("invalid backup pool ownership marker")
        else:
            if any(path.name != "pool.lock" for path in self.root.iterdir()):
                raise SnapshotError("refusing to adopt a nonempty backup pool")
            self.write(marker, _json_bytes({"schema": POOL_SCHEMA}))
        for name in ("blobs", "snapshots", ".staging"):
            path = self.root / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise SnapshotError("invalid backup pool directory")
            path.mkdir(exist_ok=True)
        self.recover_staging()
        self.space(0)

    def recover_staging(self):
        """Remove only marked, backup-owned abandoned transactions under lock.

        No catalog can reference staging. Unknown directories remain untouched
        and count against the pool limit; an operator must review those entries.
        CAS garbage collection still happens only after catalog publication.
        """
        for directory in (self.root / ".staging").iterdir():
            match = re.fullmatch(r"receive-([0-9a-f]{32})", directory.name)
            if not match or directory.is_symlink() or not directory.is_dir():
                continue
            marker = directory / "transaction.json"
            if (marker.is_symlink() or not marker.is_file()
                    or marker.stat().st_size > 4096):
                continue
            try:
                owner = _json(_read_small(marker, 4096))
            except SnapshotError:
                continue
            if owner != {"schema": STAGING_SCHEMA, "transaction_id": match[1]}:
                continue
            entries = list(directory.iterdir())
            if any(path.is_symlink() or not path.is_file()
                   or (not SHA_RE.fullmatch(path.name) and path.name not in
                       {"transaction.json", "receipt.json", "catalog.json"}) for path in entries):
                continue
            shutil.rmtree(directory)

    def catalog(self) -> tuple[list[str], dict[str, dict]]:
        path = self.root / "catalog.json"
        if not path.exists():
            return [], {}
        value = _json(_read_small(path, MAX_COMPLETE_BYTES))
        ids = value.get("snapshots")
        if (value.get("schema") != CATALOG_SCHEMA or not isinstance(ids, list)
                or len(ids) > 2 or len(set(ids)) != len(ids)
                or any(not isinstance(s, str) or not SHA_RE.fullmatch(s) for s in ids)):
            raise SnapshotError("invalid snapshot catalog")
        headers = {}
        verified = set()
        for snapshot_id in ids:
            receipt = _json(_read_small(self.root / "snapshots" / f"{snapshot_id}.json",
                                       MAX_COMPLETE_BYTES))
            if (receipt.get("schema") != RECEIPT_SCHEMA
                    or receipt.get("snapshot_id") != snapshot_id
                    or receipt.get("header_sha256") != snapshot_id):
                raise SnapshotError("invalid published snapshot receipt")
            raw = _read_small(self.root / "blobs" / snapshot_id, MAX_HEADER_BYTES)
            if hashlib.sha256(raw).hexdigest() != snapshot_id:
                raise SnapshotError("published snapshot header is missing")
            headers[snapshot_id] = _manifest(raw, 5 * GiB)
            for item in headers[snapshot_id]["files"]:
                if item["sha256"] not in verified:
                    if not self.verify_blob(item["sha256"], item["size"]):
                        raise SnapshotError("published snapshot blob is missing")
                    verified.add(item["sha256"])
        return ids, headers

    def gc(self, ids: list[str], headers: dict[str, dict]) -> list[str]:
        live = set(ids)
        for snapshot_id in ids:
            live.update(item["sha256"] for item in headers[snapshot_id]["files"])
        warnings = []
        for directory, keep, suffix in (("blobs", live, ""), ("snapshots", set(ids), ".json")):
            for path in (self.root / directory).iterdir():
                if suffix and not path.name.endswith(suffix):
                    continue
                key = path.name[:-len(suffix)] if suffix and path.name.endswith(suffix) else path.name
                if SHA_RE.fullmatch(key) and path.is_file() and not path.is_symlink() and key not in keep:
                    try:
                        path.unlink()
                    except OSError as exc:
                        warnings.append(str(exc))
        return warnings


@contextmanager
def _locked_pool(path: Path, cleanup: Callable[[], None]):
    if ".." in path.parts:
        raise SnapshotError("backup pool path cannot contain parent traversal")
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):
            raise SnapshotError("backup pool and ancestors must be real directories")
    path.mkdir(parents=True, exist_ok=True)
    lock = path / "pool.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SnapshotError("another snapshot receiver owns the pool lock") from exc
        yield
    finally:
        try:
            cleanup()
        finally:
            os.close(fd)


def receive_snapshot(stream: BinaryIO, pool: Path, *, confirm_source_success: Callable[[], bool],
                     max_snapshot_bytes: int = 5 * GiB, max_pool_bytes: int = 12 * GiB,
                     min_free_bytes: int = 8 * GiB, retain: int = 2) -> dict:
    """Receive one strict USTAR snapshot and atomically retain two receipts.

    ``confirm_source_success`` runs after EOF and must return exactly ``True``.
    It should wait for the sender/transport to exit successfully. Limits include
    both existing data and staging; lack of room rejects the new receive without
    pruning previous snapshots. ``manifest_path`` names the original raw header
    in CAS, while ``catalog_path`` is the authority for committed snapshots.
    """
    if (any(type(value) is not int for value in
            (max_snapshot_bytes, max_pool_bytes, min_free_bytes, retain))
            or max_snapshot_bytes <= 0 or max_pool_bytes <= 0 or min_free_bytes < 0
            or retain not in (1, 2) or not callable(confirm_source_success)):
        raise SnapshotError("invalid snapshot limits or confirmation callback")
    root = Path(pool).absolute()
    staging = None
    committed = False
    def cleanup_failed_receive():
        if staging is not None and not committed:
            shutil.rmtree(staging, ignore_errors=True)
    try:
        with _locked_pool(root, cleanup_failed_receive):
            store = _Pool(root, max_pool_bytes, min_free_bytes)
            store.initialize()
            ids, headers = store.catalog()
            tar = _TarStream(stream, max_snapshot_bytes + MAX_HEADER_BYTES + MAX_COMPLETE_BYTES
                             + (MAX_FILES + 2) * 1024 + 10240)
            header_size = tar.member("snapshot.json", maximum=MAX_HEADER_BYTES)
            raw = tar.read(header_size)
            tar.padding(header_size)
            header = _manifest(raw, max_snapshot_bytes)
            snapshot_id = hashlib.sha256(raw).hexdigest()
            items = sorted(header["files"], key=lambda item: item["sha256"])
            missing = {item["sha256"]: item["size"] for item in items
                       if not store.verify_blob(item["sha256"], item["size"])}
            if not store.verify_blob(snapshot_id, len(raw)):
                missing[snapshot_id] = len(raw)
            next_ids = ([snapshot_id] + [item for item in ids if item != snapshot_id])[:retain]
            received_at = datetime.now(timezone.utc).isoformat()
            receipt_path = root / "snapshots" / f"{snapshot_id}.json"
            receipt = {"schema": RECEIPT_SCHEMA, "snapshot_id": snapshot_id,
                       "header_sha256": snapshot_id, "received_at": received_at,
                       "blob_count": len(items), "total_blob_bytes": sum(x["size"] for x in items)}
            if snapshot_id in ids:
                receipt = _json(_read_small(receipt_path, MAX_COMPLETE_BYTES))
            receipt_bytes = _json_bytes(receipt)
            catalog_bytes = _json_bytes({"schema": CATALOG_SCHEMA, "snapshots": next_ids})
            store.space(sum(missing.values()) + len(receipt_bytes) + len(catalog_bytes)
                        + (len(missing) + 3) * 4096)
            transaction_id = uuid.uuid4().hex
            staging = root / ".staging" / f"receive-{transaction_id}"
            staging.mkdir(mode=0o700)
            store.write(staging / "transaction.json",
                        _json_bytes({"schema": STAGING_SCHEMA, "transaction_id": transaction_id}))
            if snapshot_id in missing:
                store.write(staging / snapshot_id, raw)
            bytes_written = len(raw) if snapshot_id in missing else 0
            for item in items:
                sha, size = item["sha256"], item["size"]
                tar.member(item["member"], size)
                digest = hashlib.sha256()
                out = (staging / sha).open("xb", buffering=0) if sha in missing else None
                try:
                    left = size
                    while left:
                        chunk = tar.read(min(left, CHUNK_BYTES))
                        digest.update(chunk)
                        if out is not None:
                            store.space(len(chunk))
                            _write_all(out, chunk)
                            bytes_written += len(chunk)
                        left -= len(chunk)
                    if out is not None:
                        os.fsync(out.fileno())
                finally:
                    if out is not None:
                        out.close()
                tar.padding(size)
                if digest.hexdigest() != sha:
                    raise SnapshotError(f"blob SHA256 mismatch: {sha}")
            complete_size = tar.member("complete.json", maximum=MAX_COMPLETE_BYTES)
            complete = _json(tar.read(complete_size))
            tar.padding(complete_size)
            if (complete.get("schema") != COMPLETE_SCHEMA
                    or complete.get("snapshot_sha256") != snapshot_id
                    or type(complete.get("blob_count")) is not int
                    or complete["blob_count"] != len(items)
                    or type(complete.get("total_blob_bytes")) is not int
                    or complete["total_blob_bytes"] != receipt["total_blob_bytes"]):
                raise SnapshotError("invalid snapshot completion marker")
            tar.finish()
            if confirm_source_success() is not True:
                raise SnapshotError("source transport did not confirm success")
            store.space(len(receipt_bytes) + len(catalog_bytes) + 8192)
            if snapshot_id not in ids:
                store.write(staging / "receipt.json", receipt_bytes)
            store.write(staging / "catalog.json", catalog_bytes)
            # Verify bytes actually persisted, not just the incoming stream.
            # This also covers short writes and storage corruption before any
            # CAS object or receipt can become authoritative.
            for sha, size in missing.items():
                path = staging / sha
                if path.stat().st_size != size:
                    raise SnapshotError("staged blob size mismatch")
                with path.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != sha:
                        raise SnapshotError("staged blob SHA256 mismatch")
            if (snapshot_id not in ids and
                    _read_small(staging / "receipt.json", MAX_COMPLETE_BYTES) != receipt_bytes):
                raise SnapshotError("staged receipt differs from verified metadata")
            if _read_small(staging / "catalog.json", MAX_COMPLETE_BYTES) != catalog_bytes:
                raise SnapshotError("staged catalog differs from verified metadata")
            # All data is verified and transport is successful before publishing
            # immutable objects. Only the final catalog rename commits a snapshot.
            for sha in missing:
                os.replace(staging / sha, root / "blobs" / sha)
            _fsync_dir(root / "blobs")
            if snapshot_id not in ids:
                os.replace(staging / "receipt.json", receipt_path)
                _fsync_dir(root / "snapshots")
            os.replace(staging / "catalog.json", root / "catalog.json")
            committed = True
            warnings = []
            durable = True
            try:
                _fsync_dir(root)
            except Exception as exc:
                durable = False
                warnings.append(f"catalog published but directory fsync failed; GC skipped: {exc}")
            headers[snapshot_id] = header
            if durable:
                try:
                    warnings.extend(store.gc(next_ids, headers))
                except Exception as exc:
                    warnings.append(f"catalog committed; garbage collection failed: {exc}")
            try:
                shutil.rmtree(staging)
            except Exception as exc:
                warnings.append(f"catalog committed; staging cleanup failed: {exc}")
            staging = None
            try:
                pool_bytes = store.usage()
            except Exception as exc:
                pool_bytes = None
                warnings.append(f"catalog committed; pool usage unavailable: {exc}")
            return {"snapshot_id": snapshot_id, "manifest_path": str(root / "blobs" / snapshot_id),
                    "catalog_path": str(root / "catalog.json"), "receipt_path": str(receipt_path),
                    "received_at": receipt["received_at"], "blob_count": len(items),
                    "total_blob_bytes": receipt["total_blob_bytes"], "bytes_written": bytes_written,
                    "blobs_reused": sum(item["sha256"] not in missing for item in items),
                    "pool_bytes": pool_bytes, "retained_snapshot_ids": next_ids,
                    "gc_warnings": warnings}
    except SnapshotError:
        raise
    except Exception as exc:
        raise SnapshotError(f"snapshot receive failed: {exc}") from exc
