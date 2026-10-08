"""The snapshot receiver must fail closed before publishing incomplete data."""
from __future__ import annotations

import io
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts import lc0_snapshot_store as store
from scripts.lc0_snapshot_store import SnapshotError, receive_snapshot


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def member(name, raw, *, kind=tarfile.REGTYPE, size=None):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = len(raw) if size is None else size
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        info.linkname = "/outside-the-pool"
    return info.tobuf(format=tarfile.USTAR_FORMAT) + raw + bytes((-len(raw)) % 512)


def snapshot(number=1, *, change_header=None, change_complete=None, change_payload=None,
             extra=b"", trailer=bytes(1024), raw_header=None):
    state = json.dumps({"completed_chunks": number}).encode()
    weights = b"shared deterministic weights"
    payloads = {sha(state): state, sha(weights): weights}
    header = {
        "schema": store.HEADER_SCHEMA, "completed_chunks": number,
        "state_sha256": sha(state),
        "capabilities": {"model_recovery": True, "strict_campaign_resume": False,
                         "corpus_payloads_included": False},
        "files": [
            {"sha256": sha(state), "size": len(state), "member": "blobs/" + sha(state),
             "sources": [{"role": "state", "path": "/source/lc0_state.json"}]},
            {"sha256": sha(weights), "size": len(weights), "member": "blobs/" + sha(weights),
             "sources": [{"role": "active_quant", "path": "/source/active.nnue"}]},
        ],
    }
    if change_header:
        change_header(header)
    raw = raw_header if raw_header is not None else json.dumps(header, sort_keys=True).encode()
    complete = {"schema": store.COMPLETE_SCHEMA, "snapshot_sha256": sha(raw),
                "blob_count": len(payloads), "total_blob_bytes": sum(map(len, payloads.values()))}
    if change_complete:
        change_complete(complete)
    if change_payload:
        change_payload(payloads)
    data = member("snapshot.json", raw)
    data += b"".join(member("blobs/" + key, value) for key, value in sorted(payloads.items()))
    data += extra + member("complete.json", json.dumps(complete).encode()) + trailer
    return data, raw, header, payloads


class SnapshotReceiverBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.pool = Path(self.temporary.name).resolve() / "pool"

    def receive(self, data, **overrides):
        options = {"confirm_source_success": lambda: True, "min_free_bytes": 0,
                   "max_pool_bytes": 32 * 1024 ** 2, "max_snapshot_bytes": 1024 ** 2}
        options.update(overrides)
        return receive_snapshot(io.BytesIO(data), self.pool, **options)

    def test_invalid_limits_are_rejected_before_reading_transport(self):
        class Unreadable:
            def read(self, _size=-1):
                raise AssertionError("invalid limits must be checked before transport reads")

        for override in (
            {"max_snapshot_bytes": 0},
            {"max_pool_bytes": -1},
            {"min_free_bytes": -1},
            {"retain": 0},
        ):
            with self.subTest(override=override), self.assertRaises(SnapshotError):
                receive_snapshot(Unreadable(), self.pool,
                                 confirm_source_success=lambda: True, **override)

    def test_malformed_stream_never_confirms_or_publishes(self):
        confirmations = []
        with self.assertRaises(SnapshotError):
            receive_snapshot(io.BytesIO(b"not a tar stream"), self.pool,
                             confirm_source_success=lambda: confirmations.append(True))
        self.assertEqual(confirmations, [])
        self.assertFalse((self.pool / "catalog.json").exists())

    def test_existing_regular_file_is_not_treated_as_a_pool(self):
        self.pool.write_bytes(b"unrelated user data")
        with self.assertRaises(SnapshotError):
            receive_snapshot(io.BytesIO(b""), self.pool,
                             confirm_source_success=lambda: True)
        self.assertEqual(self.pool.read_bytes(), b"unrelated user data")

    def test_symlink_pool_is_rejected_without_touching_its_target(self):
        target = Path(self.temporary.name).resolve() / "unrelated"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_bytes(b"keep me")
        self.pool.symlink_to(target, target_is_directory=True)
        with self.assertRaises(SnapshotError):
            receive_snapshot(io.BytesIO(b""), self.pool,
                             confirm_source_success=lambda: True)
        self.assertEqual(sorted(p.name for p in target.iterdir()), ["keep.txt"])
        self.assertEqual(sentinel.read_bytes(), b"keep me")

    def test_unowned_nonempty_directory_is_not_adopted_or_cleaned(self):
        self.pool.mkdir()
        sentinel = self.pool / "unrelated.bin"
        sentinel.write_bytes(b"do not delete")
        with self.assertRaises(SnapshotError):
            receive_snapshot(io.BytesIO(b""), self.pool,
                             confirm_source_success=lambda: True)
        self.assertEqual(sentinel.read_bytes(), b"do not delete")

    def test_transport_read_failure_is_reported_without_confirmation(self):
        class FailedTransport:
            def read(self, _size=-1):
                raise OSError("connection closed")

        confirmations = []
        with self.assertRaises(SnapshotError):
            receive_snapshot(FailedTransport(), self.pool,
                             confirm_source_success=lambda: confirmations.append(True))
        self.assertEqual(confirmations, [])
        self.assertFalse((self.pool / "catalog.json").exists())

    def test_symlinked_pool_ancestor_is_rejected(self):
        base = Path(self.temporary.name).resolve()
        target = base / "target"
        target.mkdir()
        alias = base / "alias"
        alias.symlink_to(target, target_is_directory=True)
        self.pool = alias / "pool"
        with self.assertRaises(SnapshotError):
            self.receive(snapshot()[0])
        self.assertFalse((target / "pool").exists())

    def test_valid_snapshot_is_published_only_after_source_confirmation(self):
        data, raw, header, payloads = snapshot()
        observed = []
        def confirm():
            observed.append(not (self.pool / "catalog.json").exists())
            return True
        result = self.receive(data, confirm_source_success=confirm)
        self.assertEqual(observed, [True])
        self.assertEqual(result["snapshot_id"], sha(raw))
        self.assertEqual(Path(result["manifest_path"]).read_bytes(), raw)
        self.assertEqual(json.loads(Path(result["catalog_path"]).read_text())["snapshots"], [sha(raw)])
        self.assertEqual(result["total_blob_bytes"], sum(map(len, payloads.values())))
        for item in header["files"]:
            self.assertEqual((self.pool / item["member"]).read_bytes(), payloads[item["sha256"]])

    def test_two_snapshots_retained_and_shared_blob_not_garbage_collected(self):
        first = self.receive(snapshot(1)[0])
        second = self.receive(snapshot(2)[0])
        third = self.receive(snapshot(3)[0])
        self.assertEqual(third["retained_snapshot_ids"], [third["snapshot_id"], second["snapshot_id"]])
        self.assertFalse(Path(first["receipt_path"]).exists())
        self.assertFalse(Path(first["manifest_path"]).exists())
        self.assertEqual(len(list((self.pool / "snapshots").glob("*.json"))), 2)
        self.assertTrue((self.pool / "blobs" / sha(b"shared deterministic weights")).exists())

    def test_identical_snapshot_deduplicates_all_cas_bytes(self):
        data = snapshot()[0]
        first = self.receive(data)
        blobs = {p.name: p.read_bytes() for p in (self.pool / "blobs").iterdir()}
        again = self.receive(data)
        self.assertEqual(again["snapshot_id"], first["snapshot_id"])
        self.assertEqual(again["bytes_written"], 0)
        self.assertEqual(again["blobs_reused"], 2)
        self.assertEqual(blobs, {p.name: p.read_bytes() for p in (self.pool / "blobs").iterdir()})

    def test_source_failure_after_complete_tar_preserves_published_snapshots(self):
        self.receive(snapshot(1)[0])
        self.receive(snapshot(2)[0])
        previous = (self.pool / "catalog.json").read_bytes()
        for confirmation in (lambda: False, lambda: None, mock.Mock(side_effect=OSError("SSH failed"))):
            with self.subTest(confirmation=confirmation), self.assertRaises(SnapshotError):
                self.receive(snapshot(3)[0], confirm_source_success=confirmation)
            self.assertEqual((self.pool / "catalog.json").read_bytes(), previous)
        self.assertEqual(len(list((self.pool / "snapshots").glob("*.json"))), 2)

    def test_truncated_transport_at_header_payload_or_terminator_never_publishes(self):
        data = snapshot()[0]
        for length in (0, 511, 700, len(data) - 1024, len(data) - 512, len(data) - 1):
            with self.subTest(length=length), self.assertRaises(SnapshotError):
                self.receive(data[:length])
            self.assertFalse((self.pool / "catalog.json").exists())

    def test_checksum_mismatch_is_rejected_even_for_deduplicated_blob(self):
        self.receive(snapshot()[0])
        previous = (self.pool / "catalog.json").read_bytes()
        data = snapshot(change_payload=lambda blobs: blobs.update({key: b"x" * len(value)
                        for key, value in list(blobs.items())}))[0]
        with self.assertRaisesRegex(SnapshotError, "SHA256"):
            self.receive(data)
        self.assertEqual((self.pool / "catalog.json").read_bytes(), previous)

    def test_corrupt_reused_cas_blob_is_not_trusted(self):
        data, _, _, payloads = snapshot()
        self.receive(data)
        key = next(iter(payloads))
        (self.pool / "blobs" / key).write_bytes(b"x" * len(payloads[key]))
        with self.assertRaisesRegex(SnapshotError, "SHA256"):
            self.receive(data)

    def test_unexpected_links_paths_and_extension_members_are_rejected(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("snapshot.json", tarfile.SYMTYPE), ("snapshot.json", tarfile.LNKTYPE),
                           ("snapshot.json", tarfile.XHDTYPE), ("snapshot.json", tarfile.GNUTYPE_LONGNAME)):
            with self.subTest(name=name, kind=kind), self.assertRaises(SnapshotError):
                self.receive(member(name, b"{}", kind=kind) + bytes(1024))
        with self.assertRaises(SnapshotError):
            self.receive(snapshot(extra=member("unexpected", b"payload"))[0])

    def test_bad_complete_markers_and_nonzero_trailers_are_rejected(self):
        for change in (lambda m: m.update(snapshot_sha256="0" * 64),
                       lambda m: m.update(blob_count=True), lambda m: m.update(total_blob_bytes=0)):
            with self.subTest(change=change), self.assertRaises(SnapshotError):
                self.receive(snapshot(change_complete=change)[0])
        for trailer in (bytes(512), bytes(1024) + b"x" * 512, bytes(11264)):
            with self.subTest(trailer=len(trailer)), self.assertRaises(SnapshotError):
                self.receive(snapshot(trailer=trailer)[0])

    def test_invalid_header_contract_is_rejected(self):
        mutations = [lambda h: h.update(schema="other"),
                     lambda h: h.update(completed_chunks=True),
                     lambda h: h.update(state_sha256="0" * 64),
                     lambda h: h["capabilities"].update(strict_campaign_resume=True),
                     lambda h: h["files"][0].update(size=-1),
                     lambda h: h["files"][0].update(size=True),
                     lambda h: h["files"][0].update(member="../escape"),
                     lambda h: h["files"][0]["sources"][0].update(path="relative"),
                     lambda h: h["files"][1]["sources"][0].update(role="state"),
                     lambda h: h["files"].append(h["files"][0]),
                     lambda h: h.update(files=h["files"] * 65)]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(SnapshotError):
                self.receive(snapshot(change_header=change)[0])
        with self.assertRaises(SnapshotError):
            self.receive(snapshot(raw_header=b'{"schema":"x","schema":"y"}')[0])

    def test_size_caps_reject_declared_oversize_before_blob_reads(self):
        data = snapshot(change_header=lambda h: h["files"][0].update(size=2 * 1024 ** 2))[0]
        with self.assertRaisesRegex(SnapshotError, "max_snapshot_bytes"):
            self.receive(data)
        with self.assertRaisesRegex(SnapshotError, "too large"):
            self.receive(member("snapshot.json", b"", size=store.MAX_HEADER_BYTES + 1))

    def test_pool_cap_counts_existing_staging_without_pruning_live_snapshots(self):
        self.receive(snapshot()[0])
        previous = (self.pool / "catalog.json").read_bytes()
        stale = self.pool / ".staging" / "unknown-unowned-directory"
        stale.mkdir()
        payload = stale / "do-not-delete"
        payload.write_bytes(bytes(128 * 1024))
        with self.assertRaisesRegex(SnapshotError, "capacity"):
            self.receive(snapshot(2)[0], max_pool_bytes=128 * 1024)
        self.assertEqual((self.pool / "catalog.json").read_bytes(), previous)
        self.assertTrue(payload.exists())

    def test_free_floor_is_checked_before_receive(self):
        with mock.patch.object(store.shutil, "disk_usage", return_value=mock.Mock(free=1024)):
            with self.assertRaisesRegex(SnapshotError, "free-space"):
                self.receive(snapshot()[0], min_free_bytes=1024)

    def test_free_floor_is_rechecked_during_blob_writes(self):
        calls = []
        def available(_path):
            calls.append(1)
            return mock.Mock(free=1024 ** 3 if len(calls) < 5 else 0)
        with mock.patch.object(store.shutil, "disk_usage", side_effect=available):
            with self.assertRaisesRegex(SnapshotError, "free-space"):
                self.receive(snapshot()[0], min_free_bytes=1)
        self.assertGreaterEqual(len(calls), 5)
        self.assertFalse((self.pool / "catalog.json").exists())

    def test_single_receiver_lock_is_exclusive(self):
        self.pool.mkdir()
        with (self.pool / "pool.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(SnapshotError, "lock"):
                self.receive(snapshot()[0])

    def test_cleanup_remains_inside_pool_lock_on_failure(self):
        original = store.shutil.rmtree
        lock_was_held = []
        def remove(path, *args, **kwargs):
            with (self.pool / "pool.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_was_held.append(False)
                except BlockingIOError:
                    lock_was_held.append(True)
            return original(path, *args, **kwargs)
        with mock.patch.object(store.shutil, "rmtree", side_effect=remove):
            with self.assertRaises(SnapshotError):
                self.receive(snapshot()[0], confirm_source_success=lambda: False)
        self.assertEqual(lock_was_held, [True])

    def test_short_writes_cannot_publish_incomplete_blobs(self):
        original = Path.open
        class ShortWriter:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
            def write(self, raw): return self.stream.write(raw[:max(1, len(raw) // 2)])
            def fileno(self): return self.stream.fileno()
            def close(self): self.stream.close()
        def opened(path, mode="r", *args, **kwargs):
            stream = original(path, mode, *args, **kwargs)
            return ShortWriter(stream) if mode == "xb" else stream
        data, raw, _, payloads = snapshot()
        with mock.patch.object(Path, "open", opened):
            result = self.receive(data)
        self.assertEqual(Path(result["manifest_path"]).read_bytes(), raw)
        for key, value in payloads.items():
            self.assertEqual((self.pool / "blobs" / key).read_bytes(), value)

    def test_post_commit_gc_failure_returns_committed_receipt_with_warning(self):
        with mock.patch.object(store._Pool, "gc", side_effect=OSError("GC denied")):
            result = self.receive(snapshot()[0])
        self.assertTrue(result["gc_warnings"])
        self.assertEqual(json.loads(Path(result["catalog_path"]).read_text())["snapshots"],
                         [result["snapshot_id"]])

    def test_post_commit_cleanup_and_usage_failures_are_committed_with_warnings(self):
        original_usage = store._Pool.usage
        original_remove = store.shutil.rmtree
        def usage(pool):
            if (pool.root / "catalog.json").exists():
                raise OSError("usage probe failed")
            return original_usage(pool)
        def remove(path, *args, **kwargs):
            if (self.pool / "catalog.json").exists():
                raise OSError("cleanup failed")
            return original_remove(path, *args, **kwargs)
        with mock.patch.object(store._Pool, "usage", usage), \
                mock.patch.object(store.shutil, "rmtree", side_effect=remove):
            result = self.receive(snapshot()[0])
        self.assertIsNone(result["pool_bytes"])
        self.assertEqual(len(result["gc_warnings"]), 2)
        self.assertTrue(Path(result["catalog_path"]).exists())

    def test_catalog_directory_fsync_failure_skips_gc_after_publication(self):
        self.receive(snapshot(1)[0])
        self.receive(snapshot(2)[0])
        before = {path.name for path in (self.pool / "blobs").iterdir()}
        original = store._fsync_dir
        def sync(path):
            if path == self.pool:
                raise OSError("directory sync failed")
            return original(path)
        with mock.patch.object(store, "_fsync_dir", side_effect=sync), \
                mock.patch.object(store._Pool, "gc", side_effect=AssertionError("must skip GC")):
            result = self.receive(snapshot(3)[0])
        self.assertTrue(result["gc_warnings"])
        self.assertTrue(before <= {path.name for path in (self.pool / "blobs").iterdir()})

    def test_marked_stale_transaction_is_recovered_before_capacity_admission(self):
        self.receive(snapshot()[0])
        transaction_id = "a" * 32
        stale = self.pool / ".staging" / ("receive-" + transaction_id)
        stale.mkdir()
        (stale / "transaction.json").write_text(json.dumps({
            "schema": "piebot-lc0-snapshot-staging-v1", "transaction_id": transaction_id}))
        (stale / ("b" * 64)).write_bytes(bytes(256 * 1024))
        result = self.receive(snapshot(2)[0], max_pool_bytes=128 * 1024)
        self.assertFalse(stale.exists())
        self.assertEqual(len(result["retained_snapshot_ids"]), 2)

    def test_wrong_staging_marker_is_not_cleaned(self):
        self.receive(snapshot()[0])
        stale = self.pool / ".staging" / ("receive-" + "a" * 32)
        stale.mkdir()
        marker = stale / "transaction.json"
        marker.write_text(json.dumps({"schema": "unrelated"}))
        self.receive(snapshot(2)[0])
        self.assertTrue(marker.exists())

    def test_gc_leaves_files_outside_the_owned_receipt_namespace_untouched(self):
        self.receive(snapshot()[0])
        unrelated = self.pool / "snapshots" / ("f" * 64)
        unrelated.write_bytes(b"not a snapshot receipt")
        self.receive(snapshot(2)[0])
        self.assertEqual(unrelated.read_bytes(), b"not a snapshot receipt")

    def test_oversized_pool_catalog_is_rejected_before_reading_it(self):
        self.receive(snapshot()[0])
        catalog = self.pool / "catalog.json"
        catalog.write_bytes(b" " * (64 * 1024 + 1))
        original = Path.read_bytes
        reads = []
        def read(path):
            reads.append(path)
            return original(path)
        with mock.patch.object(Path, "read_bytes", read), self.assertRaises(SnapshotError):
            self.receive(snapshot(2)[0])
        self.assertNotIn(catalog, reads)

    def test_staged_byte_corruption_cannot_publish_a_verified_snapshot(self):
        original = store.os.fsync
        corrupted = []
        def sync(fd):
            original(fd)
            if not corrupted:
                for path in (self.pool / ".staging").glob("receive-*/*"):
                    if store.SHA_RE.fullmatch(path.name) and path.stat().st_size:
                        with path.open("r+b") as out:
                            out.write(b"!")
                        corrupted.append(path)
                        break
        with mock.patch.object(store.os, "fsync", side_effect=sync):
            with self.assertRaises(SnapshotError):
                self.receive(snapshot()[0])
        self.assertTrue(corrupted)
        self.assertFalse((self.pool / "catalog.json").exists())

    def test_precommit_catalog_replace_failure_preserves_previous_catalog(self):
        self.receive(snapshot()[0])
        previous = (self.pool / "catalog.json").read_bytes()
        original = store.os.replace
        def replace(source, target):
            if Path(target).name == "catalog.json":
                raise OSError("publication denied")
            return original(source, target)
        with mock.patch.object(store.os, "replace", side_effect=replace):
            with self.assertRaises(SnapshotError):
                self.receive(snapshot(2)[0])
        self.assertEqual((self.pool / "catalog.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
