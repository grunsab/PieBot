"""Contracts for pinned, separately installed ranked comparison engines."""

import hashlib
import io
import json
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts import fetch_ranked_engines as fetch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(payload=b"engine"):
    return {
        "id": "test-1", "name": "Test 1", "rating": 3508, "rank": 100,
        "rating_source": "https://example.org/rating", "rating_snapshot": "2026-09-05",
        "rating_threads": 1, "repository": "example/test", "tag": "v1",
        "source_commit": "a" * 40, "license": "MIT",
        "license_url": "https://example.org/LICENSE", "url": "https://example.org/test",
        "bytes": len(payload), "sha256": digest(payload), "format": "binary",
        "executable": "test", "expected_uci_name": "^Test 1$",
        "single_thread_only": False, "uci_options": {},
    }


def download_fixture(payload):
    def download(url, path):
        path.write_bytes(b"MIT license" if url.endswith("LICENSE") else payload)
    return download


class RankedInstallationTests(unittest.TestCase):
    def test_checksum_mismatch_leaves_no_installed_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetch.install_engine(descriptor(), dest, downloader=download_fixture(b"wrong!"))
            self.assertFalse((dest / "test-1").exists())

    def test_verified_existing_install_skips_network_and_rejects_metadata_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            spec = descriptor()
            first = fetch.install_engine(spec, dest, downloader=download_fixture(b"engine"))
            with mock.patch.object(fetch, "download", side_effect=AssertionError("network")):
                second = fetch.install_engine(spec, dest)
            self.assertEqual(first, second)
            self.assertEqual(first["executable_sha256"], digest(b"engine"))
            with self.assertRaisesRegex(ValueError, "inventory"):
                fetch.install_engine(dict(spec, rating=3509), dest)

    def test_manifest_covers_network_runtime_license_and_rejects_any_drift(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("bundle/test", "engine")
            archive.writestr("bundle/model.nnue", "network")
            archive.writestr("bundle/runtime.so", "runtime")
        data = payload.getvalue()
        spec = dict(descriptor(data), format="zip", executable="bundle/test",
                    uci_options={"EvalFile": "bundle/model.nnue"})
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            manifest = fetch.install_engine(spec, dest, downloader=download_fixture(data))
            files = {f["path"] for f in manifest["files"]}
            self.assertTrue({"package/bundle/test", "package/bundle/model.nnue",
                             "package/bundle/runtime.so", "LICENSE.upstream", "source.download"} <= files)
            self.assertEqual(manifest["uci_options"]["EvalFile"],
                             str(dest.resolve() / "test-1/package/bundle/model.nnue"))
            self.assertTrue(all(Path(item["path"]).is_absolute() for item in manifest["auxiliary_files"]))
            (dest / "test-1/package/bundle/runtime.so").write_text("changed")
            with self.assertRaisesRegex(ValueError, "installed files"):
                fetch.verify_existing(dest / "test-1", spec)

    def test_unexpected_file_and_symlink_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            spec = descriptor()
            fetch.install_engine(spec, dest, downloader=download_fixture(b"engine"))
            extra = dest / "test-1/package/unexpected.nnue"
            extra.write_bytes(b"surprise")
            with self.assertRaisesRegex(ValueError, "installed files"):
                fetch.verify_existing(dest / "test-1", spec)
            extra.unlink()
            extra.symlink_to("test")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                fetch.verify_existing(dest / "test-1", spec)

    def test_qualification_failure_does_not_publish_partial_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(fetch, "qualify_engine", side_effect=ValueError("bad UCI")):
                with self.assertRaisesRegex(ValueError, "bad UCI"):
                    fetch.install_engine(descriptor(), Path(tmp), qualify=True,
                                         downloader=download_fixture(b"engine"))
            self.assertFalse((Path(tmp) / "test-1").exists())

    def test_dry_run_does_not_create_output_or_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"schema": fetch.INVENTORY_SCHEMA,
                                             "default_cohort": ["test-1"],
                                             "engines": [descriptor()]}))
            with mock.patch.object(fetch, "download", side_effect=AssertionError("network")), mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(fetch.main(["--inventory", str(inventory), "--dest",
                                             str(root / "absent"), "--dry-run"]), 0)
            self.assertFalse((root / "absent").exists())


class ArchiveSafetyTests(unittest.TestCase):
    def test_tar_traversal_links_special_files_and_duplicate_members_are_rejected(self):
        for kind in ("traversal", "symlink", "hardlink", "fifo", "duplicate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / "source.tar"
                with tarfile.open(archive, "w") as out:
                    item = tarfile.TarInfo("../escape" if kind == "traversal" else "file")
                    if kind == "symlink":
                        item.type = tarfile.SYMTYPE
                        item.linkname = "../../escape"
                    elif kind == "hardlink":
                        item.type = tarfile.LNKTYPE
                        item.linkname = "../escape"
                    elif kind == "fifo":
                        item.type = tarfile.FIFOTYPE
                    out.addfile(item)
                    if kind == "duplicate":
                        out.addfile(item)
                with self.assertRaises(ValueError):
                    fetch.extract_archive(archive, root / "package", "tar")
                self.assertFalse((root / "escape").exists())

    def test_zip_rejects_absolute_windows_traversal_and_links(self):
        for name in ("/escape", "../escape", "C:/escape", "..\\escape", "link"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / "source.zip"
                with zipfile.ZipFile(archive, "w") as out:
                    item = zipfile.ZipInfo(name)
                    if name == "link":
                        item.create_system = 3
                        item.external_attr = (stat.S_IFLNK | 0o777) << 16
                    out.writestr(item, "outside")
                with self.assertRaises(ValueError):
                    fetch.extract_archive(archive, root / "package", "zip")

    def test_uncompressed_byte_limit_prevents_archive_bombs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr("large", b"x" * 20)
            with self.assertRaisesRegex(ValueError, "limit"):
                fetch.extract_archive(archive, root / "package", "zip", max_bytes=10)


class UciQualificationTests(unittest.TestCase):
    def test_explicitly_disabled_tablebases_allow_only_the_known_empty_path_notice(self):
        cases = (
            (True, 'info string Tablebases failed to load', {}, True),
            (False, 'info string Tablebases failed to load', {}, False),
            (True, 'info string Tablebases failed to load', {'SyzygyPath': '/missing'}, False),
            (True, 'info string Network failed to load', {}, False),
            (True, 'info string Tablebases failed to load: missing runtime', {}, False),
        )
        for advertises_path, notice, overrides, accepted in cases:
            with self.subTest(notice=notice, path=advertises_path), tempfile.TemporaryDirectory() as tmp:
                executable = Path(tmp) / "test"
                executable.write_text(f"""#!/usr/bin/env python3
import sys
for line in sys.stdin:
    line = line.strip()
    if line == 'uci':
        print('id name Test 1')
        print('option name Threads type spin default 4 min 1 max 16')
        if {advertises_path!r}: print('option name SyzygyPath type string default <empty>')
        print('uciok', flush=True)
    elif line == 'isready':
        print({notice!r})
        print('readyok', flush=True)
    elif line.startswith('go '): print('bestmove e2e4', flush=True)
    elif line == 'quit': break
""")
                executable.chmod(0o755)
                if accepted:
                    result = fetch.qualify_engine(executable, descriptor(), overrides)
                    self.assertEqual(result['options']['SyzygyPath'], '')
                    self.assertEqual(result['status'], 'passed')
                else:
                    with self.assertRaisesRegex(ValueError, 'rejected qualification options'):
                        fetch.qualify_engine(executable, descriptor(), overrides)

    def test_bounded_search_records_legal_move_name_and_single_thread_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "test"
            executable.write_text("""#!/usr/bin/env python3
import sys
for line in sys.stdin:
    line = line.strip()
    if line == 'uci':
        print('id name Test 1')
        print('option name Threads type spin default 4 min 1 max 16')
        print('option name Hash type spin default 16 min 1 max 1024')
        print('uciok', flush=True)
    elif line == 'isready': print('readyok', flush=True)
    elif line.startswith('go '): print('bestmove e2e4', flush=True)
    elif line == 'quit': break
""")
            executable.chmod(0o755)
            result = fetch.qualify_engine(executable, descriptor(), {})
            self.assertEqual(result["name"], "Test 1")
            self.assertEqual(result["options"]["Threads"], 1)
            self.assertEqual(result["bestmove"], "e2e4")
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["single_thread_qualified"])

    def test_version_mismatch_refuses_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "test"
            executable.write_text("#!/usr/bin/env python3\nimport sys\nfor line in sys.stdin:\n"
                                  " if line.strip() == 'uci': print('id name Other 2\\nuciok', flush=True)\n")
            executable.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "identity"):
                fetch.qualify_engine(executable, descriptor(), {}, timeout=1)


if __name__ == "__main__":
    unittest.main()
