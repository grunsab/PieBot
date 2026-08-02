import datetime as _dt
import json
import tempfile
import unittest
from pathlib import Path

import importlib.util
from types import SimpleNamespace
from typing import Optional
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "fetch_lc0_bins.py"
_spec = importlib.util.spec_from_file_location("training.nnue.fetch_lc0_bins", MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = module
_spec.loader.exec_module(module)  # type: ignore[attr-defined]


class ChooseRecentFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.threshold = _dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc)
        self.suite_url = "https://example.com/test80/"

    def test_filters_suffix_and_threshold(self) -> None:
        hrefs = [
            "2025-02-10-0000.bin",
            "2024-12-15-0001.bin",
            "2025-03-01-0002.bin.zst",
            "notes.txt",
        ]
        ts_map = {
            self.suite_url + "2025-02-10-0000.bin": _dt.datetime(2025, 2, 11, tzinfo=_dt.timezone.utc),
            self.suite_url + "2024-12-15-0001.bin": _dt.datetime(2024, 12, 16, tzinfo=_dt.timezone.utc),
            self.suite_url + "2025-03-01-0002.bin.zst": _dt.datetime(2025, 3, 2, tzinfo=_dt.timezone.utc),
        }

        def fake_head(url: str) -> Optional[_dt.datetime]:
            return ts_map.get(url)

        picked = module.choose_recent_files(
            self.suite_url,
            hrefs,
            limit=10,
            threshold=self.threshold,
            head_func=fake_head,
        )
        self.assertEqual(["2025-03-01-0002.bin.zst", "2025-02-10-0000.bin"], [p.name for p in picked])

    def test_limit_is_respected(self) -> None:
        hrefs = [f"2025-04-0{i}.bin" for i in range(5)]
        base_dt = _dt.datetime(2025, 4, 10, tzinfo=_dt.timezone.utc)

        def fake_head(url: str) -> Optional[_dt.datetime]:
            idx = int(url.split("-0")[-1].split(".")[0])
            return base_dt + _dt.timedelta(days=idx)

        picked = module.choose_recent_files(
            self.suite_url,
            hrefs,
            limit=2,
            threshold=self.threshold,
            head_func=fake_head,
        )
        self.assertEqual(2, len(picked))
        self.assertEqual(["2025-04-04.bin", "2025-04-03.bin"], [p.name for p in picked])


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.threshold = _dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc)

    def test_plan_downloads_skips_existing(self) -> None:
        hrefs = ["fileA.bin", "fileB.bin", "fileC.bin", "fileD.bin"]
        suite = "test90/"
        suite_url = module.TRAINING_DATA_BASE + suite
        sample_dt = _dt.datetime(2025, 2, 1, tzinfo=_dt.timezone.utc)

        def fake_list(url: str) -> list[str]:
            self.assertEqual(suite_url, url)
            return hrefs

        def fake_head(url: str) -> _dt.datetime:
            return sample_dt

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            existing = temp_dir / "test90" / "fileB.bin"
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"complete")
            empty = existing.parent / "fileC.bin"
            empty.touch()
            partial = existing.parent / "fileD.bin.part"
            partial.write_bytes(b"partial")

            manifest, to_download = module.plan_suite_downloads(
                suites=[suite],
                out_dir=temp_dir,
                threshold=self.threshold,
                limit_per_suite=10,
                list_func=fake_list,
                head_func=fake_head,
                skip_existing=True,
            )
            self.assertEqual(
                [
                    existing.parent / "fileA.bin",
                    existing.parent / "fileC.bin",
                    existing.parent / "fileD.bin",
                ],
                [job.dest for job in to_download],
            )
            suite_manifest = manifest["suites"]["test90/"]
            self.assertEqual(
                {"fileA.bin", "fileB.bin", "fileC.bin", "fileD.bin"},
                {entry["name"] for entry in suite_manifest},
            )
            status_by_name = {entry["name"]: entry["status"] for entry in suite_manifest}
            self.assertEqual("queued", status_by_name["fileA.bin"])
            self.assertEqual("exists", status_by_name["fileB.bin"])
            self.assertEqual("queued", status_by_name["fileC.bin"])
            self.assertEqual("queued", status_by_name["fileD.bin"])

    def test_plan_downloads_records_metadata_lookup_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, jobs = module.plan_suite_downloads(
                suites=["test90/"],
                out_dir=Path(tmp),
                threshold=self.threshold,
                limit_per_suite=10,
                list_func=lambda _url: ["unreachable.bin"],
                head_func=lambda _url: None,
            )

        self.assertEqual([], jobs)
        self.assertEqual("head", manifest["failures"][0]["stage"])
        self.assertIn("unreachable.bin", manifest["failures"][0]["url"])

    def test_atomic_manifest_write_preserves_previous_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"old": true}\n', encoding="utf-8")

            with mock.patch.object(module.json, "dump", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    module.write_json_atomic(path, {"new": True})

            self.assertEqual('{"old": true}\n', path.read_text(encoding="utf-8"))
            self.assertFalse(Path(f"{path}.part").exists())

    def test_main_returns_failure_and_records_listing_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(
                out=out,
                days=90,
                suites=["broken/"],
                limit_per_suite=1,
                skip_existing=True,
                decompress=False,
                concurrency=1,
                manifest=None,
            )
            planned = {
                "suites": {
                    "broken/": {"error": "listing unavailable", "files": []}
                }
            }
            with (
                mock.patch.object(module, "parse_args", return_value=args),
                mock.patch.object(
                    module, "plan_suite_downloads", return_value=(planned, [])
                ),
            ):
                rc = module.main()

            self.assertEqual(1, rc)
            written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(written["failures"])

    def test_main_returns_failure_and_updates_manifest_for_download_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dest = out / "test90" / "sample.bin"
            args = SimpleNamespace(
                out=out,
                days=90,
                suites=["test90/"],
                limit_per_suite=1,
                skip_existing=False,
                decompress=False,
                concurrency=1,
                manifest=None,
            )
            entry = {
                "name": "sample.bin",
                "url": "https://example.com/sample.bin",
                "dest": str(dest),
                "status": "queued",
            }
            planned = {"suites": {"test90/": [entry]}}
            jobs = [module.DownloadJob(entry["url"], dest)]
            with (
                mock.patch.object(module, "parse_args", return_value=args),
                mock.patch.object(
                    module, "plan_suite_downloads", return_value=(planned, jobs)
                ),
                mock.patch.object(
                    module, "download", side_effect=OSError("network lost")
                ),
            ):
                rc = module.main()

            self.assertEqual(1, rc)
            written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("download", written["failures"][0]["stage"])
            self.assertIn("network lost", written["suites"]["test90/"][0]["status"])


class DownloadTests(unittest.TestCase):
    class _Response:
        def __init__(self, chunks, content_length=None):
            self._chunks = chunks
            self.headers = {}
            if content_length is not None:
                self.headers["Content-Length"] = str(content_length)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            del chunk_size
            yield from self._chunks

    class _Requests:
        def __init__(self, response):
            self.response = response

        def get(self, *_args, **_kwargs):
            return self.response

    def test_interrupted_download_cleans_part_and_preserves_destination(self) -> None:
        def interrupted_chunks():
            yield b"new"
            raise OSError("connection reset")

        response = self._Response(interrupted_chunks())
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sample.bin"
            dest.write_bytes(b"old-complete")
            job = module.DownloadJob("https://example.com/sample.bin", dest)

            with mock.patch.object(
                module, "_require_requests", return_value=self._Requests(response)
            ):
                with self.assertRaisesRegex(OSError, "connection reset"):
                    module.download(job)

            self.assertEqual(b"old-complete", dest.read_bytes())
            self.assertFalse(Path(f"{dest}.part").exists())

    def test_download_rejects_truncated_content_length(self) -> None:
        response = self._Response([b"short"], content_length=100)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sample.bin"
            job = module.DownloadJob("https://example.com/sample.bin", dest)

            with mock.patch.object(
                module, "_require_requests", return_value=self._Requests(response)
            ):
                with self.assertRaisesRegex(RuntimeError, "expected 100 bytes"):
                    module.download(job)

            self.assertFalse(dest.exists())
            self.assertFalse(Path(f"{dest}.part").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
