import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "fetch_lc0_pgns.py"
_spec = importlib.util.spec_from_file_location(
    "training.nnue.fetch_lc0_pgns", MODULE_PATH
)
module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = module
_spec.loader.exec_module(module)  # type: ignore[attr-defined]


class FetchPgnTests(unittest.TestCase):
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

    def test_zero_byte_and_part_files_are_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "games.pgn.zst"
            dest.touch()
            self.assertFalse(module.is_complete_download(dest))

            dest.write_bytes(b"complete")
            self.assertTrue(module.is_complete_download(dest))

            Path(f"{dest}.part").write_bytes(b"partial")
            self.assertFalse(module.is_complete_download(dest))

    def test_interrupted_download_cleans_part_and_preserves_destination(self) -> None:
        def interrupted_chunks():
            yield b"new"
            raise OSError("connection reset")

        response = self._Response(interrupted_chunks())
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "games.pgn.zst"
            dest.write_bytes(b"old-complete")

            with mock.patch.object(
                module, "_require_requests", return_value=self._Requests(response)
            ):
                with self.assertRaisesRegex(OSError, "connection reset"):
                    module.download("https://example.com/games.pgn.zst", dest)

            self.assertEqual(b"old-complete", dest.read_bytes())
            self.assertFalse(Path(f"{dest}.part").exists())

    def test_download_rejects_truncated_content_length(self) -> None:
        response = self._Response([b"short"], content_length=100)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "games.pgn.zst"
            with mock.patch.object(
                module, "_require_requests", return_value=self._Requests(response)
            ):
                with self.assertRaisesRegex(RuntimeError, "expected 100 bytes"):
                    module.download("https://example.com/games.pgn.zst", dest)

            self.assertFalse(dest.exists())
            self.assertFalse(Path(f"{dest}.part").exists())

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
                suites=["broken/"],
                limit_per_suite=1,
                skip_existing=True,
                decompress=False,
                concurrency=1,
                manifest=None,
            )
            with (
                mock.patch.object(module, "parse_args", return_value=args),
                mock.patch.object(
                    module, "list_dir", side_effect=OSError("listing unavailable")
                ),
            ):
                rc = module.main()

            self.assertEqual(1, rc)
            written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                "listing unavailable", written["suites"]["broken/"]["error"]
            )
            self.assertTrue(written["failures"])

    def test_main_returns_failure_and_updates_manifest_for_download_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(
                out=out,
                suites=["test80/"],
                limit_per_suite=1,
                skip_existing=False,
                decompress=False,
                concurrency=1,
                manifest=None,
            )
            with (
                mock.patch.object(module, "parse_args", return_value=args),
                mock.patch.object(
                    module, "list_dir", return_value=["sample.pgn.zst"]
                ),
                mock.patch.object(module, "head_last_modified", return_value=123.0),
                mock.patch.object(
                    module, "download", side_effect=OSError("network lost")
                ),
            ):
                rc = module.main()

            self.assertEqual(1, rc)
            written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("download", written["failures"][0]["stage"])
            self.assertIn("network lost", written["suites"]["test80/"][0]["status"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
