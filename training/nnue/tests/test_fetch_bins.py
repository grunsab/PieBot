import datetime as _dt
import unittest
from pathlib import Path

import importlib.util
from typing import Optional

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
        hrefs = ["fileA.bin", "fileB.bin"]
        suite = "test90/"
        suite_url = module.TRAINING_DATA_BASE + suite
        sample_dt = _dt.datetime(2025, 2, 1, tzinfo=_dt.timezone.utc)

        def fake_list(url: str) -> list[str]:
            self.assertEqual(suite_url, url)
            return hrefs

        def fake_head(url: str) -> _dt.datetime:
            return sample_dt

        temp_dir = Path("__test_out__")
        existing = temp_dir / "test90" / "fileB.bin"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.touch()

        try:
            manifest, to_download = module.plan_suite_downloads(
                suites=[suite],
                out_dir=temp_dir,
                threshold=self.threshold,
                limit_per_suite=10,
                list_func=fake_list,
                head_func=fake_head,
                skip_existing=True,
            )
            self.assertEqual([existing.parent / "fileA.bin"], [job.dest for job in to_download])
            suite_manifest = manifest["suites"]["test90/"]
            self.assertEqual({"fileA.bin", "fileB.bin"}, {entry["name"] for entry in suite_manifest})
            status_by_name = {entry["name"]: entry["status"] for entry in suite_manifest}
            self.assertEqual("queued", status_by_name["fileA.bin"])
            self.assertEqual("exists", status_by_name["fileB.bin"])
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
