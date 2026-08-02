"""Fetch recent LCZero training BIN files for NNUE training."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

TRAINING_DATA_BASE = "https://storage.lczero.org/files/training_data/"
DEFAULT_SUITES: tuple[str, ...] = ("test90/", "test80/")
BIN_SUFFIXES: tuple[str, ...] = (".bin", ".bin.zst", ".bin.zstd")
LINK_RE = re.compile(r'<a href="([^"]+)">')


def _require_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised by import-only tests
        raise RuntimeError("requests is required for remote BIN fetching") from exc
    return requests


@dataclass(frozen=True)
class FileCandidate:
    """Metadata about a BIN file candidate discovered in remote storage."""

    name: str
    url: str
    last_modified: _dt.datetime


@dataclass
class DownloadJob:
    """A download job pairing remote URL with local destination."""

    url: str
    dest: Path


def _part_path(path: Path) -> Path:
    """Return the same-directory staging path used for atomic replacement."""

    return Path(f"{path}.part")


def is_complete_download(path: Path) -> bool:
    """Return whether ``path`` is a non-empty, atomically completed download."""

    try:
        return path.is_file() and path.stat().st_size > 0 and not _part_path(path).exists()
    except OSError:
        return False


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON without exposing a truncated manifest to later runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(path)
    try:
        with open(part, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def list_dir(url: str) -> List[str]:
    """Return href names from an LCZero directory listing."""

    requests = _require_requests()
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return LINK_RE.findall(response.text)


def head_last_modified(url: str) -> Optional[_dt.datetime]:
    """Fetch the Last-Modified timestamp for a remote object."""

    requests = _require_requests()
    try:
        response = requests.head(url, timeout=60)
        response.raise_for_status()
    except Exception:
        return None
    header = response.headers.get("Last-Modified")
    if not header:
        return None
    try:
        dt = parsedate_to_datetime(header)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def choose_recent_files(
    suite_url: str,
    hrefs: Sequence[str],
    limit: int,
    threshold: _dt.datetime,
    head_func: Callable[[str], Optional[_dt.datetime]],
) -> List[FileCandidate]:
    """Filter directory entries down to recent BIN files sorted by recency."""

    threshold = threshold.astimezone(_dt.timezone.utc)
    candidates: List[FileCandidate] = []
    for name in hrefs:
        lowered = name.lower()
        if not lowered.endswith(BIN_SUFFIXES):
            continue
        url = suite_url + name
        last_modified = head_func(url)
        if last_modified is None:
            continue
        last_modified = last_modified.astimezone(_dt.timezone.utc)
        if last_modified < threshold:
            continue
        candidates.append(FileCandidate(name=name, url=url, last_modified=last_modified))
    candidates.sort(key=lambda c: c.last_modified, reverse=True)
    if limit > 0:
        return candidates[:limit]
    return candidates


def plan_suite_downloads(
    suites: Iterable[str],
    out_dir: Path,
    threshold: _dt.datetime,
    limit_per_suite: int,
    list_func: Callable[[str], Sequence[str]] = list_dir,
    head_func: Callable[[str], Optional[_dt.datetime]] = head_last_modified,
    skip_existing: bool = False,
) -> tuple[dict, List[DownloadJob]]:
    """Prepare manifest metadata and download jobs for the requested suites."""

    threshold = threshold.astimezone(_dt.timezone.utc)
    manifest: dict = {
        "base": TRAINING_DATA_BASE,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "threshold": threshold.isoformat(),
        "suites": {},
        "failures": [],
    }
    jobs: List[DownloadJob] = []

    for suite in suites:
        suite = suite if suite.endswith("/") else suite + "/"
        suite_url = TRAINING_DATA_BASE + suite
        try:
            hrefs = list_func(suite_url)
        except Exception as exc:
            manifest["suites"][suite] = {
                "error": str(exc),
                "files": [],
            }
            continue
        def tracked_head(url: str) -> Optional[_dt.datetime]:
            try:
                value = head_func(url)
            except Exception as exc:
                manifest["failures"].append(
                    {"stage": "head", "suite": suite, "url": url, "error": str(exc)}
                )
                return None
            if value is None:
                manifest["failures"].append(
                    {
                        "stage": "head",
                        "suite": suite,
                        "url": url,
                        "error": "missing or unavailable Last-Modified metadata",
                    }
                )
            return value

        candidates = choose_recent_files(
            suite_url,
            hrefs,
            limit=limit_per_suite,
            threshold=threshold,
            head_func=tracked_head,
        )
        suite_entries: List[dict] = []
        for cand in candidates:
            dest = out_dir / suite.strip("/") / cand.name
            status = "queued"
            if skip_existing and is_complete_download(dest):
                status = "exists"
            else:
                jobs.append(DownloadJob(url=cand.url, dest=dest))
            suite_entries.append(
                {
                    "name": cand.name,
                    "url": cand.url,
                    "last_modified": cand.last_modified.isoformat(),
                    "dest": str(dest),
                    "status": status,
                }
            )
        manifest["suites"][suite] = suite_entries

    return manifest, jobs


def download(job: DownloadJob) -> dict:
    """Download a single BIN file and atomically publish it on success."""

    requests = _require_requests()
    job.dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(job.dest)
    start = time.time()
    written = 0
    try:
        with requests.get(
            job.url,
            stream=True,
            timeout=120,
            headers={"Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            raw_expected = getattr(response, "headers", {}).get("Content-Length")
            expected = int(raw_expected) if raw_expected is not None else None
            with open(part, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        if written == 0:
            raise RuntimeError(f"downloaded an empty file from {job.url}")
        if expected is not None and written != expected:
            raise RuntimeError(
                f"downloaded {written} bytes from {job.url}; expected {expected} bytes"
            )
        os.replace(part, job.dest)
    finally:
        part.unlink(missing_ok=True)
    return {
        "url": job.url,
        "path": str(job.dest),
        "size": written,
        "elapsed": time.time() - start,
    }


def maybe_decompress_zst(path: Path, overwrite: bool = False) -> Optional[Path]:
    """Decompress a .zst BIN file if zstandard is available."""

    lowered = path.suffix.lower()
    if lowered not in {".zst", ".zstd"}:
        return None
    out_path = path.with_suffix("")
    if is_complete_download(out_path) and not overwrite:
        return out_path
    try:
        import zstandard as zstd
    except ImportError:
        print("zstandard not installed; skipping decompression for", path, file=sys.stderr)
        return None
    start = time.time()
    part = _part_path(out_path)
    try:
        with open(path, "rb") as src, open(part, "wb") as dst:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if part.stat().st_size == 0:
            raise RuntimeError(f"decompressed output is empty for {path}")
        os.replace(part, out_path)
    finally:
        part.unlink(missing_ok=True)
    print(f"decompressed {path.name} -> {out_path.name} in {time.time() - start:.1f}s")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--limit-per-suite",
        type=int,
        default=20,
        help="Maximum files per suite (0 = unlimited)",
    )
    parser.add_argument(
        "--suites",
        nargs="*",
        default=list(DEFAULT_SUITES),
        help="Suites relative to training_data to scan",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Only download files with Last-Modified within the last N days",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=max(0, args.days))
    manifest, jobs = plan_suite_downloads(
        suites=args.suites,
        out_dir=args.out,
        threshold=threshold,
        limit_per_suite=args.limit_per_suite,
        skip_existing=args.skip_existing,
    )

    dest_index = {}
    for entries in manifest.get("suites", {}).values():
        if isinstance(entries, dict):
            continue
        for entry in entries:
            dest_index[entry["dest"]] = entry

    downloads: List[dict] = []
    decompressed: List[str] = []
    failures: List[dict] = list(manifest.get("failures", []))

    for suite, entries in manifest.get("suites", {}).items():
        if isinstance(entries, dict) and entries.get("error"):
            failures.append(
                {"stage": "list", "suite": suite, "error": str(entries["error"])}
            )

    def decompress_requested(path: Path, entry: Optional[dict] = None) -> None:
        if path.suffix.lower() not in {".zst", ".zstd"}:
            return
        try:
            out_path = maybe_decompress_zst(path)
            if out_path is None:
                raise RuntimeError("zstandard support is unavailable")
            decompressed.append(str(out_path))
        except Exception as exc:
            if entry is not None:
                entry["status"] = f"decompress error: {exc}"
            failures.append(
                {
                    "stage": "decompress",
                    "url": str(path),
                    "error": str(exc),
                }
            )
            print(f"decompression failed for {path}: {exc}", file=sys.stderr)

    if args.decompress:
        for entries in manifest.get("suites", {}).values():
            if isinstance(entries, dict):
                continue
            for entry in entries:
                if entry.get("status") == "exists":
                    decompress_requested(Path(entry["dest"]), entry)

    if jobs:
        args.out.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_job = {executor.submit(download, job): job for job in jobs}
            for future in cf.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    info = future.result()
                    downloads.append(info)
                    entry = dest_index.get(info["path"])
                    if entry:
                        entry["status"] = "downloaded"
                    if args.decompress:
                        decompress_requested(Path(info["path"]), entry)
                except Exception as exc:
                    entry = dest_index.get(str(job.dest))
                    if entry:
                        entry["status"] = f"error: {exc}"
                    failures.append(
                        {"stage": "download", "url": job.url, "error": str(exc)}
                    )
                    print(f"download failed for {job.url}: {exc}", file=sys.stderr)
    else:
        if failures:
            print("No BIN downloads started; see manifest failures.", file=sys.stderr)
        else:
            print("No new BIN files to download within the requested window.")

    manifest["downloaded"] = downloads
    if args.decompress:
        manifest["decompressed"] = decompressed
    manifest["failures"] = failures

    manifest_path = args.manifest or (args.out / "manifest.json")
    write_json_atomic(manifest_path, manifest)
    print(f"wrote manifest {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
