#!/usr/bin/env python3
"""
Fetch recent LCZero self-play PGNs and (optionally) decompress them.

Sources:
  Base: https://storage.lczero.org/files/
  Training PGNs: https://storage.lczero.org/files/training_pgns/
  Suites: test80, test79, test78

Strategy:
  - List suite directories and collect .pgn.zst entries
  - Sort by Last-Modified (via HEAD) and pick latest N per suite
  - Download concurrently (threaded) with streaming
  - Optionally decompress with zstandard to .pgn
  - Write manifest JSON with metadata

Usage:
  python training/nnue/fetch_lc0_pgns.py --out data/lc0_pgns --limit-per-suite 5 --decompress --concurrency 4

Notes:
  - Requires `requests`; for decompression, install `zstandard` (pip install zstandard)
  - This script does not process PGNs into NNUE features; it preps raw PGNs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

BASE = "https://storage.lczero.org/files/"
TRAINING_PGNS = BASE + "training_pgns/"
DEFAULT_SUITES = ["test80/", "test79/", "test78/"]

LINK_RE = re.compile(r'<a href="([^"]+)">')


def _require_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised by import-only tests
        raise RuntimeError("requests is required for remote PGN fetching") from exc
    return requests


def _part_path(path: Path) -> Path:
    """Return the same-directory staging path used for atomic replacement."""

    return Path(f"{path}.part")


def is_complete_download(path: Path) -> bool:
    """Return whether ``path`` is non-empty and has no partial-download marker."""

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
    requests = _require_requests()
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hrefs = LINK_RE.findall(r.text)
    return hrefs


def head_last_modified(url: str) -> float:
    requests = _require_requests()
    try:
        h = requests.head(url, timeout=30)
        h.raise_for_status()
        if "Last-Modified" in h.headers:
            # parse HTTP-date
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(h.headers["Last-Modified"])
            return dt.timestamp()
    except Exception:
        pass
    return 0.0


def download(url: str, dest: Path) -> Dict:
    requests = _require_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    t0 = time.time()
    written = 0
    try:
        with requests.get(
            url,
            stream=True,
            timeout=60,
            headers={"Accept-Encoding": "identity"},
        ) as r:
            r.raise_for_status()
            raw_expected = getattr(r, "headers", {}).get("Content-Length")
            expected = int(raw_expected) if raw_expected is not None else None
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
                f.flush()
                os.fsync(f.fileno())
        if written == 0:
            raise RuntimeError(f"downloaded an empty file from {url}")
        if expected is not None and written != expected:
            raise RuntimeError(
                f"downloaded {written} bytes from {url}; expected {expected} bytes"
            )
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return {
        "url": url,
        "path": str(dest),
        "size": written,
        "elapsed": time.time() - t0,
    }


def maybe_decompress_zst(path: Path, overwrite: bool = False) -> Optional[Path]:
    if path.suffix.lower() not in {".zst", ".zstd"}:
        return None
    out = path.with_suffix("")
    if is_complete_download(out) and not overwrite:
        return out
    try:
        import zstandard as zstd
    except ImportError:
        print("zstandard not installed; skipping decompression for", path, file=sys.stderr)
        return None
    t0 = time.time()
    part = _part_path(out)
    try:
        with open(path, "rb") as f_in, open(part, "wb") as f_out:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(f_in, f_out)
            f_out.flush()
            os.fsync(f_out.fileno())
        if part.stat().st_size == 0:
            raise RuntimeError(f"decompressed output is empty for {path}")
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    print(f"decompressed {path.name} -> {out.name} in {time.time()-t0:.1f}s")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--limit-per-suite", type=int, default=5)
    ap.add_argument(
        "--suites",
        nargs="*",
        default=DEFAULT_SUITES,
        help="Suites to fetch (relative to training_pgns)",
    )
    ap.add_argument(
        "--decompress", action="store_true", help="Decompress .zst to .pgn"
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    manifest = {
        "base": TRAINING_PGNS,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suites": {},
        "downloaded": [],
        "failures": [],
    }
    to_download: list[tuple[str, Path]] = []
    dest_index: dict[str, dict] = {}
    for suite in args.suites:
        if not suite.endswith("/"):
            suite += "/"
        url = TRAINING_PGNS + suite
        try:
            hrefs = list_dir(url)
        except Exception as e:
            print(f"list_dir failed for {url}: {e}", file=sys.stderr)
            manifest["suites"][suite] = {"error": str(e), "files": []}
            manifest["failures"].append(
                {"stage": "list", "suite": suite, "error": str(e)}
            )
            continue
        # filter .pgn.zst files
        files = [h for h in hrefs if h.endswith(".pgn.zst")]
        # get last-mod times
        metas = []
        for name in files:
            f_url = url + name
            ts = head_last_modified(f_url)
            metas.append((ts, name, f_url))
        metas.sort(reverse=True)
        pick = metas if args.limit_per_suite <= 0 else metas[:args.limit_per_suite]
        entries = []
        for ts, name, f_url in pick:
            out_path = args.out / suite.strip("/") / name
            status = "queued"
            if args.skip_existing and is_complete_download(out_path):
                print("skip existing", out_path)
                status = "exists"
            else:
                to_download.append((f_url, out_path))
            entry = {
                "name": name,
                "url": f_url,
                "last_modified": ts,
                "dest": str(out_path),
                "status": status,
            }
            entries.append(entry)
            dest_index[str(out_path)] = entry
        manifest["suites"][suite] = entries

    def decompress_requested(path: Path, entry: dict) -> None:
        try:
            out = maybe_decompress_zst(path)
            if out is None:
                raise RuntimeError("zstandard support is unavailable")
        except Exception as exc:
            entry["status"] = f"decompress error: {exc}"
            manifest["failures"].append(
                {"stage": "decompress", "path": str(path), "error": str(exc)}
            )
            print(f"decompression failed for {path}: {exc}", file=sys.stderr)

    if args.decompress:
        for entries in manifest["suites"].values():
            if isinstance(entries, dict):
                continue
            for entry in entries:
                if entry["status"] == "exists":
                    decompress_requested(Path(entry["dest"]), entry)

    # download concurrently
    args.out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        future_to_job = {
            ex.submit(download, url, dest): (url, dest) for url, dest in to_download
        }
        for fut in cf.as_completed(future_to_job):
            url, dest = future_to_job[fut]
            try:
                info = fut.result()
                manifest["downloaded"].append(info)
                entry = dest_index[info["path"]]
                entry["status"] = "downloaded"
                print(
                    f"downloaded {info['path']} ({info['size']/1e6:.1f} MB) "
                    f"in {info['elapsed']:.1f}s"
                )
                if args.decompress:
                    decompress_requested(Path(info["path"]), entry)
            except Exception as e:
                print("download failed:", e, file=sys.stderr)
                entry = dest_index.get(str(dest))
                if entry is not None:
                    entry["status"] = f"error: {e}"
                manifest["failures"].append(
                    {"stage": "download", "url": url, "error": str(e)}
                )

    # write manifest
    man_path = args.manifest or (args.out / "manifest.json")
    write_json_atomic(man_path, manifest)
    print("wrote manifest", man_path)
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
