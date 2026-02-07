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
import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Dict

import requests

BASE = "https://storage.lczero.org/files/"
TRAINING_PGNS = BASE + "training_pgns/"
DEFAULT_SUITES = ["test80/", "test79/", "test78/"]

LINK_RE = re.compile(r'<a href="([^"]+)">')

def list_dir(url: str) -> List[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hrefs = LINK_RE.findall(r.text)
    return hrefs

def head_last_modified(url: str) -> float:
    try:
        h = requests.head(url, timeout=30)
        if 'Last-Modified' in h.headers:
            # parse HTTP-date
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(h.headers['Last-Modified'])
            return dt.timestamp()
    except Exception:
        pass
    return 0.0

def download(url: str, dest: Path) -> Dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
    return {"url": url, "path": str(dest), "size": dest.stat().st_size, "elapsed": time.time()-t0}

def maybe_decompress_zst(path: Path, overwrite: bool=False) -> Optional[Path]:
    if not path.suffix.lower().endswith('zst') and not path.name.endswith('.zstd'):
        return None
    out = path.with_suffix('')  # drop .zst
    if out.exists() and not overwrite:
        return out
    try:
        import zstandard as zstd
    except ImportError:
        print("zstandard not installed; skipping decompression for", path, file=sys.stderr)
        return None
    t0 = time.time()
    with open(path, 'rb') as f_in, open(out, 'wb') as f_out:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(f_in, f_out)
    print(f"decompressed {path.name} -> {out.name} in {time.time()-t0:.1f}s")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True, help='Output directory')
    ap.add_argument('--limit-per-suite', type=int, default=5)
    ap.add_argument('--suites', nargs='*', default=DEFAULT_SUITES, help='Suites to fetch (relative to training_pgns)')
    ap.add_argument('--decompress', action='store_true', help='Decompress .zst to .pgn')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--manifest', type=Path, default=None)
    args = ap.parse_args()

    manifest = {"base": TRAINING_PGNS, "suites": {}, "downloaded": []}
    to_download = []
    for suite in args.suites:
        if not suite.endswith('/'):
            suite += '/'
        url = TRAINING_PGNS + suite
        try:
            hrefs = list_dir(url)
        except Exception as e:
            print(f"list_dir failed for {url}: {e}", file=sys.stderr)
            continue
        # filter .pgn.zst files
        files = [h for h in hrefs if h.endswith('.pgn.zst')]
        # get last-mod times
        metas = []
        for name in files:
            f_url = url + name
            ts = head_last_modified(f_url)
            metas.append((ts, name, f_url))
        metas.sort(reverse=True)
        pick = metas[:args.limit_per_suite]
        manifest["suites"][suite] = [{"name": n, "url": u, "last_modified": ts} for ts, n, u in pick]
        for ts, name, f_url in pick:
            out_path = args.out / suite.strip('/') / name
            if args.skip_existing and out_path.exists():
                print("skip existing", out_path)
                if args.decompress:
                    maybe_decompress_zst(out_path)
                continue
            to_download.append((f_url, out_path))

    # download concurrently
    args.out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(download, url, dest) for url, dest in to_download]
        for fut in cf.as_completed(futs):
            try:
                info = fut.result()
                manifest["downloaded"].append(info)
                print(f"downloaded {info['path']} ({info['size']/1e6:.1f} MB) in {info['elapsed']:.1f}s")
                if args.decompress:
                    maybe_decompress_zst(Path(info['path']))
            except Exception as e:
                print("download failed:", e, file=sys.stderr)

    # write manifest
    man_path = args.manifest or (args.out / 'manifest.json')
    with open(man_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print("wrote manifest", man_path)

if __name__ == '__main__':
    main()
