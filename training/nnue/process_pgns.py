#!/usr/bin/env python3
"""
Convert PGN files (e.g., LCZero training PGNs) into training JSONL suitable for NNUE training.

Each output line is a JSON object with at least:
  {"fen": "...", "result": 1|0|-1}

Options:
  - Shard output into multiple JSONL files (by number of positions per shard)
  - Sample every K plies to reduce volume
  - Limit games processed
  - Read .pgn or .pgn.zst (requires zstandard)

Usage:
  python training/nnue/process_pgns.py --in-dir data/lc0_pgns/test80 \
    --out data/nnue_jsonl/test80 --shard-size 200000 --sample-every 1 --max-games 0

Dependencies:
  - python-chess (pip install python-chess)
  - zstandard (pip install zstandard) for .zst
  - tqdm (optional) for progress
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional, TextIO


MANIFEST_NAME = ".piebot_pgn_ingest.json"
MANIFEST_VERSION = 1
CONVERSION_SCHEMA_VERSION = 1
STAGING_PREFIX = ".pgn_stage_"

try:
    import tqdm  # type: ignore
    def _tqdm(it, **kw):
        return tqdm.tqdm(it, **kw)
except Exception:
    def _tqdm(it, **kw):
        return it

def open_text(path: Path):
    if path.suffix.lower() in (".zst", ".zstd"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(open(path, 'rb'))
    return open(path, 'rb')

def iter_games_from_pgn(path: Path):
    import chess.pgn
    with open_text(path) as f:
        # Wrap in TextIO
        import io
        if hasattr(f, 'read') and isinstance(f.read(0), (bytes, bytearray)):
            tf = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
        else:
            tf = f
        while True:
            game = chess.pgn.read_game(tf)
            if game is None:
                break
            yield game

def game_result_to_wdl(headers: dict) -> Optional[int]:
    res = headers.get('Result')
    if res == '1-0':
        return 1
    if res == '0-1':
        return -1
    if res == '1/2-1/2':
        return 0
    return None


class ShardWriter:
    """Write JSONL shards without reusing an existing shard name."""

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = max(1, shard_size)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = self._next_shard_index()
        self.in_shard = 0
        self.shard_fp: Optional[TextIO] = None

    def _next_shard_index(self) -> int:
        indices = []
        for path in self.out_dir.glob("shard_*.jsonl"):
            suffix = path.stem.removeprefix("shard_")
            if suffix.isdigit():
                indices.append(int(suffix))
        return max(indices, default=-1) + 1

    def _open_new_shard(self) -> None:
        self.close()
        shard_path = self.out_dir / f"shard_{self.shard_index:06}.jsonl"
        self.shard_fp = shard_path.open('x', encoding='utf-8')
        self.shard_index += 1
        self.in_shard = 0

    def write(self, record: dict) -> None:
        if self.shard_fp is None or self.in_shard >= self.shard_size:
            self._open_new_shard()
        assert self.shard_fp is not None
        self.shard_fp.write(json.dumps(record) + "\n")
        self.in_shard += 1

    def close(self) -> None:
        if self.shard_fp is not None:
            self.shard_fp.flush()
            os.fsync(self.shard_fp.fileno())
            self.shard_fp.close()
            self.shard_fp = None


def _empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "completed": [], "pending": None}


def _write_manifest_atomic(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{MANIFEST_NAME}.", suffix=".tmp", dir=out_dir)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, out_dir / MANIFEST_NAME)
        try:
            directory_fd = os.open(out_dir, os.O_RDONLY)
        except OSError:  # pragma: no cover - directory handles vary by platform
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:  # pragma: no cover - unsupported on Windows
                    pass
            finally:
                os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return _empty_manifest()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot safely resume: invalid PGN manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise RuntimeError(f"cannot safely resume: unsupported PGN manifest {path}")
    if not isinstance(manifest.get("completed"), list):
        raise RuntimeError(f"cannot safely resume: malformed completed list in {path}")
    if manifest.get("pending") is not None and not isinstance(manifest["pending"], dict):
        raise RuntimeError(f"cannot safely resume: malformed pending transaction in {path}")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_shard_path(out_dir: Path, name: object) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise RuntimeError(f"unsafe shard name in {MANIFEST_NAME}: {name!r}")
    stem = Path(name).stem
    if not name.endswith(".jsonl") or not stem.startswith("shard_") or not stem[6:].isdigit():
        raise RuntimeError(f"invalid shard name in {MANIFEST_NAME}: {name!r}")
    return out_dir / name


def _cleanup_staging_dirs(out_dir: Path) -> None:
    for path in out_dir.glob(f"{STAGING_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path)


def _recover_pending_transaction(out_dir: Path, manifest: dict) -> None:
    pending = manifest.get("pending")
    if pending is not None:
        targets = pending.get("target_shards", [])
        if not isinstance(targets, list):
            raise RuntimeError(f"malformed pending shard list in {out_dir / MANIFEST_NAME}")
        for name in targets:
            _safe_shard_path(out_dir, name).unlink(missing_ok=True)
        manifest["pending"] = None
        _write_manifest_atomic(out_dir, manifest)
    _cleanup_staging_dirs(out_dir)


def _conversion_identity(path: Path, options: dict) -> dict:
    source_sha256 = _sha256_file(path)
    payload = {"source_sha256": source_sha256, "options": options}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "id": hashlib.sha256(encoded).hexdigest(),
        "source": str(path.resolve()),
        "source_sha256": source_sha256,
        "source_size": path.stat().st_size,
        "options": options,
    }


def _entry_artifacts_valid(out_dir: Path, entry: dict) -> bool:
    shards = entry.get("shards")
    if not isinstance(shards, list):
        return False
    for shard in shards:
        if not isinstance(shard, dict):
            return False
        try:
            path = _safe_shard_path(out_dir, shard.get("name"))
        except RuntimeError:
            return False
        expected = shard.get("sha256")
        if not isinstance(expected, str) or not path.is_file() or _sha256_file(path) != expected:
            return False
    return True


def _remove_completed_entry(out_dir: Path, manifest: dict, entry: dict) -> None:
    for shard in entry.get("shards", []):
        if isinstance(shard, dict):
            _safe_shard_path(out_dir, shard.get("name")).unlink(missing_ok=True)
    manifest["completed"].remove(entry)
    _write_manifest_atomic(out_dir, manifest)


def _next_output_shard_index(out_dir: Path) -> int:
    indices = []
    for path in out_dir.glob("shard_*.jsonl"):
        suffix = path.stem.removeprefix("shard_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def _move_staged_shard(source: Path, target: Path) -> None:
    os.replace(source, target)


def _absolute_ply(board: object) -> int:
    ply_method = getattr(board, "ply", None)
    if callable(ply_method):
        return int(ply_method())
    fullmove = max(1, int(getattr(board, "fullmove_number", 1)))
    white_to_move = bool(getattr(board, "turn", True))
    return 2 * (fullmove - 1) + (0 if white_to_move else 1)


def process_pgn(
    path: Path,
    out_dir: Path,
    shard_size: int,
    sample_every: int,
    max_games: int = 0,
    *,
    writer: Optional[ShardWriter] = None,
) -> tuple[int, int]:
    """Process one PGN, returning ``(games, positions)`` written."""

    owned_writer = writer is None
    if writer is None:
        writer = ShardWriter(out_dir, shard_size)

    processed_games = 0
    written_positions = 0
    interval = max(1, sample_every)
    try:
        for game in _tqdm(iter_games_from_pgn(path), desc=f"{path.name}"):
            if max_games and processed_games >= max_games:
                break
            result = game_result_to_wdl(game.headers)
            if result is None:
                continue
            # python-chess constructs this from the PGN's SetUp/FEN headers,
            # including the correct side to move and fullmove number.
            board = game.board()
            for source_ply, move in enumerate(game.mainline_moves()):
                if source_ply % interval == 0:
                    writer.write({"fen": board.fen(), "result": result, "ply": _absolute_ply(board)})
                    written_positions += 1
                board.push(move)
            processed_games += 1
    finally:
        if owned_writer:
            writer.close()
    return processed_games, written_positions


def process_paths(
    paths: Iterable[Path],
    out_dir: Path,
    shard_size: int,
    sample_every: int,
    max_games: int = 0,
) -> tuple[int, int]:
    """Convert new source/options identities and atomically record completion."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    _recover_pending_transaction(out_dir, manifest)
    new_games = 0
    new_positions = 0
    accounted_games = 0
    normalized_shard_size = max(1, shard_size)
    normalized_sample_every = max(1, sample_every)

    for path in paths:
        remaining = 0 if max_games == 0 else max_games - accounted_games
        if max_games and remaining <= 0:
            break
        options = {
            "conversion_schema": CONVERSION_SCHEMA_VERSION,
            "game_limit": remaining,
            "sample_every": normalized_sample_every,
            "shard_size": normalized_shard_size,
        }
        identity = _conversion_identity(path, options)
        completed = next(
            (entry for entry in manifest["completed"] if entry.get("id") == identity["id"]),
            None,
        )
        if completed is not None:
            if _entry_artifacts_valid(out_dir, completed):
                accounted_games += int(completed.get("games", 0))
                continue
            _remove_completed_entry(out_dir, manifest, completed)

        staging_dir = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_dir))
        writer = ShardWriter(staging_dir, normalized_shard_size)
        try:
            games, positions = process_pgn(
                path,
                staging_dir,
                normalized_shard_size,
                normalized_sample_every,
                remaining,
                writer=writer,
            )
            writer.close()
            if _sha256_file(path) != identity["source_sha256"]:
                raise RuntimeError(f"source changed while converting: {path}")
            staged_shards = sorted(staging_dir.glob("shard_*.jsonl"))
            next_index = _next_output_shard_index(out_dir)
            target_names = [
                f"shard_{next_index + offset:06}.jsonl"
                for offset in range(len(staged_shards))
            ]
            manifest["pending"] = {
                "id": identity["id"],
                "source": identity["source"],
                "target_shards": target_names,
            }
            _write_manifest_atomic(out_dir, manifest)
            try:
                shard_metadata = []
                for staged, target_name in zip(staged_shards, target_names):
                    target = _safe_shard_path(out_dir, target_name)
                    _move_staged_shard(staged, target)
                    shard_metadata.append(
                        {"name": target_name, "sha256": _sha256_file(target)}
                    )
                completed_entry = dict(identity)
                completed_entry.update(
                    {"games": games, "positions": positions, "shards": shard_metadata}
                )
                completed_manifest = {
                    "version": MANIFEST_VERSION,
                    "completed": [*manifest["completed"], completed_entry],
                    "pending": None,
                }
                _write_manifest_atomic(out_dir, completed_manifest)
                manifest = completed_manifest
            except Exception:
                _recover_pending_transaction(out_dir, manifest)
                raise
        finally:
            writer.close()
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

        accounted_games += games
        new_games += games
        new_positions += positions

    return new_games, new_positions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--glob', default='*.pgn*', help='Glob pattern for PGNs')
    ap.add_argument('--shard-size', type=int, default=200_000)
    ap.add_argument('--sample-every', type=int, default=1)
    ap.add_argument('--max-games', type=int, default=0, help='0 = unlimited')
    args = ap.parse_args()

    paths = sorted(args.in_dir.glob(args.glob))
    if not paths:
        print("No PGN files matched", args.in_dir / args.glob)
        return
    games, positions = process_paths(
        paths, args.out, args.shard_size, args.sample_every, args.max_games
    )
    print(f"Processed {games} games and wrote {positions} positions")

if __name__ == '__main__':
    main()
