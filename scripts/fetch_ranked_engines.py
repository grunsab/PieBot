#!/usr/bin/env python3
"""Install immutable official comparison-engine packages, separate from training.

Only --qualify executes an engine (one thread, depth 3, at most 50 ms requested).
All package files, including bundled NNUE/runtime files, are hashed on every resume.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


INVENTORY_SCHEMA = "piebot-ranked-engine-inventory-v1"
MANIFEST_SCHEMA = "piebot-ranked-engine-install-v1"
DEFAULT_INVENTORY = Path(__file__).resolve().parent / "config/ranked_engines_20260905.json"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, path: Path) -> None:
    """curl works on the production box and follows GitHub's HTTPS asset redirect."""
    if not url.startswith("https://"):
        raise ValueError("downloads require HTTPS")
    partial = path.with_name(path.name + ".partial")
    try:
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--proto", "=https", "--proto-redir", "=https",
                        "--retry", "3", "--connect-timeout", "20", "--max-time", "900",
                        "--output", str(partial), url], check=True)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or not path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return Path(*path.parts)


def extract_archive(source: Path, dest: Path, kind: str, *,
                    max_bytes: int = 1024 ** 3, max_files: int = 10_000) -> None:
    """Manually extract regular files/directories only; never follow archive links."""
    dest.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    total = 0

    def target(name, size, is_dir, mode):
        nonlocal total
        relative = relative_path(name)
        key = relative.as_posix()
        if key in seen:
            raise ValueError(f"duplicate archive path: {key}")
        seen.add(key)
        total += size
        if len(seen) > max_files or total > max_bytes or size < 0:
            raise ValueError("archive exceeds extraction limit")
        output = dest / relative
        if is_dir:
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError(f"archive file/directory collision: {key}")
        return output, 0o755 if mode & 0o111 else 0o644

    if kind == "tar":
        with tarfile.open(source, "r:*") as archive:
            for item in archive:
                if not (item.isdir() or item.isfile()) or item.issparse():
                    raise ValueError(f"unsupported archive member: {item.name}")
                output, mode = target(item.name, item.size, item.isdir(), item.mode)
                if item.isfile():
                    with archive.extractfile(item) as stream, output.open("xb") as handle:
                        shutil.copyfileobj(stream, handle)
                    output.chmod(mode)
    elif kind == "zip":
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                mode = item.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError(f"unsupported archive member: {item.filename}")
                output, permissions = target(item.filename, item.file_size, item.is_dir(), mode)
                if not item.is_dir():
                    with archive.open(item) as stream, output.open("xb") as handle:
                        shutil.copyfileobj(stream, handle)
                    output.chmod(permissions)
    else:
        raise ValueError(f"unknown archive format: {kind}")


def file_inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic link in installed package: {path}")
        if path.is_file() and path != root / "manifest.json":
            result.append({"path": path.relative_to(root).as_posix(),
                           "sha256": sha256_file(path), "bytes": path.stat().st_size,
                           "mode": stat.S_IMODE(path.stat().st_mode)})
        elif not path.is_file() and not path.is_dir():
            raise ValueError(f"special file in installed package: {path}")
    return result


def verify_existing(root: Path, spec: dict | None = None) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("symbolic link instead of install manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported install manifest")
    if spec is not None and manifest["inventory_identity_sha256"] != canonical_sha256(spec):
        raise ValueError("installed inventory identity differs; use a new destination")
    files = file_inventory(root)
    if files != manifest["files"] or canonical_sha256(files) != manifest["files_identity_sha256"]:
        raise ValueError("installed files differ from manifest; refusing overwrite")
    executable = root / relative_path(manifest["executable_relative"])
    if (str(executable) != manifest["executable"]
            or sha256_file(executable) != manifest["executable_sha256"]
            or not os.access(executable, os.X_OK)):
        raise ValueError("installed executable identity differs from manifest")
    if manifest["cwd"] != str(executable.parent):
        raise ValueError("installed working directory differs from manifest")
    return manifest


def qualify_engine(executable: Path, spec: dict, network_options: dict,
                   *, timeout: float = 10.0) -> dict:
    """Bounded raw UCI handshake/search without requiring python-chess."""
    process = subprocess.Popen([str(executable)], cwd=executable.parent,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    messages: queue.Queue = queue.Queue()

    def reader():
        try:
            for line in process.stdout:
                messages.put(line.rstrip())
        finally:
            messages.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    def send(line):
        process.stdin.write(line + "\n")
        process.stdin.flush()

    def receive_until(predicate):
        deadline = time.monotonic() + timeout
        lines = []
        while len(lines) < 20_000:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("UCI qualification deadline exceeded")
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("UCI qualification deadline exceeded") from exc
            if line is None:
                raise ValueError("engine exited during UCI qualification")
            lines.append(line)
            if predicate(line):
                return lines
        raise ValueError("excessive output during UCI qualification")

    try:
        send("uci")
        handshake = receive_until(lambda line: line == "uciok")
        names = [line.removeprefix("id name ") for line in handshake if line.startswith("id name ")]
        if len(names) != 1 or not re.search(spec["expected_uci_name"], names[0], re.I):
            raise ValueError(f"UCI identity mismatch: {names}")
        options = {}
        for line in handshake:
            match = re.match(r"option name (.+?) type (\w+)\b", line)
            if match:
                options[match[1]] = line
        configured = {}
        if "Threads" in options:
            configured["Threads"] = 1
        elif not spec.get("single_thread_only", False):
            raise ValueError("no Threads option or upstream single-thread declaration")
        for key, value in {"Hash": 16, "Ponder": False, "MultiPV": 1,
                           "UCI_LimitStrength": False, "SyzygyPath": "",
                           "SyzygyProbeLimit": 0}.items():
            if key in options:
                configured[key] = value
        for key, value in network_options.items():
            if key not in options:
                raise ValueError(f"missing network UCI option: {key}")
            configured[key] = value
        for key, value in configured.items():
            encoded = str(value).lower() if isinstance(value, bool) else str(value)
            send(f"setoption name {key} value {encoded}")
        send("isready")
        ready = receive_until(lambda line: line == "readyok")
        if any(re.search(r"\b(error|failed|invalid|unknown)\b", line, re.I) for line in ready):
            raise ValueError(f"engine rejected qualification options: {ready}")
        send("ucinewgame")
        send("position startpos")
        send("go depth 3 movetime 50")
        search = receive_until(lambda line: line.startswith("bestmove "))
        bestmove = search[-1].split()[1]
        legal = {f"{f}2{f}{r}" for f in "abcdefgh" for r in "34"}
        legal.update({"b1a3", "b1c3", "g1f3", "g1h3"})
        if bestmove not in legal:
            raise ValueError(f"illegal start-position bestmove: {bestmove}")
        return {"status": "passed", "name": names[0], "options": configured,
                "single_thread_qualified": True,
                "advertised_options": options, "bestmove": bestmove,
                "search": "go depth 3 movetime 50", "platform": platform.platform(),
                "machine": platform.machine(),
                "verified_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    finally:
        try:
            send("quit")
            process.wait(timeout=1)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        process.stdin.close()
        thread.join(timeout=2)
        process.stdout.close()


def install_engine(spec: dict, dest: Path, *, qualify: bool = False, downloader=None) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", spec["id"]):
        raise ValueError("unsafe engine id")
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / spec["id"]
    if target.is_symlink():
        raise ValueError("symbolic link instead of engine directory")
    if target.exists():
        manifest = verify_existing(target, spec)
        if qualify:
            qualify_engine(Path(manifest["executable"]), spec, manifest["uci_options"])
        return manifest
    downloader = downloader or download
    with tempfile.TemporaryDirectory(prefix=f".{spec['id']}-", dir=dest) as temporary:
        stage = Path(temporary) / "install"
        stage.mkdir()
        artifact = stage / "source.download"
        downloader(spec["url"], artifact)
        if spec.get("sha256") and sha256_file(artifact) != spec["sha256"]:
            raise ValueError("download SHA256 differs from upstream pin")
        if artifact.stat().st_size != spec["bytes"]:
            raise ValueError("download byte count differs from upstream pin")
        package = stage / "package"
        relative = relative_path(spec["executable"])
        if spec["format"] == "binary":
            (package / relative).parent.mkdir(parents=True)
            shutil.copyfile(artifact, package / relative)
        else:
            extract_archive(artifact, package, spec["format"])
        executable = package / relative
        if not executable.is_file() or executable.is_symlink():
            raise ValueError("release package does not contain expected executable")
        executable.chmod(0o755)
        downloader(spec["license_url"], stage / "LICENSE.upstream")
        if not (stage / "LICENSE.upstream").stat().st_size:
            raise ValueError("empty upstream license")
        relative_options = spec.get("uci_options", {})
        for value in relative_options.values():
            if not (package / relative_path(value)).is_file():
                raise ValueError("missing configured NNUE file")
        qualification = None
        if qualify:
            qualification = qualify_engine(executable, spec,
                {key: str(package / value) for key, value in relative_options.items()})
            # Stable paths in stored qualification, independent of staging directory.
            qualification["options"].update(
                {key: str(target / "package" / value) for key, value in relative_options.items()})
        files = file_inventory(stage)
        final_executable = target / "package" / relative
        manifest = {"schema": MANIFEST_SCHEMA,
                    **{key: spec.get(key) for key in ("id", "name", "rating", "rank", "rating_source",
                                                      "rating_snapshot", "rating_threads", "rating_note")},
                    "upstream": dict(spec), "inventory_identity_sha256": canonical_sha256(spec),
                    "executable": str(final_executable), "executable_relative": "package/" + relative.as_posix(),
                    "executable_sha256": sha256_file(executable), "cwd": str(final_executable.parent),
                    "files": files, "files_identity_sha256": canonical_sha256(files),
                    "auxiliary_files": [dict(item, path=str(target / item["path"]),
                                             role="network" if item["path"].endswith(".nnue") else "runtime")
                                        for item in files if item["path"].startswith("package/")
                                        and item["path"] != "package/" + relative.as_posix()],
                    "uci_options": {key: str(target / "package" / value) for key, value in relative_options.items()},
                    "qualification": qualification,
                    "installed_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        with (stage / "manifest.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise ValueError("engine installation appeared concurrently; refusing overwrite")
        os.rename(stage, target)
    return verify_existing(target, spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--engine", action="append", help="repeatable id; default is boundary cohort")
    parser.add_argument("--all", action="store_true", help="include optional top engines and Leorik")
    parser.add_argument("--dry-run", action="store_true", help="print fixed inventory without writes/network")
    parser.add_argument("--qualify", action="store_true", help="run bounded UCI check on this machine")
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unsupported engine inventory")
    specs = {spec["id"]: spec for spec in inventory["engines"]}
    ids = args.engine or (list(specs) if args.all else inventory["default_cohort"])
    selected = [specs[engine_id] for engine_id in dict.fromkeys(ids)]
    if args.dry_run:
        print(json.dumps({"destination": str(args.dest.resolve()), "engines": selected}, indent=2))
        return 0
    args.dest.mkdir(parents=True, exist_ok=True)
    with (args.dest / ".fetch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for spec in selected:
            manifest = install_engine(spec, args.dest, qualify=args.qualify)
            print(json.dumps({"id": spec["id"], "manifest": str(args.dest.resolve() / spec["id"] / "manifest.json"),
                              "executable": manifest["executable"], "sha256": manifest["executable_sha256"],
                              "qualification": "passed" if args.qualify else "not requested"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
