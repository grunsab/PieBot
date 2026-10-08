#!/usr/bin/env python3
"""Read-only SSH capture of portable LC0 model/Adam recovery snapshots.

The receiver owns pool admission, atomic publication and retention. This driver
advances its separate journal only after receiving and confirming exporter exit.
It does not establish strict campaign resume without the complete corpus.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

GIB = 1024 ** 3
MAX_PROBE_BYTES = 256 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_STATE_BYTES = 1024 * 1024
SCHEMA = 'piebot-lc0-backup-monitor-v1'
SNAPSHOT_SCHEMA = 'piebot-lc0-model-snapshot-v1'
SHA = re.compile(r'^[0-9a-f]{64}$')
TERMINAL = {'complete', 'completed', 'deadline-reached'}

# Execute the exact checked bytes, avoiding a second open between hash and exec.
# Keep the real __main__ globals so dataclasses and postponed annotations work.
VERIFIED_HELPER = '''import hashlib, os, sys
helper, expected = sys.argv[1:3]
with open(helper, 'rb') as source:
    payload = source.read(1024 * 1024 + 1)
if len(payload) > 1024 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit('backup helper SHA-256 mismatch')
sys.argv = [helper] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(helper))
globals()['__file__'] = helper
exec(compile(payload, helper, 'exec'), globals())
'''


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupConfig:
    pool: Path
    state_path: Path
    helper_sha256: str
    host: str = 'root@104.8.120.185'
    port: int = 40728
    helper: str = '/workspace/piebot_lc0_backup_tools/lc0_snapshot_export.py'
    campaign_root: str = '/workspace/piebot_lc0_20260907'
    bootstrap_root: str = '/workspace/piebot_campaign_v8'
    corpus_root: str | None = None
    cadence_seconds: float = 86400.
    poll_seconds: float = 60.
    probe_timeout: float = 30.
    transfer_timeout: float = 900.
    max_snapshot_bytes: int = 5 * GIB
    max_pool_bytes: int = 12 * GIB
    min_free_bytes: int = 8 * GIB
    retain: int = 2


def _validate(config):
    if (not isinstance(config.helper_sha256, str) or not SHA.fullmatch(config.helper_sha256)
            or not re.fullmatch(r'[A-Za-z0-9_@.:[\]-]+', config.host)
            or config.host.startswith('-') or not 1 <= config.port <= 65535):
        raise BackupError('invalid SSH destination, port or helper SHA-256')
    for name in ('helper', 'campaign_root', 'bootstrap_root', 'corpus_root'):
        value = getattr(config, name)
        if name == 'corpus_root' and value is None:
            continue
        if not isinstance(value, str) or not value.startswith('/') or '\0' in value:
            raise BackupError(f'{name} must be an absolute remote path')
    for name, maximum in (('cadence_seconds', None), ('poll_seconds', 60),
                          ('probe_timeout', 30), ('transfer_timeout', 900)):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0 or (maximum and value > maximum):
            raise BackupError(f'invalid {name}')
    for name in ('max_snapshot_bytes', 'max_pool_bytes', 'retain'):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BackupError(f'invalid {name}')
    if not isinstance(config.min_free_bytes, int) or config.min_free_bytes < 0:
        raise BackupError('invalid min_free_bytes')
    if Path(config.state_path).resolve().is_relative_to(Path(config.pool).resolve()):
        raise BackupError('monitor state must be outside the receiver-owned pool')


def ssh_command(config, *, probe):
    _validate(config)
    remote = ['python3', '-c', VERIFIED_HELPER, config.helper, config.helper_sha256,
              '--campaign-root', config.campaign_root]
    if probe:
        remote.append('--probe')
    else:
        remote += ['--bootstrap-root', config.bootstrap_root,
                   '--max-bytes', str(config.max_snapshot_bytes)]
        if config.corpus_root is not None:
            remote += ['--corpus-root', config.corpus_root]
    return ['ssh', '-T', '-p', str(config.port), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', config.host, shlex.join(remote)]


class _Transport:
    def __init__(self, command, timeout, popen=None, stop=None):
        self.command, self.timeout = command, timeout
        self.popen = popen or subprocess.Popen
        self.tail = bytearray()
        self.expired = threading.Event()
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.stop = stop

    def __enter__(self):
        try:
            self.process = self.popen(self.command, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      start_new_session=True)
        except OSError as exc:
            raise BackupError(f'could not start SSH: {exc}') from exc
        self.group_pid = None
        pid = getattr(self.process, 'pid', None)
        if pid is not None:
            try:
                if os.getpgid(pid) == pid:
                    self.group_pid = pid
            except ProcessLookupError:
                # Our own Popen always starts a new session, even if the
                # leader exits before this check while a child keeps a pipe.
                if self.popen is subprocess.Popen:
                    self.group_pid = pid
        self.deadline = time.monotonic() + self.timeout

        def drain():
            try:
                while chunk := self.process.stderr.read(8192):
                    self.tail.extend(chunk)
                    del self.tail[:-MAX_STDERR_BYTES]
            except (OSError, ValueError):
                pass

        def timeout():
            self.expired.set()
            self.kill()

        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()
        self.timer = threading.Timer(self.timeout, timeout)
        self.timer.daemon = True
        self.timer.start()
        def watch_stop():
            while not self.done.wait(.05):
                if self.stop.is_set():
                    self.cancelled.set()
                    self.kill()
                    return
        self.canceller = None
        if self.stop is not None:
            self.canceller = threading.Thread(target=watch_stop, daemon=True)
            self.canceller.start()
        return self

    def kill(self):
        if self.group_pid is not None:
            try:
                os.killpg(self.group_pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            self.process.kill()
        except (OSError, ProcessLookupError):
            pass

    def confirm(self):
        try:
            code = self.process.wait(timeout=max(.001, self.deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            self.expired.set()
            self.kill()
            raise BackupError('SSH transport timed out') from exc
        if self.expired.is_set():
            raise BackupError('SSH transport timed out')
        if self.cancelled.is_set():
            raise BackupError('SSH transport cancelled')
        if code != 0:
            self.reader.join(timeout=1)
            detail = bytes(self.tail[-2048:]).decode('utf-8', errors='replace').strip()
            raise BackupError(f'SSH/exporter exited {code}: {detail}')
        return True

    def __exit__(self, kind, value, traceback):
        self.timer.cancel()
        self.done.set()
        if self.canceller is not None:
            self.canceller.join(timeout=.2)
        # SSH/proxy descendants can retain stderr after a successful leader
        # exit. Close the owned group before joining or closing pipe readers.
        self.kill()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        self.reader.join(timeout=2)
        for stream in (self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        if self.expired.is_set():
            raise BackupError('SSH transport timed out')
        if self.cancelled.is_set():
            raise BackupError('SSH transport cancelled')


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _decode(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError) as exc:
        raise BackupError(f'invalid JSON: {exc}') from exc
    if not isinstance(value, dict):
        raise BackupError('JSON must be an object')
    return value


def _validate_summary(summary, *, require_ready=False):
    if require_ready and not isinstance(summary.get('ready'), bool):
        raise BackupError('probe must declare boolean ready')
    if require_ready and summary['ready'] is False:
        return summary
    if (isinstance(summary.get('completed_chunks'), bool)
            or not isinstance(summary.get('completed_chunks'), int)
            or summary['completed_chunks'] < 1):
        raise BackupError('snapshot requires at least one completed chunk')
    for key in ('state_sha256', 'model_identity_key'):
        if not isinstance(summary.get(key), str) or not SHA.fullmatch(summary[key]):
            raise BackupError(f'invalid {key}')
    if summary.get('status') not in {'running', 'paused', 'error', *TERMINAL}:
        raise BackupError('invalid campaign status')
    return summary


def probe(config, *, popen=None, stop=None):
    try:
        with _Transport(ssh_command(config, probe=True), config.probe_timeout, popen, stop) as session:
            raw = session.process.stdout.read(MAX_PROBE_BYTES + 1)
            if len(raw) > MAX_PROBE_BYTES:
                raise BackupError('probe exceeds its bounded JSON size')
            session.confirm()
            return _validate_summary(_decode(raw), require_ready=True)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f'SSH probe failed: {exc}') from exc


def transfer(config, *, receiver=None, popen=None, stop=None):
    _validate(config)
    if receiver is None:
        if __package__:
            from .lc0_snapshot_store import receive_snapshot
        else:
            # A staged standalone monitor must use its pinned sibling, even
            # when the mutable checkout is reachable through PYTHONPATH.
            from lc0_snapshot_store import receive_snapshot
        receiver = receive_snapshot
    try:
        with _Transport(ssh_command(config, probe=False), config.transfer_timeout, popen, stop) as session:
            confirmed = False

            def confirm_source_success():
                nonlocal confirmed
                confirmed = session.confirm()
                return confirmed

            receipt = receiver(session.process.stdout, Path(config.pool),
                               confirm_source_success=confirm_source_success,
                               max_snapshot_bytes=config.max_snapshot_bytes,
                               max_pool_bytes=config.max_pool_bytes,
                               min_free_bytes=config.min_free_bytes, retain=config.retain)
            if confirmed is not True:
                raise BackupError('receiver did not confirm successful exporter exit')
            return receipt
    except Exception as exc:
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f'snapshot transfer failed: {exc}') from exc


def _reject_symlink_components(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise BackupError(f'monitor path must not contain symlinks: {component}')
    return path


def _identity(config):
    identity = dict(host=config.host, port=config.port, helper=config.helper,
                    helper_sha256=config.helper_sha256, campaign_root=config.campaign_root,
                    bootstrap_root=config.bootstrap_root, pool=str(Path(config.pool).resolve()))
    # Named only when set, so journals written before the option existed stay valid.
    if config.corpus_root is not None:
        identity['corpus_root'] = config.corpus_root
    return identity


def _load_state(config):
    path = _reject_symlink_components(config.state_path)
    if not path.exists():
        return {}
    if not path.is_file() or path.stat().st_size > MAX_STATE_BYTES:
        raise BackupError('invalid monitor state file')
    state = _decode(path.read_bytes())
    if state.get('schema') != SCHEMA or state.get('identity') != _identity(config):
        raise BackupError('monitor state identity mismatch')
    if (not isinstance(state.get('last_completed_at'), (int, float))
            or not math.isfinite(state['last_completed_at'])
            or not SHA.fullmatch(str(state.get('last_model_identity_key', '')))
            or not isinstance(state.get('final_snapshot'), bool)):
        raise BackupError('invalid monitor progress')
    return state


def _atomic_state(path, state):
    path = _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.monitor-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _captured_summary(receipt, config):
    if not isinstance(receipt, dict) or not SHA.fullmatch(str(receipt.get('snapshot_id', ''))):
        raise BackupError('receiver returned an invalid snapshot identity')
    path = Path(receipt.get('manifest_path', ''))
    pool = Path(config.pool).resolve()
    if (not path.is_absolute() or not path.resolve().is_relative_to(pool)
            or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES):
        raise BackupError('receiver returned an invalid captured manifest path')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['snapshot_id']:
        raise BackupError('captured manifest checksum mismatch')
    captured = _decode(raw)
    capabilities = captured.get('capabilities')
    if (captured.get('schema') != SNAPSHOT_SCHEMA or not isinstance(capabilities, dict)
            or capabilities.get('strict_campaign_resume') is not False):
        raise BackupError('unsupported recovery snapshot manifest')
    return _validate_summary(captured)


def run_once(config, *, probe_fn=None, transfer_fn=None, now=time.time, stop=None):
    _validate(config)
    state = _load_state(config)
    summary = _validate_summary(probe_fn(config) if probe_fn else probe(config, stop=stop), require_ready=True)
    if not summary['ready']:
        return 'waiting'
    terminal = summary['status'] in TERMINAL
    if terminal and state.get('final_snapshot') and summary['state_sha256'] == state.get('last_state_sha256'):
        return 'complete'
    stamp = now()
    due = (not state or terminal
           or (summary['model_identity_key'] != state['last_model_identity_key']
               and stamp - state['last_completed_at'] >= config.cadence_seconds))
    if not due:
        return 'not-due'
    receipt = transfer_fn(config) if transfer_fn else transfer(config, stop=stop)
    captured = _captured_summary(receipt, config)
    final = captured['status'] in TERMINAL
    updated = dict(schema=SCHEMA, identity=_identity(config), last_completed_at=now(),
                   last_model_identity_key=captured['model_identity_key'],
                   last_state_sha256=captured['state_sha256'],
                   completed_chunks=captured['completed_chunks'], final_snapshot=final,
                   snapshots=int(state.get('snapshots', 0)) + 1,
                   last_receipt=receipt, strict_campaign_resume=False)
    _atomic_state(config.state_path, updated)
    return 'final-captured' if final else 'captured'


@contextmanager
def _monitor_lock(config):
    path = _reject_symlink_components(Path(config.state_path).with_suffix('.lock'))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BackupError('another backup monitor holds the journal lock') from exc
        yield
    finally:
        os.close(descriptor)


def run(config, *, once=False, stop=None, wait=None):
    _validate(config)
    stop = stop or threading.Event()
    wait = wait or stop.wait
    with _monitor_lock(config):
        while not stop.is_set():
            try:
                status = run_once(config, stop=stop)
                print(json.dumps(dict(event='lc0-backup-monitor', status=status)), flush=True)
                if once or status in {'complete', 'final-captured'}:
                    return 0
            except Exception as exc:
                print(json.dumps(dict(event='lc0-backup-error', error=str(exc)[:2048])), file=sys.stderr, flush=True)
                if once:
                    return 1
            wait(config.poll_seconds)
    return 0


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=root / 'out/lc0_model_backups')
    parser.add_argument('--state-path', type=Path, default=root / 'out/lc0_backup_monitor/state.json')
    parser.add_argument('--helper-sha256', required=True)
    for name, default in [('host', 'root@104.8.120.185'), ('helper', '/workspace/piebot_lc0_backup_tools/lc0_snapshot_export.py'),
                          ('campaign-root', '/workspace/piebot_lc0_20260907'), ('bootstrap-root', '/workspace/piebot_campaign_v8')]:
        parser.add_argument('--' + name, default=default)
    parser.add_argument('--corpus-root', default=None,
                        help='Remote root whose data/ holds the corpus, when it is not the campaign')
    parser.add_argument('--port', type=int, default=40728)
    for name, default in [('cadence-seconds', 86400.), ('poll-seconds', 60.),
                          ('probe-timeout', 30.), ('transfer-timeout', 900.)]:
        parser.add_argument('--' + name, type=float, default=default)
    for name, default in [('max-snapshot-bytes', 5 * GIB), ('max-pool-bytes', 12 * GIB),
                          ('min-free-bytes', 8 * GIB), ('retain', 2)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--once', action='store_true')
    args = vars(parser.parse_args(argv))
    once = args.pop('once')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        return run(BackupConfig(**args), once=once, stop=stop)
    except (BackupError, OSError) as exc:
        print(f'LC0 backup monitor refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
