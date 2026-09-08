"""Verify committed LC0 archive outputs before an explicitly requested eviction.

Preparation explicitly opts in to ``evict``; acquisition resume uses ``verify``
to recognize durable converted archives without rewriting source provenance.
The context owns the existing corpus ``prepare.lock`` and callers share this
context instead of nesting a second flock.

Proofs bind immutable provenance and one committed SQLite archive row, not the
changing whole database. All derived bytes are rehashed. Only SQLite's current
DELETE journal mode is supported. Cooperating corpus/raw writers must honor
the preparation lock; this is not a general transactional filesystem.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid


SCHEMA = 'piebot-lc0-raw-eviction-v1'
MAX_METADATA_BYTES = 16 * 1024**2
SHA = re.compile(r'[0-9a-f]{64}\Z')


class ReceiptError(RuntimeError):
    """The requested proof or eviction could not be safely established."""


def _guard(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
            raise ReceiptError(f'{fn.__name__}: {exc}') from exc
    return guarded


def _path(value):
    path = Path(value).expanduser()
    if '..' in path.parts:
        raise ReceiptError('parent traversal is not allowed')
    return path.absolute()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _directory(path):
    """Open every ancestor relative to its descriptor, rejecting symlinks."""
    path = _path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _regular(path):
    path = _path(path)
    with _directory(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReceiptError(f'not a regular file: {path}')
        yield fd
    finally:
        os.close(fd)


def _inspect(path, *, collect=False):
    with _regular(path) as fd:
        before = _signature(os.fstat(fd))
        if collect and before[3] > MAX_METADATA_BYTES:
            raise ReceiptError(f'metadata exceeds bounded read limit: {path}')
        digest, parts, count = hashlib.sha256(), [], 0
        while chunk := os.read(fd, 1024**2):
            count += len(chunk)
            if collect and count > MAX_METADATA_BYTES:
                raise ReceiptError(f'metadata exceeds bounded read limit: {path}')
            digest.update(chunk)
            if collect:
                parts.append(chunk)
        if count != before[3] or _signature(os.fstat(fd)) != before:
            raise ReceiptError(f'file changed while verifying: {path}')
        return dict(sha256=digest.hexdigest(), size=count, signature=before,
                    raw=b''.join(parts) if collect else None)


def _json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ReceiptError(f'duplicate JSON key: {key}')
            obj[key] = value
        return obj
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ReceiptError('metadata must be an object')
    return value


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ReceiptError('expected lowercase SHA-256')
    return value


def _number(value, *, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ReceiptError('invalid nonnegative integer metadata')
    return value


def _fsync_directory(path):
    with _directory(path) as fd:
        os.fsync(fd)


class ArchiveReceiptStore:
    def __init__(self, raw_manifest, corpus_root, *, raw_root):
        self.raw_manifest, self.corpus_root, self.raw_root = map(
            _path, (raw_manifest, corpus_root, raw_root))
        if (self.raw_root == self.corpus_root or self.raw_root in self.corpus_root.parents
                or self.corpus_root in self.raw_root.parents):
            raise ReceiptError('raw and corpus roots must be separate')
        if self.raw_root not in self.raw_manifest.parents:
            raise ReceiptError('raw manifest must be beneath raw root')
        self.receipts = self.corpus_root / 'raw_evictions'
        self._lock_fd = None

    @_guard
    def __enter__(self):
        if self._lock_fd is not None:
            raise ReceiptError('receipt context is already locked')
        with _directory(self.raw_root), _directory(self.corpus_root):
            pass
        # Preparation creates this lock before any committed archives exist.
        path = self.corpus_root / 'prepare.lock'
        with _directory(path.parent) as parent:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ReceiptError('prepare lock is not a regular file')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self._lock_fd = fd
        return self

    def __exit__(self, *_args):
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _require_lock(self):
        if self._lock_fd is None:
            raise ReceiptError('an active preparation-lock context is required')

    def _proof(self, archive_key):
        self._require_lock()
        _sha(archive_key)
        signatures = {}
        def read(path):
            inspected = _inspect(path, collect=True)
            signatures[path] = inspected['signature']
            return _json(inspected['raw']), inspected['sha256']
        manifest, manifest_sha = read(self.raw_manifest)
        if (manifest.get('schema') != 'piebot-lc0-raw-v1' or manifest.get('complete') is not True
                or manifest.get('failures') or not isinstance(manifest.get('files'), list)):
            raise ReceiptError('a complete verified raw inventory is required')
        entries = manifest['files']
        keys = []
        for entry in entries:
            _sha(entry['sha256']); _number(entry['size'], minimum=1)
            if not isinstance(entry.get('url'), str) or not entry['url']:
                raise ReceiptError('archive URL is required')
            if entry.get('status') not in ('downloaded', 'verified'):
                raise ReceiptError('raw inventory has unverified archive status')
            keys.append(_hash([entry['url'], entry['sha256']]))
        if len(keys) != len(set(keys)) or len({entry['dest'] for entry in entries}) != len(entries):
            raise ReceiptError('duplicate archive identity or destination')
        if archive_key not in keys:
            raise ReceiptError('archive is not in the frozen inventory')
        index = keys.index(archive_key)
        entry = entries[index]
        if not Path(entry['dest']).is_absolute():
            raise ReceiptError('raw archive path must be absolute')
        raw = _path(entry['dest'])
        if self.raw_root not in raw.parents or raw.suffix != '.tar' or raw == self.raw_manifest:
            raise ReceiptError('raw archive must be a complete .tar beneath raw root')

        config, config_sha = read(self.corpus_root / 'identity.json')
        corpus_id = _sha(config.get('corpus_id'))
        specification = {key: value for key, value in config.items() if key != 'corpus_id'}
        if config.get('schema') != 'piebot-lc0-corpus-v1' or _hash(specification) != corpus_id:
            raise ReceiptError('corpus identity checksum mismatch')
        sources = [{key: item.get(key) for key in ('url', 'sha256', 'size', 'suite')} for item in entries]
        if config.get('sources') != sources:
            raise ReceiptError('corpus source identity differs from raw manifest')
        chunk_limit = _number(config['chunk_positions'], minimum=1)
        db = self.corpus_root / 'progress.sqlite3'
        with _regular(db) as fd:
            signatures[db] = _signature(os.fstat(fd))
        with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=0)) as conn:
            if conn.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
                raise ReceiptError('eviction requires SQLite DELETE journal mode')
            conn.execute('BEGIN')
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'archives', 'games', 'source_aliases'} <= tables:
                raise ReceiptError('committed corpus database tables are missing')
            row = conn.execute('SELECT directory,metadata FROM archives WHERE archive_key=?',
                               (archive_key,)).fetchone()
            if row is None:
                raise ReceiptError('archive has no committed database row')
            directory, info = _path(row[0]), _json(row[1])
        expected_dir = self.corpus_root / 'archives' / f'{index:06}_{archive_key[:16]}'
        if directory != expected_dir:
            raise ReceiptError('committed archive directory does not match namespace')
        metadata, metadata_sha = read(directory / 'archive.json')
        if metadata != info:
            raise ReceiptError('archive metadata differs from committed database row')
        derived, names = [], {'archive.json'}
        for kind, prefix, stat_key in (('chunks', 'train', 'training_positions'),
                                       ('holdout', 'holdout', 'holdout_positions')):
            if not isinstance(info.get(kind), list):
                raise ReceiptError('archive chunk metadata must be a list')
            positions = 0
            for chunk in info[kind]:
                name = chunk['name']
                if (not isinstance(name, str) or not re.fullmatch(prefix + r'_\d{6}\.jsonl\.gz', name)
                        or name in names):
                    raise ReceiptError('invalid or duplicate derived chunk name')
                names.add(name)
                count = _number(chunk['positions'], minimum=1)
                if count > chunk_limit:
                    raise ReceiptError('derived chunk exceeds configured position limit')
                positions += count
                path = directory / name
                inspected = _inspect(path)
                signatures[path] = inspected['signature']
                if not inspected['size'] or inspected['sha256'] != _sha(chunk['sha256']):
                    raise ReceiptError(f'derived chunk checksum mismatch: {name}')
                derived.append(dict(name=name, kind=kind, size=inspected['size'],
                                    positions=count, sha256=inspected['sha256']))
            if positions != _number(info['stats'][stat_key]):
                raise ReceiptError('archive position totals differ from chunks')
        with _directory(directory) as fd:
            if set(os.listdir(fd)) != names:
                raise ReceiptError('unexpected files in committed archive directory')
        core = dict(schema=SCHEMA, corpus_id=corpus_id, corpus_root=str(self.corpus_root),
                    identity_sha256=config_sha, raw_root=str(self.raw_root),
                    raw_manifest_path=str(self.raw_manifest), raw_manifest_sha256=manifest_sha,
                    archive_key=archive_key, archive_directory=str(directory),
                    archive_json_sha256=metadata_sha,
                    archive_row_sha256=_hash(dict(archive_key=archive_key, directory=str(directory), metadata=info)),
                    raw=dict(path=str(raw), url=entry['url'], size=entry['size'], sha256=entry['sha256']),
                    derived_files=derived)
        proof = dict(core=core, signatures=signatures, directory=directory, raw=raw)
        self._check_unchanged(proof)
        return proof

    def _check_unchanged(self, proof):
        for path, expected in proof['signatures'].items():
            with _regular(path) as fd:
                if _signature(os.fstat(fd)) != expected:
                    raise ReceiptError(f'verified file identity changed: {path}')

    def _receipt(self, proof):
        path = self.receipts / (proof['core']['archive_key'] + '.json')
        try:
            data = _inspect(path, collect=True)
        except FileNotFoundError:
            return path, None
        receipt = _json(data['raw'])
        if {key: value for key, value in receipt.items() if key != 'created_at'} != proof['core']:
            raise ReceiptError('receipt differs from current provenance or committed outputs')
        stamp = datetime.fromisoformat(receipt['created_at'])
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ReceiptError('receipt creation timestamp must be UTC')
        return path, receipt

    def _raw_present(self, path):
        with _directory(path.parent) as fd:
            try:
                info = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
        if not stat.S_ISREG(info.st_mode):
            raise ReceiptError('raw archive is not a regular file')
        return True

    def _sync_committed(self, proof):
        for path, expected in proof['signatures'].items():
            with _regular(path) as fd:
                if _signature(os.fstat(fd)) != expected:
                    raise ReceiptError('verified file changed before fsync')
                os.fsync(fd)
        for directory in (proof['directory'], proof['directory'].parent,
                          self.corpus_root, self.raw_manifest.parent):
            _fsync_directory(directory)

    def _publish_receipt(self, path, receipt):
        with _directory(self.corpus_root) as corpus_fd:
            try:
                os.mkdir('raw_evictions', mode=0o700, dir_fd=corpus_fd)
            except FileExistsError:
                pass
            os.fsync(corpus_fd)
        payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(payload) > MAX_METADATA_BYTES:
            raise ReceiptError('receipt exceeds bounded metadata limit')
        temporary = '.' + path.name + '.tmp-' + uuid.uuid4().hex
        with _directory(self.receipts) as parent:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            try:
                written = 0
                while written < len(payload):
                    count = os.write(fd, payload[written:])
                    if count <= 0:
                        raise ReceiptError('receipt write made no progress')
                    written += count
                os.fsync(fd)
                # Atomic no-overwrite publication; never replace an existing proof.
                os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent,
                        follow_symlinks=False)
                os.unlink(temporary, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(fd)
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def _unlink_raw(self, path, expected):
        with _directory(path.parent) as parent:
            if _signature(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != expected:
                raise ReceiptError('raw file identity changed before unlink')
            os.unlink(path.name, dir_fd=parent)

    @_guard
    def verify(self, archive_key):
        """Return a current verified receipt without creating or deleting files."""
        proof = self._proof(archive_key)
        path, receipt = self._receipt(proof)
        if receipt is None:
            raise ReceiptError('verified eviction receipt is missing')
        self._check_unchanged(proof)
        return dict(receipt=receipt, receipt_path=str(path), raw_present=self._raw_present(proof['raw']))

    @_guard
    def evict(self, archive_key):
        """Explicitly evict one verified raw archive, or resume that eviction."""
        proof = self._proof(archive_key)
        path, receipt = self._receipt(proof)
        present = self._raw_present(proof['raw'])
        if not present:
            if receipt is None:
                raise ReceiptError('missing raw archive has no prior eviction receipt')
            return dict(receipt=receipt, receipt_path=str(path), raw_present=False,
                        raw_deleted=False, warnings=[])
        raw = _inspect(proof['raw'])
        if raw['sha256'] != proof['core']['raw']['sha256'] or raw['size'] != proof['core']['raw']['size']:
            raise ReceiptError('raw archive checksum or size mismatch')
        self._sync_committed(proof)
        if receipt is None:
            receipt = dict(proof['core'], created_at=datetime.now(timezone.utc).isoformat())
            self._publish_receipt(path, receipt)
        # Reverify persisted bytes, including on a retry after a failed unlink.
        _, persisted = self._receipt(proof)
        if persisted != receipt:
            raise ReceiptError('receipt publication verification failed')
        with _regular(path) as fd:
            os.fsync(fd)
        _fsync_directory(path.parent)
        self._check_unchanged(proof)
        self._unlink_raw(proof['raw'], raw['signature'])
        warnings = []
        try:
            _fsync_directory(proof['raw'].parent)
        except OSError as exc:
            # Derived data and proof are already durable. A crash could restore
            # the raw directory entry; the same receipt safely resumes eviction.
            warnings.append(f'raw archive unlinked, but parent fsync failed: {exc}')
        return dict(receipt=receipt, receipt_path=str(path), raw_present=False,
                    raw_deleted=True, warnings=warnings)
