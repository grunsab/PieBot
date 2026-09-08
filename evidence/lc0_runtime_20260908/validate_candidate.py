"""Fresh Python battery with before/after scope and production-engine checks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRODUCTION = 'b517c47cb86b9f524118df6469b29d7d76495898'
PYTHON = '/private/tmp/piebot-lc0-venv/bin/python'
PROTECTED = ['scripts/' + name + '.py' for name in
             ('uci_elo_arena', 'uci_elo_ladder', 'lc0_anchor_monitor', 'ranked_engine_monitor')]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot():
    files = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '--',
                           'training/nnue', 'scripts', 'deploy', 'PieBot').decode().splitlines()))
    return {name: sha(ROOT / name) for name in files if (ROOT / name).is_file()}


def protected_check():
    assert git('diff', PRODUCTION, '--', 'PieBot') == b'', 'production Rust source differs'
    assert git('ls-files', '--others', '--exclude-standard', '--', 'PieBot') == b''
    result = {}
    for path in PROTECTED:
        if not git('ls-tree', PRODUCTION, '--', path):
            assert not (ROOT / path).exists(), path
            result[path] = None  # Separate ranked checkout; absent in production source.
            continue
        expected = hashlib.sha256(git('show', PRODUCTION + ':' + path)).hexdigest()
        assert sha(ROOT / path) == expected, path
        result[path] = expected
    return result


def run_suite(name, directory):
    command = [PYTHON, '-m', 'unittest', 'discover', '-v', directory]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.monotonic()
    log = OUT / (name + '.log')
    with log.open('wb') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    counts = re.findall(r'^Ran (\d+) tests? in ', text, re.MULTILINE)
    assert counts, 'missing unittest summary: ' + name
    return dict(command=command, tests=int(counts[-1]), exit_code=result.returncode,
                seconds=time.monotonic() - started, log=str(log.relative_to(ROOT)), sha256=sha(log),
                skipped=len(re.findall(r'\.\.\. skipped ', text)),
                ok=bool(re.search(r'^OK(?: \(.*\))?$', text, re.MULTILINE)))


def main():
    before = snapshot()
    protected = protected_check()
    selected = json.loads((OUT / 'selected_sources.json').read_text())
    for path, record in selected['selected_files'].items():
        assert before[path] == record['source_sha256'], path
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(run_suite, name, directory)
                   for name, directory in [('nnue', 'training/nnue/tests'), ('scripts', 'scripts/tests')]}
        runs = {name: future.result() for name, future in futures.items()}
    after = snapshot()
    assert before == after, 'source/test bytes changed during validation'
    assert protected == protected_check()
    record = dict(schema='piebot-lc0-runtime-validation-v1', started_at_utc=started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  candidate_base=git('rev-parse', 'HEAD').decode().strip(),
                  production_base=PRODUCTION, source_origin=selected['selected_source_commit'],
                  python_version=subprocess.check_output([PYTHON, '--version'], text=True).strip(),
                  parallel_independent_processes=True,
                  thread_environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                  fresh_tests=runs, source_and_test_hashes_unchanged=True,
                  source_and_test_sha256=after, protected_script_sha256=protected,
                  production_piebot_tree=git('rev-parse', PRODUCTION + ':PieBot').decode().strip(),
                  entire_production_piebot_tree_unchanged=True,
                  rust_reused_not_rerun=json.loads((OUT / 'rust_reuse.json').read_text()))
    (OUT / 'validation.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({name: {key: value for key, value in run.items() if key != 'command'}
                      for name, run in runs.items()}, indent=2), flush=True)
    if any(run['exit_code'] or not run['ok'] for run in runs.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
