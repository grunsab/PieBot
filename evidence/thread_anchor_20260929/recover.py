"""Finish validation, audit the known flaky UCI retry, then resume the match."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FIX = ROOT / 'out/lazy_smp_fix_20260929'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.recovery.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    state = {'pid': os.getpid(), 'phase': 'recovering_validation', 'games_per_setting': 100,
             'reason': 'Optional-backend UCI stop test passed unchanged in an isolated retry; rerun again after the full battery finishes.'}
    original = json.loads((OUT / 'status.json').read_text())
    write(OUT / 'recovery_previous_status.json', original)
    write(OUT / 'status.json', state)
    print('Recovery is waiting for the complete validation battery; no match games have been played.', flush=True)
    try:
        deadline = time.monotonic() + 3600
        while True:
            results = {}
            for name in ('default', 'all_features'):
                try:
                    results[name] = json.loads((FIX / f'rust_results_{name}.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    results[name] = []
            state['validation'] = {k: len(v) for k, v in results.items()}
            write(OUT / 'status.json', state)
            if all(len(rows) == 72 for rows in results.values()):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Validation did not finish within the recovery window')
            time.sleep(10)
        for name, rows in results.items():
            retries = []
            for r in rows:
                if r['returncode'] == 0:
                    continue
                if name != 'all_features' or r['name'] != 'uci_stop':
                    raise RuntimeError(f'Additional validation failure needs investigation: {name}/{r["name"]}')
                state.update(phase='retrying_validation', retry=r['name'])
                write(OUT / 'status.json', state)
                log = FIX / f'retry_{name}_{r["name"]}_after_battery.log'
                digest = hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
                start = time.monotonic()
                with log.open('w') as output:
                    proc = subprocess.run([r['path'], '--test'], cwd=ROOT / 'PieBot',
                                          stdout=output, stderr=subprocess.STDOUT, timeout=600)
                retry = dict(r, returncode=proc.returncode, log=str(log),
                             binary_sha256=digest, elapsed_s=time.monotonic() - start)
                retries.append(retry)
                write(FIX / f'rust_retries_{name}.json', retries)
                if proc.returncode:
                    raise RuntimeError(f'The unchanged retry failed: {log}')
        write(OUT / 'recovery.json', {'status': 'verified_retries_passed',
              'initial_driver_sha256': hashlib.sha256((OUT / 'run.initial.py').read_bytes()).hexdigest(),
              'resumed_driver_sha256': hashlib.sha256((OUT / 'run.py').read_bytes()).hexdigest(),
              'note': 'Raw validation results remain untouched. Only hash-verified successful reruns can supersede failures. Engine binaries, model, options and game plans are unchanged.'})
        print('Validation complete; resuming the authorized comparison with its pinned inputs.', flush=True)
        os.execv(sys.executable, [sys.executable, '-u', str(OUT / 'run.py')])
    except BaseException as error:
        state.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(OUT / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
