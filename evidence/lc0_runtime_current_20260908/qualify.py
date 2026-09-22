"""Local four-file reconciliation qualification; no production or Git mutations."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = '9c432db72bff8d406765caafc8fe0dfff096fa97'
SOURCE = 'a13ad857067891f33ee391e159531d183ed91af1'
EXPECTED_TREE = '5c6360ddbead51609f71aa70737a57956531ffe2'
FILES = ['training/nnue/features_v2.py', 'training/nnue/lc0_bin.py',
         'training/nnue/tests/test_features_v2_throughput.py',
         'training/nnue/tests/test_lc0_fen_throughput.py']
V = runpy.run_path(str(OUT / 'validate_candidate.py'))
git, sha = V['git'], V['sha']

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def scope_check():
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    assert git('diff', BASE, '--', 'PieBot', 'scripts', 'deploy') == b''
    assert git('ls-files', '--others', '--exclude-standard', '--', 'PieBot', 'scripts', 'deploy') == b''
    assert git('rev-parse', BASE + ':PieBot').decode().strip() == EXPECTED_TREE
    baseline = json.loads((OUT / 'source_pins_base.json').read_text())['files']
    current = V['snapshot']()
    changed = sorted(p for p in set(baseline) | set(current) if baseline.get(p) != current.get(p))
    assert changed == sorted(FILES), changed
    for p in FILES:
        assert (ROOT / p).read_bytes() == git('show', SOURCE + ':' + p), p
    return current

started = datetime.now(timezone.utc).isoformat()
before = scope_check()
protected = V['protected_check']()
write('source_pins_before.json', before)
selected = {'selected_source_commit': {'current_runtime_base': BASE, 'reviewed_encoder_source': SOURCE},
            'selected_files': {p: {'source_commit': SOURCE, 'source_sha256': before[p]} for p in FILES}}
write('selected_sources.json', selected)
rust = json.loads((ROOT / 'evidence/lc0_runtime_20260908/rust_reuse.json').read_text())
for run in rust['rust_runs'].values():
    log = ROOT / run['log']
    assert sha(log) == run['log_sha256']
    summaries = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', log.read_text())
    assert sum(int(x[0]) for x in summaries) == run['passed']
    assert all(int(x[1]) == int(x[2]) == 0 for x in summaries)
    assert len(re.findall(r'^Success$', log.read_text(), re.M)) == run['criterion_test_mode_successes']
rust['whole_piebot_tree_objects'][BASE] = EXPECTED_TREE
write('rust_reuse.json', rust)
runs = {}
for name, directory in [('nnue', 'training/nnue/tests'), ('scripts', 'scripts/tests')]:
    print('Starting ' + name, flush=True)
    runs[name] = V['run_suite'](name, directory)
    print(json.dumps(runs[name], sort_keys=True), flush=True)
    assert runs[name]['exit_code'] == 0 and runs[name]['ok'] and runs[name]['skipped'] == 0

gold = subprocess.check_output([V['PYTHON'], '-m', 'training.nnue.features_v2'], cwd=ROOT)
assert gold == (ROOT / 'PieBot/tests/data/halfkp_dp_fixture.json').read_bytes()
(OUT / 'features_v2_generated.json').write_bytes(gold)
after = scope_check()
assert after == before
assert protected == V['protected_check']()
write('source_pins_after.json', after)
write('validation.json', {
    'schema': 'piebot-lc0-runtime-current-validation-v1',
    'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    'runtime_base': BASE, 'reviewed_encoder_source': SOURCE,
    'source_scope': FILES, 'source_and_test_hashes_unchanged': True,
    'fresh_tests': runs, 'sequential_suites': True,
    'python_version': subprocess.check_output([V['PYTHON'], '--version'], text=True).strip(),
    'thread_environment': {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    'entire_piebot_scripts_deploy_equal_current_runtime': True,
    'production_piebot_tree': EXPECTED_TREE, 'protected_script_sha256': protected,
    'parallel_download_concurrency_default': 4,
    'parallel_launcher_sha256': sha(ROOT / 'scripts/run_vast_lc0.sh'),
    'production_config_sha256': sha(ROOT / 'deploy/vast/piebot_lc0.conf'),
    'features_gold': {'path': 'PieBot/tests/data/halfkp_dp_fixture.json', 'sha256': hashlib.sha256(gold).hexdigest(), 'exact_bytes': True},
    'rust_reused_not_rerun': rust, 'remote_actions': False, 'committed_by_qualification': False,
})
print('QUALIFICATION COMPLETE; all child test processes exited.', flush=True)
