import hashlib,json,os,pathlib,subprocess,tarfile,time
base=pathlib.Path('/home/ubuntu/lichess-bot');state=base/'deployment_lazy_smp_20261001';stage=base/'engines/PieBot-lazy-smp-20261001'
# Never compile against a live game. The normal bot has already been stopped.
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   assert str(base/'lichess-bot.py').encode() not in args, 'Live runner remains'
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
assert not stage.exists(), 'Staging directory already exists; inspect before retry'
stage.mkdir()
with tarfile.open(state/'candidate.tar.gz') as t:t.extractall(stage,filter='data')
manifest=json.loads((state/'source_manifest.json').read_text())
assert all(hashlib.sha256((stage/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
(stage/'models').mkdir();(stage/'models/lc0_chunk_00034965.nnue').symlink_to(base/'engines/PieBot/models/lc0_chunk_00034965.nnue')
assert hashlib.sha256((stage/'models/lc0_chunk_00034965.nnue').read_bytes()).hexdigest()=='56434c1bacf5165baaebc6a5a06d5542d69828dc7b1ff09a934fb8263ea917d0'
env=dict(os.environ,PATH='/home/ubuntu/.cargo/bin:'+os.environ['PATH'],CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='2')
commands=[
 ['cargo','build','--release','--locked','--offline','--bin','uci','--bin','accept','--bin','accept_temp'],
 ['cargo','test','--release','--locked','--offline','--lib'],
 ['cargo','test','--release','--locked','--offline','--test','lazy_smp','--test','lichess_smp_blunders','--test','search_core_temp_regressions','--test','uci_moves','--test','uci_stop'],
]
results=[]
for i,cmd in enumerate(commands):
 start=time.time();log=state/f'build_test_{i}.log'
 with log.open('wb') as f:r=subprocess.run(cmd,cwd=stage/'PieBot',env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(command=cmd,returncode=r.returncode,seconds=time.time()-start,log=str(log)))
 (state/'build_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results[-1]),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
print(json.dumps({'done':True,'binary_sha256':hashlib.sha256((stage/'PieBot/target/release/uci').read_bytes()).hexdigest()}),flush=True)
