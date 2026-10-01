import hashlib,json,os,pathlib,subprocess,time
base=pathlib.Path('/home/ubuntu/lichess-bot');state=base/'deployment_lazy_smp_20261001';stage=base/'engines/PieBot-lazy-smp-20261001';cargo=stage/'PieBot';python=str(base/'venv/bin/python');model=str(base/'engines/PieBot/models/lc0_chunk_00034965.nnue');engine=str(cargo/'target/release/uci')
env=dict(os.environ,PATH='/home/ubuntu/.cargo/bin:'+os.environ['PATH'],CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='2')
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:assert str(base/'lichess-bot.py').encode() not in (p/'cmdline').read_bytes().split(b'\0'),'Runner remains active'
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
steps=[
 ('library_tests',['cargo','test','--release','--locked','--lib'],{}),
 ('integration_tests',['cargo','test','--release','--locked','--test','lazy_smp','--test','lichess_smp_blunders','--test','search_core_temp_regressions','--test','uci_moves','--test','uci_stop'],{}),
 ('accept_1t',[str(cargo/'target/release/accept')],{'PIEBOT_TEST_THREADS':'1','PIEBOT_TEST_START_DEPTH':'7','PIEBOT_TEST_MAX_DEPTH':'7'}),
 ('accept_2t',[str(cargo/'target/release/accept')],{'PIEBOT_TEST_THREADS':'2','PIEBOT_TEST_START_DEPTH':'7','PIEBOT_TEST_MAX_DEPTH':'7'}),
 ('all_45',[python,str(state/'verify_cases.py'),'--binary',engine,'--model',model,'--fixture',str(state/'all_45_cases.json'),'--output',str(state/'linux_all_45.ndjson'),'--threads','2','--modes','clock','--repetitions','1'],{}),
 ('six_cases',[python,str(state/'verify_cases.py'),'--binary',engine,'--model',model,'--fixture',str(cargo/'tests/data/lichess_smp_blunders_20261001.json'),'--output',str(state/'linux_six_cases.ndjson'),'--threads','2','--repetitions','3'],{}),
]
results=[]
for name,cmd,extra in steps:
 start=time.time();log=state/f'{name}.log';print(json.dumps({'starting':name}),flush=True)
 with log.open('wb') as f:r=subprocess.run(cmd,cwd=cargo,env=dict(env,**extra),stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(name=name,command=cmd,extra_env=extra,returncode=r.returncode,seconds=time.time()-start,log=str(log)))
 (state/'validation_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results[-1]),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
print(json.dumps({'complete':True,'binary_sha256':hashlib.sha256(pathlib.Path(engine).read_bytes()).hexdigest()}),flush=True)
