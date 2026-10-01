"""Atomically select a fully verified candidate; preserve one-thread rollback."""
import datetime,fcntl,hashlib,json,os,pathlib,re,subprocess
import yaml
base=pathlib.Path('/home/ubuntu/lichess-bot');state=base/'deployment_lazy_smp_20261001';config=base/'config.yml';stage=base/'engines/PieBot-lazy-smp-20261001';binary=stage/'PieBot/target/release/uci'
with (state/'management.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    assert str(base/'lichess-bot.py').encode() not in (p/'cmdline').read_bytes().split(b'\0'),'Runner remains active'
    exe=os.readlink(p/'exe');assert not ('/lichess-bot/engines/' in exe and exe.endswith('/uci')),'Diagnostic engine remains active'
   except (PermissionError,FileNotFoundError,ProcessLookupError):pass
 results=json.loads((state/'validation_results.json').read_text());assert len(results)==6 and all(r['returncode']==0 for r in results),'Validation incomplete'
 manifest=json.loads((state/'source_manifest.json').read_text());assert all(hashlib.sha256((stage/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items()),'Candidate source changed'
 assert (state/'independent_review_passed.json').exists(),'Independent tactical review not recorded'
 reviewed=json.loads((state/'independent_review_passed.json').read_text());sha=hashlib.sha256(binary.read_bytes()).hexdigest();assert reviewed['binary_sha256']==sha
 rows=[json.loads(s) for s in (state/'linux_six_cases.ndjson').read_text().splitlines()];rows=[r for r in rows if r['type']=='result'];assert len(rows)==42 and all(r['avoids_recorded_blunder'] and not r['mate_in_one_replies'] for r in rows)
 text=config.read_text();cfg=yaml.safe_load(text);assert cfg['engine']['uci_options']['Threads']==1
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=state/f'config.rollback_1t.{stamp}.yml';backup.write_text(text);backup.chmod(0o600)
 expected=yaml.safe_load(text);expected['engine']['dir']=str(stage/'PieBot/target/release')+'/';expected['engine']['working_dir']=str(stage);expected['engine']['name']='uci';expected['engine']['uci_options']['Threads']=2
 for key,value in [('dir',expected['engine']['dir']),('working_dir',str(stage))]:
  text,n=re.subn(r'(?m)^(  '+key+r':).*$',lambda m:m[1]+' "'+value+'"',text);assert n==1,(key,n)
 text,n=re.subn(r'(?m)^(\s+Threads:\s*)1(\s*(?:#.*)?)$',r'\g<1>2\2',text);assert n==1
 assert yaml.safe_load(text)==expected,'Unexpected config edit'
 tmp=config.with_suffix('.yml.deploy-tmp');tmp.write_text(text);tmp.chmod(0o600);tmp.replace(config)
 record={'promoted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary':str(binary),'binary_sha256':sha,'source_sha256':manifest['PieBot/src/search/alphabeta.rs'],'model_sha256':hashlib.sha256(pathlib.Path(expected['engine']['uci_options']['NNUEQuantFile']).read_bytes()).hexdigest(),'engine_configuration':{k:expected['engine'][k] for k in ('dir','name','working_dir','uci_options')},'rollback_config':str(backup),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
 (state/'promotion.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
# The separate helper reacquires the same lock and refuses duplicate runners.
subprocess.run([str(base/'venv/bin/python'),str(base/'manage_lazy_smp_deployment.py'),'start'],check=True)
