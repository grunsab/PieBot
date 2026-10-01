import argparse,datetime,fcntl,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
import requests,yaml
BASE=Path('/home/ubuntu/lichess-bot');CONFIG=BASE/'config.yml';RUNNER=BASE/'lichess-bot.py';PYTHON=BASE/'venv/bin/python';STATE=BASE/'deployment_lazy_smp_20261001';STATE.mkdir(exist_ok=True)
def live(pid):
 try:return '\nState:\tZ' not in Path(f'/proc/{pid}/status').read_text()
 except FileNotFoundError:return False
def runners():
 found=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    args=(p/'cmdline').read_bytes().split(b'\0')
    if str(RUNNER).encode() in args and live(int(p.name)):found.append(int(p.name))
   except (PermissionError,FileNotFoundError,ProcessLookupError):pass
 return found
def engines():
 found=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    exe=os.readlink(p/'exe')
    if '/lichess-bot/engines/' in exe and exe.endswith('/uci') and live(int(p.name)):found.append(int(p.name))
   except (PermissionError,FileNotFoundError,ProcessLookupError):pass
 return found
def ongoing(config):
 r=requests.get('https://lichess.org/api/account/playing',headers={'Authorization':'Bearer '+config['token']},timeout=20);r.raise_for_status();return [g['gameId'] for g in r.json().get('nowPlaying',[])]
def stop_when_idle():
 deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  cfg=yaml.safe_load(CONFIG.read_text());active=engines();games=ongoing(cfg)
  if active or games:
   print(json.dumps({'event':'waiting_for_game_to_finish','engine_pids':active,'games':games}),flush=True);time.sleep(10);continue
  pids=runners();assert len(pids)<=1,pids
  for pid in pids:os.kill(pid,signal.SIGINT)
  end=time.monotonic()+45
  while runners() and time.monotonic()<end:time.sleep(.5)
  assert not runners(),'Runner did not exit; refusing forced termination'
  assert not engines(),'Engine remains; refusing configuration change'
  return pids
 raise RuntimeError('No idle window within ten minutes; runner unchanged')
def start():
 assert not runners(),'Refusing duplicate runner'
 cfg=yaml.safe_load(CONFIG.read_text());threads=cfg['engine']['uci_options']['Threads'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');log=BASE/f'nohup.lazy-smp.{threads}t.{stamp}.log'
 with log.open('ab',buffering=0) as f:
  child=subprocess.Popen(['nohup',str(PYTHON),'-u',str(RUNNER),'--config',str(CONFIG)],cwd=BASE,env=dict(os.environ,RAYON_NUM_THREADS=str(threads)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(4);assert child.poll() is None,'Runner exited; inspect '+str(log)
 return {'pid':child.pid,'log':str(log),'threads':threads}
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['mitigate','stop','start','status']);args=ap.parse_args()
with (STATE/'management.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result={'action':args.action,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 if args.action in ('mitigate','stop'):
  result['stopped_pids']=stop_when_idle()
 if args.action=='mitigate':
  text=CONFIG.read_text();cfg=yaml.safe_load(text);assert cfg['engine']['uci_options']['Threads']==2
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=STATE/f'config.before_mitigation.{stamp}.yml';backup.write_text(text);backup.chmod(0o600)
  updated,n=re.subn(r'(?m)^(\s+Threads:\s*)2(\s*(?:#.*)?)$',r'\g<1>1\2',text);assert n==1,n
  new=yaml.safe_load(updated);expected=cfg.copy();expected['engine']=dict(cfg['engine']);expected['engine']['uci_options']=dict(cfg['engine']['uci_options'],Threads=1);assert new==expected
  tmp=CONFIG.with_suffix('.yml.deploy-tmp');tmp.write_text(updated);tmp.chmod(0o600);tmp.replace(CONFIG);result['backup']=str(backup);result['started']=start()
 if args.action=='start':result['started']=start()
 result['running_pids']=runners();result['engine_pids']=engines()
 cfg=yaml.safe_load(CONFIG.read_text());result['engine_configuration']={k:cfg['engine'].get(k) for k in ('dir','name','working_dir','uci_options')};result['config_sha256']=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
 (STATE/f"{args.action}_{int(time.time())}.json").write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
