"""Replay recorded UCI clocks/history without modifying any game records."""
import argparse,hashlib,json,os,pathlib,queue,subprocess,threading,time
import chess
ap=argparse.ArgumentParser();ap.add_argument('--binary',required=True);ap.add_argument('--model',required=True);ap.add_argument('--fixture',required=True);ap.add_argument('--output',required=True);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--repetitions',type=int,default=3);ap.add_argument('--modes',default='clock,depth,warm');a=ap.parse_args();a.binary=str(pathlib.Path(a.binary).resolve());a.model=str(pathlib.Path(a.model).resolve())
fixture=json.loads(pathlib.Path(a.fixture).read_text());assert hashlib.sha256(pathlib.Path(a.model).read_bytes()).hexdigest()==fixture['model_sha256']
output=pathlib.Path(a.output);assert not output.exists(), 'Do not overwrite a previous diagnostic'
q=queue.Queue();proc=subprocess.Popen([a.binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS=str(a.threads),PIEBOT_NNUE_QUANT_FILE=a.model))
def reader():
 for line in proc.stdout:q.put(line.rstrip())
 q.put(None)
threading.Thread(target=reader,daemon=True).start()
def send(s):proc.stdin.write(s+'\n');proc.stdin.flush()
def until(prefix,timeout=90):
 lines=[];deadline=time.monotonic()+timeout
 while True:
  line=q.get(timeout=max(.001,deadline-time.monotonic()))
  if line is None:raise RuntimeError('Engine exited: '+str(proc.poll()))
  lines.append(line)
  if line.startswith(prefix):return lines

def ready():send('isready');return until('readyok')
def fresh():send('ucinewgame');ready()
def search(history,go):
 board=chess.Board()
 for move in history:board.push_uci(move)
 send('position startpos'+(' moves '+' '.join(history) if history else ''))
 start=time.monotonic();send(go);lines=until('bestmove');elapsed=time.monotonic()-start
 move=chess.Move.from_uci(lines[-1].split()[1]);assert move in board.legal_moves
 san=board.san(move);child=board.copy();child.push(move);mates=[]
 for reply in list(child.legal_moves):
  child.push(reply)
  if child.is_checkmate():mates.append(reply.uci())
  child.pop()
 infos=[s for s in lines if s.startswith('info depth ')]
 return {'bestmove':move.uci(),'san':san,'go':go,'elapsed':elapsed,'final_info':infos[-1] if infos else None,'mate_in_one_replies':mates}
try:
 send('uci');startup=until('uciok');send(f'setoption name NNUEQuantFile value {a.model}');send('setoption name UseNNUE value true');send('setoption name EvalBlend value 75');send('setoption name Hash value 512');send(f'setoption name Threads value {a.threads}');startup+=ready()
 assert any(s=='option name UseNNUE type check default true' for s in startup) and not any('error' in s.lower() or 'failed' in s.lower() for s in startup), startup
 meta={'type':'metadata','binary':a.binary,'binary_sha256':hashlib.sha256(pathlib.Path(a.binary).read_bytes()).hexdigest(),'model_sha256':fixture['model_sha256'],'threads':a.threads,'hash_mb':512,'blend':75,'startup':startup,'modes':a.modes,'repetitions':a.repetitions}
 with output.open('x',buffering=1) as f:
  f.write(json.dumps(meta)+'\n')
  for mode in a.modes.split(','):
   for case in fixture['cases']:
    root=chess.Board()
    for move in case['history_uci']:root.push_uci(move)
    assert root.fen()==case['fen'],case['id']
    for rep in range(1 if mode=='warm' else a.repetitions):
     fresh();prefix=[]
     if mode=='warm':
      for step in case['prefix_searches']:
       if step['ply']>=case['ply']:continue
       result=search(case['history_uci'][:step['ply']-1],step['go'])
       prefix.append({'ply':step['ply'],**result})
       print(json.dumps({'event':'warm_prefix','id':case['id'],'ply':step['ply'],'san':result['san']}),flush=True)
     result=search(case['history_uci'],f"go depth {case['live_depth']}" if mode=='depth' else case['live_go'])
     record={'type':'result','id':case['id'],'ply':case['ply'],'mode':mode,'repetition':rep+1,'historical_bad_move':case['bad_move'],'avoids_recorded_blunder':result['bestmove']!=case['bad_move'],'prefix':prefix,**result}
     f.write(json.dumps(record)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='prefix'}),flush=True)
finally:
 try:send('quit');proc.wait(timeout=10)
 except Exception:proc.kill();proc.wait()
