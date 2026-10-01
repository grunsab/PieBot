"""Independent full-strength Stockfish assessment, never training data."""
import argparse,hashlib,json,pathlib,time
import chess,chess.engine
ap=argparse.ArgumentParser();ap.add_argument('--input',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--nodes',type=int,default=2000000);a=ap.parse_args()
root=pathlib.Path.cwd();work=root/'out/lichess_smp_deploy_20261001';cases={}
for file in [work/'all_45_cases.json',root/'PieBot/tests/data/lichess_smp_blunders_20261001.json']:
 for c in json.loads(file.read_text())['cases']:cases[(c['id'],c['ply'])]=c
moves={}
for file in a.input:
 for r in map(json.loads,pathlib.Path(file).read_text().splitlines()):
  if r['type']=='result':moves.setdefault((r['id'],r['ply']),set()).add(r['bestmove'])
output=pathlib.Path(a.out);output.mkdir(exist_ok=True);sf=root/'out/thread_anchor_20260929/bin/stockfish-18'
e=chess.engine.SimpleEngine.popen_uci(str(sf));e.configure({'Threads':1,'Hash':128,'UCI_LimitStrength':False,'SyzygyProbeLimit':0})
try:
 for key,choices in moves.items():
  c=cases[key];b=chess.Board()
  for u in c['history_uci']:b.push_uci(u)
  assert b.fen()==c['fen'];side=b.turn
  target=output/f'{key[0]}_{key[1]}.json'
  prior=json.loads(target.read_text()) if target.exists() else {'game':key[0],'ply':key[1],'fen':b.fen(),'stockfish_sha256':hashlib.sha256(sf.read_bytes()).hexdigest(),'node_budget':a.nodes,'moves':{}}
  assert prior['node_budget']==a.nodes
  def analyze(move=None):
   info=e.analyse(b,chess.engine.Limit(nodes=a.nodes),root_moves=None if move is None else [chess.Move.from_uci(move)],game=object())
   return {'cp_stm':info['score'].pov(side).score(mate_score=100000),'mate':info['score'].pov(side).mate(),'depth':info.get('depth'),'nodes':info.get('nodes'),'pv_uci':[m.uci() for m in info.get('pv',[])[:16]]}
  if 'best' not in prior:prior['best']=analyze()
  # Rejudge the historical move at the identical budget as each replacement.
  for u in sorted(choices|{c['bad_move']}):
   if u not in prior['moves']:
    prior['moves'][u]={'san':b.san(chess.Move.from_uci(u)),**analyze(u)}
  for u,m in prior['moves'].items():m['cp_loss']=max(0,prior['best']['cp_stm']-m['cp_stm'])
  target.write_text(json.dumps(prior,indent=2)+'\n')
  print(json.dumps({'id':key[0],'best':prior['best']['cp_stm'],'original_loss':prior['moves'][c['bad_move']]['cp_loss'],'candidates':{u:prior['moves'][u]['cp_loss'] for u in sorted(choices)}}),flush=True)
finally:e.quit()
