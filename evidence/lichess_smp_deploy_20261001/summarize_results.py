import collections,hashlib,json,pathlib,statistics
P=pathlib.Path(__file__).resolve().parent
six=json.loads((P.parents[1]/'PieBot/tests/data/lichess_smp_blunders_20261001.json').read_text())['cases'];major=json.loads((P/'all_45_cases.json').read_text())['cases'];lookup={(c['id'],c['ply']):c for c in major+six}
files=['linux_all_45.ndjson','linux_six_cases.ndjson','linux_one_thread_control.ndjson'];groups={}
for name in files:
 path=P/name
 if not path.exists():continue
 allrows=[json.loads(s) for s in path.read_text().splitlines()];rows=[r for r in allrows if r['type']=='result'];enriched=[]
 for r in rows:
  judge=P/'sf_judgements'/f"{r['id']}_{r['ply']}.json";r=dict(r);r.pop('prefix',None)
  if judge.exists():
   d=json.loads(judge.read_text());m=d['moves'].get(r['bestmove']);orig=d['moves'].get(r['historical_bad_move'])
   if m:r.update(sf_cp=m['cp_stm'],sf_mate=m['mate'],sf_loss=m['cp_loss'],sf_original_loss=orig['cp_loss'],sf_best_cp=d['best']['cp_stm'])
  enriched.append(r)
 groups[name]={'rows':enriched,'total':len(rows),'avoided_exact_move':sum(r['avoids_recorded_blunder'] for r in rows),'mate_in_one':sum(bool(r['mate_in_one_replies']) for r in rows),'judged':sum('sf_loss' in r for r in enriched),'at_least_200cp_loss':sum(r.get('sf_loss',0)>=200 for r in enriched),'mode_counts':dict(collections.Counter(r['mode'] for r in rows)),'binary_sha256':allrows[0]['binary_sha256']}
result={'groups':groups}
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name,g in groups.items():print(name,{k:v for k,v in g.items() if k!='rows'})
if 'linux_six_cases.ndjson' in groups:
 for c in six:
  rows=[r for r in groups['linux_six_cases.ndjson']['rows'] if r['id']==c['id']]
  print(c['id'],[(r['mode'],r['san'],r.get('sf_loss')) for r in rows])
