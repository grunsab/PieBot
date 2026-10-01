import collections,copy,datetime,hashlib,json,math,random,re
from pathlib import Path
import chess
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=json.loads((OUT/'status.json').read_text())
assert status['phase']=='complete' and status['completed']=={'1':100,'8':100}
manifest=json.loads((OUT/'manifest.json').read_text())
assert all(sha(OUT/p)==h for p,h in manifest['files'].items())
assert sha(ROOT/'models/lc0_chunk_00034965.nnue')==manifest['model_sha256']
arms={t:json.loads((OUT/f'arm_{t}t.json').read_text()) for t in (1,8)}
assert arms[1]['plans']==arms[8]['plans']
a,b=[copy.deepcopy(arms[t]['config']) for t in (1,8)]
assert a['piebot_options'].pop('Threads')==1
assert b['piebot_options'].pop('Threads')==8
assert a==b
assert a['stockfish_options']['Threads']==1 and a['stockfish_options']['UCI_Elo']==3190 and a['stockfish_options']['UCI_LimitStrength']
assert (a['initial_s'],a['increment_s'],a['concurrency'])==(60,.5,1)
comparison=json.loads((OUT/'comparison.json').read_text())
assert comparison['status']=='complete'
chronology=[]; summaries={}; pair_scores={}
for t,arm in arms.items():
 games=arm['games']; assert len(games)==100
 assert [g['game_index'] for g in games]==list(range(100))
 assert collections.Counter(g['piebot_color'] for g in games)=={'white':50,'black':50}
 pairs=collections.defaultdict(list)
 for g,p in zip(games,arm['plans']):
  for k in ('game_index','pair_index','opening_id','opening_fen','piebot_color'): assert g[k]==p[k]
  board=chess.Board(g['opening_fen'])
  for uci in g['moves']:
   move=chess.Move.from_uci(uci); assert move in board.legal_moves
   board.push(move)
  assert board.fen()==g['final_fen']
  outcome=board.outcome(claim_draw=True); assert outcome is not None
  score=.5 if outcome.winner is None else float(outcome.winner==(g['piebot_color']=='white'))
  assert score==g['piebot_score']
  assert g['termination']=='chess_'+outcome.termination.name.lower()
  assert g['termination'] in ('chess_checkmate','chess_threefold_repetition')
  assert g['played_plies']==len(g['moves']) and g['plies']==board.ply()
  pairs[g['pair_index']].append(g)
  chronology.append((g['started_at'],g['completed_at'],t,g['game_index']))
 assert sorted(pairs)==list(range(50))
 for gs in pairs.values(): assert len(gs)==2 and {g['piebot_color'] for g in gs}=={'white','black'}
 pair_scores[t]=[sum(g['piebot_score'] for g in pairs[i])/2 for i in range(50)]
 scores=[g['piebot_score'] for g in games]
 wdl=[scores.count(x) for x in (1,.5,0)]; rate=sum(scores)/100
 elo=400*math.log10(rate/(1-rate))
 summary=arm['summary']
 assert wdl==[summary[k] for k in ('wins','draws','losses')]
 assert rate==summary['score_rate'] and math.isclose(elo,summary['elo_difference'])
 summaries[t]={'wins':wdl[0],'draws':wdl[1],'losses':wdl[2],'score_rate':rate,'elo_vs_anchor':elo,'nominal_anchor_estimate':3190+elo,'termination_counts':dict(collections.Counter(g['termination'] for g in games))}
chronology.sort()
for previous,current in zip(chronology,chronology[1:]): assert previous[1]<=current[0]
expected=[]
for i in range(100):
 expected += [(t,i) for t in ((1,8) if (i//2+i%2)%2==0 else (8,1))]
assert [(t,i) for _,_,t,i in chronology]==expected
rng=random.Random(20260929); distribution=[]
for _ in range(10000):
 ids=[rng.randrange(50) for _ in range(50)]
 rates={t:sum(pair_scores[t][i] for i in ids)/50 for t in (1,8)}
 distribution.append(400*math.log10((rates[8]/(1-rates[8]))/(rates[1]/(1-rates[1]))))
distribution.sort()
ci=[distribution[int(9999*p)] for p in (.025,.975)]
delta=summaries[8]['elo_vs_anchor']-summaries[1]['elo_vs_anchor']
assert math.isclose(delta,comparison['comparison']['elo_8_minus_1'],abs_tol=1e-9)
assert all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(ci,comparison['comparison']['paired_95_ci']))
validation=json.loads((OUT/'validation.json').read_text()); validation_summary={}
for config in ('default','all_features'):
 rows=validation[config]
 assert len(rows)==72 and all(r['returncode']==0 for r in rows)
 passed=0
 for r in rows:
  log=Path(r['log']).read_text()
  assert 'FAILED' not in log
  passed+=sum(int(n) for n in re.findall(r'test result: ok\. (\d+) passed;',log))
  if 'original_attempt' in r: assert sha(Path(r['path']))==r['binary_sha256']
 validation_summary[config]={'executables':72,'tests_passed':passed,'retries':[r['name'] for r in rows if 'original_attempt' in r]}
audit={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','games_replayed':200,'matched_opening_pairs':50,'colors_per_setting':{'white':50,'black':50},'sequential_ABBA_schedule_verified':True,'only_thread_option_differs':True,'pinned_binaries_model_openings_verified':True,'all_moves_legal_and_outcomes_reproduced':True,'crashes_timeouts_adjudications':0,'summaries':summaries,'elo_8_minus_1':delta,'joint_paired_bootstrap_95_ci':ci,'bootstrap_samples':10000,'bootstrap_seed':20260929,'elapsed_hours':(datetime.datetime.fromisoformat(status['finished_at'])-datetime.datetime.fromisoformat(status['started_at'])).total_seconds()/3600,'validation':validation_summary,'raw_files_sha256':{p.name:sha(p) for p in [OUT/'arm_1t.json',OUT/'arm_8t.json',OUT/'comparison.json',OUT/'manifest.json']}}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
