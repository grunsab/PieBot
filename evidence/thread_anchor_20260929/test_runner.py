import importlib.util, pathlib, unittest, tempfile, json, hashlib
from unittest.mock import patch
p=pathlib.Path(__file__).with_name('run.py')
spec=importlib.util.spec_from_file_location('thread_anchor',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ComparisonTests(unittest.TestCase):
    def records(self, pairs):
        return [dict(game_index=2*i+j,pair_index=i,piebot_score=s) for i,pair in enumerate(pairs) for j,s in enumerate(pair)]
    def test_schedule_balances_both_arms_colors_and_first_player(self):
        plans=m.arena.build_game_plans(m.arena.builtin_opening_fens(),games=100,seed=42)
        schedule=m.schedule(plans)
        self.assertEqual(len(schedule),200)
        for t in (1,8):
            selected=[p for arm,p in schedule if arm==t]
            self.assertEqual(selected,plans)
            self.assertEqual(sum(p.piebot_color=='white' for p in selected),50)
        first=[schedule[i][0] for i in range(0,200,2)]
        self.assertEqual(first.count(1),50)
        self.assertEqual(first.count(8),50)
    def test_identical_pairs_have_zero_paired_interval(self):
        a=self.records([(0,.5),(.5,.5),(1,.5)]*3)
        r=m.compare(a,a,samples=1000,seed=7)
        self.assertEqual(r['elo_8_minus_1'],0)
        self.assertEqual(r['paired_95_ci'],[0,0])
    def test_advantage_and_incomplete_pair_exclusion(self):
        one=self.records([(0,.5)]*4)
        eight=self.records([(1,.5)]*4)
        r=m.compare(one,eight[:-1],samples=100,seed=7)
        self.assertEqual(r['matched_opening_pairs'],3)
        self.assertAlmostEqual(r['elo_8_minus_1'],381.69700377572996)
        self.assertAlmostEqual(r['paired_95_ci'][0],r['elo_8_minus_1'])
    def test_partial_validation_write_is_retried(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)
            (p/'rust_results_default.json').write_text('[{')
            with patch.object(m,'FIX',p):
                counts,ready,records=m.validation_state()
            self.assertFalse(ready)
            self.assertEqual(counts,{'default':0,'all_features':0})
    def test_failed_validation_requires_verified_successful_retry(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)
            binary=p/'test_binary'; binary.write_bytes(b'fixed test executable')
            failure={'path':str(binary),'name':'uci_stop','returncode':101}
            (p/'rust_results_all_features.json').write_text(json.dumps([failure]))
            retry=dict(failure,returncode=0,binary_sha256='incorrect')
            (p/'rust_retries_all_features.json').write_text(json.dumps([retry]))
            with patch.object(m,'FIX',p):
                with self.assertRaises(RuntimeError): m.validation_state()
                retry['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
                (p/'rust_retries_all_features.json').write_text(json.dumps([retry]))
                counts,ready,records=m.validation_state()
            self.assertFalse(ready)
            self.assertEqual(records['all_features'][0]['returncode'],0)
            self.assertEqual(records['all_features'][0]['original_attempt']['returncode'],101)
    def test_saturated_identical_scores_have_unidentified_elo(self):
        a=self.records([(1,1)]*3)
        r=m.compare(a,a,samples=100,seed=7)
        self.assertIsNone(r['elo_8_minus_1'])
        self.assertIsNone(r['paired_95_ci'])

if __name__=='__main__': unittest.main()
