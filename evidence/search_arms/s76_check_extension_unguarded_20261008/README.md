# S76 (SEE-guarded check extension) removed

2026-10-08. The main search again extends every checking move by one ply.
S76 had restricted the extension to checks whose static exchange value is not
negative, so sacrificial checks were no longer extended.

## Why

After the 2026-10-04 rebuild the box measured candidates with an engine built
from main (S76, S95, persistent Lazy SMP). Against the ranked cohort at 120+1
the score fell by more than half compared with the pre-wipe build `fcf7d65`,
with identical opponents, book, seed, clock and options:

| Net | Build | Score vs Carp / Lambergar / Schoenemann |
| --- | --- | --- |
| LCZero chunk 34965 | `fcf7d65` (pre-wipe) | 174 / 1200 |
| LCZero v2 chunk 1 (first-layer weights differ by 0.3% rms) | main `7bd8cb7` engine | 18 / 300 in the first 100 games per opponent |
| v8 cycle 168 | `fcf7d65` (pre-wipe, from the launch record) | 63 / 1200 |
| v8 cycle 168 | main `7bd8cb7` engine | 42 / 1200 |

The whole loss was draws by repetition; there were no time forfeits or
abnormal terminations. S76 had been promoted on 400 self-play games at
+9.56 Elo, 95% CI [-13.0, +33.1].

## Evidence for the fix

Mechanism (test `alphabeta_extends_a_sacrificial_check`): in
`r5k1/5Npp/8/8/2Q5/8/8/7K w - - 0 1` (Nh6+ Kh8, Qg8+ Rxg8, Nf7#) the guarded
search finds the mate at depth 4, the unguarded search at depth 3.

Outside engines, box, 120+1, net `c27f1052c0f2`, 100 games per opponent on the
same openings and colours as the first 100 games of the existing measurement
(`outside_engines_100g.json`):

| Build | Carp | Lambergar | Schoenemann | Total |
| --- | --- | --- | --- | --- |
| main engine (S76) | 4.5 | 5.0 | 8.5 | 18.0 / 300 |
| main engine without S76 | 8.5 | 9.0 | 14.0 | 31.5 / 300 |
| `fcf7d65` (control) | not run | not run | 16.5 | |

Without S76 minus with S76: +4.5 percentage points of score, paired-opening
bootstrap 95% CI [+1.5, +7.5]. The control build is not distinguishable from
the fix on Schoenemann (+2.5 points, CI [-3.5, +8.5]), so a remaining
contribution from S95 is neither shown nor excluded.

Head-to-head, Mac, 1000 ms, 400 paired games, net `lc0_chunk_00034965` at
blend 75 on both sides (`h2h_1000ms_400g.json`): without S76 scores
137 W / 119 D / 144 L, -6.1 Elo, paired 95% CI [-20.0, +6.9]; depth 15.12
against 15.44 at equal NPS.

matein3 depth 7, one thread: 91/91 solved by both. `accept` 5030782 -> 7040853;
`accept_temp` (stub) 7040853. The box build of the fix gives the same 7040853.

## What this does not show

- 300 games against three engines some 400-500 Elo stronger. The direction is
  consistent across all three; the size is uncertain.
- Nothing here measures the Stockfish 16 ladder, which read about the same
  with and without S76 (2944 before, 2926 after, different nets).
- Self-play A/B cannot see this class of regression: both sides share the
  blind spot. Search arms that change how checks or sacrifices are searched
  need a run against stronger outside engines before promotion.
