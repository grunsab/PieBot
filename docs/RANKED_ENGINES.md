# Ranked engine comparisons

The user requested super-grandmaster strength and performance comparable to a
top-100 chess engine. The supplied screenshot matches the
[CCRL Blitz complete list for September 5, 2026](https://computerchess.org.uk/404/rating_list_all.html).
Its numbers are engine-family ranks: historical versions do not consume additional
ranks. CCRL's time control is equivalent to 2 minutes + 1 second on an Intel
i7-4770K, with ponder off, generic openings up to 12 moves, and up to six-piece
tablebases. The boundary is Lambergar 1.5, rank 100, 3508 ±14 over 1360 games.

CCRL ratings, human Elo, and the campaign's limited Stockfish 16 `UCI_Elo` anchor
are separate scales. Local results against these opponents measure performance
under our matched conditions; they do not establish official CCRL placement.
Stockfish and these engines are evaluation opponents, never training-label sources.

## Fixed opponents

The inventory at `scripts/config/ranked_engines_20260905.json` records exact
upstream release URLs, source-tag commits, licenses, published asset checksums
where available, rating date, and CPU count. These versions remain fixed even if
upstream publishes a newer release or the live rating list changes.

| Role | Version | Family rank | Listed 1-CPU rating | License |
|---|---|---:|---:|---|
| Boundary cohort | [Carp 3.0.1](https://github.com/dede1751/carp/releases/tag/v3.0.1) | 95 | 3524 | GPL-3.0 |
| Boundary cohort | [Lambergar 1.5](https://github.com/jabolcni/Lambergar/releases/tag/1.5) | 100 | 3508 | MIT |
| Boundary cohort | [Schoenemann 0.5.0](https://github.com/Jochengehtab/Schoenemann/releases/tag/v0.5.0) | 101 | 3504 | AGPL-3.0 |
| Future stronger opponent | [PlentyChess 8.0.0](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0) | 3 | 3774 | GPL-3.0 |
| Future stronger opponent | [Stockfish 17.1](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1) | 1 | 3770 | GPL-3.0 |

Stockfish's rank-1 family entry uses eight CPUs and has rating 3785 in the
screenshot. The separate single-CPU row is 3770; use that rating for our
single-thread comparisons. Full-strength SF17.1 does not replace the pinned
SF16 limited-strength ladder.

The optional [Leorik 3.2 release](https://github.com/lithander/Leorik/releases/tag/3.2)
has a Linux ZIP containing executable `Leorik-3.2.1` and a separate NNUE. The
CCRL standard-chess row names 3.2, so version equivalence is unproven. Its
inventory rating and rank are deliberately unset and it is excluded from the
default cohort. Its package requires the recorded working directory, and its
bundled network must stay beside the executable.

## Download and qualification

Use the separate evaluation checkout `/workspace/piebot_ranked_repo`, the engine
root `/workspace/ranked_engines_20260907`, and results root
`/workspace/piebot_ranked_20260907`. Do not modify the active training checkout,
campaign state, source pin, model, or supervisor to provision comparison engines.

Preview the fixed boundary cohort without network requests or writes:

```bash
python3 scripts/fetch_ranked_engines.py \
  --dest /workspace/ranked_engines_20260907 --dry-run
```

On the Linux AVX2 evaluation machine, install and qualify the default three:

```bash
python3 scripts/fetch_ranked_engines.py \
  --dest /workspace/ranked_engines_20260907 --qualify
```

Repeat `--engine ID` to select explicitly. The stronger optional IDs are
`plentychess-8.0.0` and `stockfish-17.1`. `--all` additionally includes the
version-ambiguous Leorik package; it is not needed for boundary comparisons.

The downloader uses HTTPS official release assets, verifies exact byte counts
and upstream hashes, and downloads the license from the pinned source commit.
Carp and SF17.1 do not publish release digests: their first downloaded asset hashes
are recorded in the installation manifest rather than described as upstream
verified. Preserve those manifests with all comparison results.

Archives accept only ordinary files/directories; traversal, links, special
files, duplicate names, and excessive expansion are rejected. Installation is
staged and renamed into place only after success. A failed download, extraction,
or qualification leaves no installed engine directory. A lock prevents two CLI
installers from racing.

`--qualify` checks the reported UCI identity, sets `Threads=1`, requests a
depth-3 search with a 50 ms move budget, requires a legal start-position move,
and enforces an outer timeout. Hash is 16 MiB for this tiny check. It records the
reported name, advertised options, applied settings, platform, and qualification
time. Schoenemann reports `id name Schoenemann` without a version and advertises
Threads with minimum and maximum 1; its release asset digest pins version 0.5.0.
Qualification is a compatibility check, not strength evidence.

Each `<engine-id>/manifest.json` contains absolute executable and working-directory
paths, executable SHA256, upstream identity, and a sorted hash/size/mode inventory
of every installed file. The original archive, downloaded license, separate
networks, and bundled runtime files are all covered. Auxiliary asset paths are
absolute for arena consumption. `verify_existing(path)` and repeated installs
verify every file, including detecting unexpected files and symlinks; changed
packages are refused rather than silently overwritten. A new destination is
required for a different inventory entry.

## Comparison and evidence

Evaluate the preserved accepted baseline, then changed validation-best candidates
on a best-effort daily cadence, with at least 24 hours between measurement start
times. If a battery takes longer, the next changed candidate is due immediately;
an unchanged candidate does not trigger redundant games. The boundary battery
uses 400 games against each of the three fixed opponents, 120+1 seconds, one thread
per engine, 64 MiB hash, the pinned 1279-position opening book, and paired openings
with reversed colors. Each opponent arena plays games sequentially, with at most
three arenas running simultaneously within the reserved CPU allocation so LCZero
training continues with its compute budget intact.

The arena must pin the candidate network and blend, PieBot executable, opponent
executable, all auxiliary network/runtime files, working directory, UCI options,
opening-book checksum, seed, and time control. Resume only under identical
identities. Report win/draw/loss totals, paired uncertainty, termination causes,
and completed game counts. Preserve separate results for each opponent and
candidate. A loss-heavy battery measures the remaining gap; never treat missing
games, timeout-heavy results, or improved training loss as proof of top-100 strength.

Run focused tooling tests before deployment:

```bash
python3 -m unittest -v scripts.tests.test_fetch_ranked_engines
```

The repository's complete Python/Rust test battery remains required before merge.
No search implementation or training objective changes are introduced by this
download/qualification tooling.
