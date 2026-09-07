# LCZero capacity audit — September 7, 2026

Read-only observations at 19:51–19:57 UTC. Exact measurements, sources and formulas are in [capacity_audit.json](capacity_audit.json). No running code, source pin, or campaign data changed.

The frozen inventory contains **1,433 archives totaling 259.13 GB**. Its first archive is **40.66 GB**, versus 340 MB for the next largest. File growth measured **2.09 MB/s**, implying about **5.2 hours remaining for that first download** at 19:52 UTC. If all transfers sustain that rate, remaining acquisition takes roughly 34 hours; if later files sustain 100 MB/s, those files take about 36 minutes. These are scenarios, not completion promises. The downloader is sequential and restarts an interrupted object from byte zero.

The first 999 file headers in the large archive span **June 27–July 9**. It mixes older backfill with eligible collection dates, so the sample does not justify omitting it.

The completed 115.52 MB smoke archive produced **540,647 training positions**, 5,069 holdout positions and **59.00 MB of total derived files** in 24.66 seconds. Uniform extrapolation gives approximately **1.21 billion training positions**, **132.34 GB of derived files**, and 13–15 hours of preparation depending on the large archive's eligible content. Other archive distributions can change those estimates. The fourteen-day training deadline starts after acquisition and preparation.

At 19:52 UTC, the box had **471.16 GB free**. Subtracting the remaining raw downloads and projected derived files leaves **81.34 GB**, or **27.65 GB above the 50 GiB reserve**, before training working files. A 7 GB working allowance leaves about 20.65 GB of additional headroom. This fits the measured ratio but is not a worst-case guarantee; a derived/raw ratio above about 0.59 would consume that allowance and reserve. Track actual derived growth before considering extra storage or removing reproducible setup artifacts.

The two smoke chunks contained only 32,768 positions each. Their observed completion intervals were 61 and 125 seconds, including checkpoint loading, validation, writing and export. Each saved checkpoint was **888.6 MB**, plus **335.6 MB of Adam state**. Production chunks cap at 700,000 positions, but each archive creates its own final partial chunk; uniform extrapolation gives about **2,587 chunks per pass**. Frequent reloads and exports can therefore matter substantially. Full production chunk timings are needed before predicting how many passes fit in fourteen days.

Two operational caveats:

- If preparation reaches its disk reserve while writing a derived archive, restart checks free space before removing that archive's unfinished `.part` directory. After a verified supervisor stop, reclaim only that specific uncommitted cache and restore headroom before resuming. Preserve checkpoints, accepted models, SQLite progress, source pins and state.
- The source contains Chess960 starts despite input format 1. A bounded sample found 142 invalid-castling rows among 2,857 inspected records, for example `nnrkqrbb/pppppppp/8/8/8/8/PPPPPPPP/NNRKQRBB w KQkq - 0 1`. Those rows are filtered, while later valid standard-FEN positions from the same games may remain. The current corpus guarantees individually valid standard FENs, not that every originating game used the standard starting position.
