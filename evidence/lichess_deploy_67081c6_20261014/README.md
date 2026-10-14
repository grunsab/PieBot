# Lichess bot moved to 67081c6 (in-search repetition draws)

**Deployed 2026-10-14 at 22:19:49 UTC** on the OVH VPS
(`/home/ubuntu/lichess-bot`). The runner was stopped with SIGINT only after
the game in progress (`2zwEfkZB`) had finished, at 22:14:49 UTC, so no game
was interrupted; the bot was down about five minutes. The first game on the
new engine (`Vov01cVS`, rapid) started at 22:21 UTC.

What changed on the bot: only `engine.dir` and `engine.working_dir` in
`config.yml`, from `engines/PieBot-lazy-smp-20261001` (commit `7bd8cb7`) to
`engines/PieBot-67081c6`. The net (`lc0_chunk_00034965.nnue`, sha256
`56434c1b...`), EvalBlend 100, Threads 3, Hash 1024 and the rapid-only
challenge settings are unchanged.

The new engine includes everything on main since `7bd8cb7`: every check is
extended again (S76 guard removed, `9d316d5`), full NNUE skips the
piece-square score (`1047d9d`), and repetitions inside the search are draws
(`0f5654a`, +29.3 Elo over 1000 games; see
`evidence/search_arms/s110_in_search_repetition_20261014/`).

## Checks on the server before the switch

- Source extracted from `git archive 67081c6 PieBot`; all 49 files of
  `source_manifest.json` matched.
- `cargo build --release --locked`; library tests and the repetition,
  game-derived blunder, Lazy SMP, core temp, UCI move and UCI stop tests:
  75 passed, 0 failed.
- matein3 at depth 7: 91/91 on one thread with 7087288 nodes (identical to the
  Mac) and 91/91 on two threads.
- The forced perpetual `6k1/6p1/5p2/7Q/8/1r6/r1q3PP/7K w - - 0 1` scores 0
  at depth 10 through the UCI binary.
- Binary sha256 `341126652c7105316f4c0eafdcbfd963eabdcfa4e3d650260df350be2eb74d07`.

## Rollback

The previous config is at
`/home/ubuntu/lichess-bot/deployment_67081c6/config.before_67081c6.20261014T221947Z.yml`
and the old engine is still in `engines/PieBot-lazy-smp-20261001`. Stop the
runner between games, copy that file back to `config.yml`, and start the
runner again.

## Files

`deploy_remote.py` (run on the host), `run_deploy.sh` (launcher from the Mac),
`source_manifest.json`, the host's `deploy.log` and `result.json`, and the
matein3 logs. The config backup is not copied here because it contains the
bot's API token.
