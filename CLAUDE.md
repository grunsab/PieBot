# Claude Code instructions

Follow `AGENTS.md`; it is the contributor and agent guide for this
repository. The essentials:

- Build inside `PieBot/` (`cd PieBot && cargo build --release`) so the native
  CPU flags apply.
- Always pass `UseNNUE=true` and an absolute `NNUEQuantFile`; a missing net
  silently falls back to the PST evaluation.
- Search changes go through `alphabeta_temp.rs`, the matein3 suite (91/91 at
  depths 7-14) and a 1000-game `uci_match.py` confirmation before promotion.
- Stockfish is an evaluation anchor only, never a source of training labels.
- Write the failing test first; run the full test battery before merging.
- Commit only when asked; never rewrite published history.

Archived campaign handoffs: `documents/history/agent_handoffs_2026.md`.
