# Post-deadline measurement battery — runbook

Execute after the 72h run's clean deadline exit (2026-08-07T03:54:03Z). The
supervisor is expected to stay stopped (`exitcodes=0`, `autorestart=unexpected`)
— verify, never assume. Box: `ssh -p 21990 root@192.220.55.116`.

All numbered outputs land in `/workspace/battery_v1/` on the box and are copied
to the Mac. Nothing here mutates the expired run's state.

## 0. Verify wind-down (read-only)

```bash
supervisorctl status piebot_training_72h_self_teacher   # expect EXITED/STOPPED
PIEBOT_STATE=/workspace/piebot_runs/main_72h_self_teacher_repair_v1/autopilot_state.json
jq '{status, last_error, next_cycle, completed: ((.completed_cycles//[])|length)}' "$PIEBOT_STATE"
df -h /workspace          # F6: verified capacity for all retention math
nproc; taskset -pc 1 2>/dev/null || true   # CPU affinity check (46 vs 192)
```

Snapshot off-box (Mac): final `autopilot_state.json`, `cycles/cycle_000094/nnue_quant.nnue`,
`cycles/cycle_000098/nnue_quant.nnue`, all `gate_*.json` from retained cycles.

## 1. Bootstrap-source verification (closes the shared fatal flaw)

Quants are NOT checkpoints (`_DIRECT_CHECKPOINT_FORMATS`, train_torch.py:63-66).

```bash
ls -la /workspace/piebot_runs/main_72h_self_teacher_repair_v1/cycles/cycle_000098/train/checkpoint.json  # (a) probably pruned
ls -la /workspace/piebot_runs/main_72h_self_teacher_repair_v1/bootstrap/                                  # (b) protected artifacts
```

Pre-registered rule: use (a) if it exists, else the protected bootstrap dense
checkpoint (b). The cycle-166 drifted training checkpoint is REJECTED. Record
the chosen path + sha256 in the battery report.
Also read the incumbent blend from live state (F4): `jq '.active_model_blend_percent' "$PIEBOT_STATE"`.

## 2. Provision measurement stack on the box

```bash
cd /workspace/piebot_rust && git fetch origin campaign-v2 && git checkout campaign-v2   # run is over; pin rule no longer binds
cargo build --release --bin uci --bin compare_play --manifest-path PieBot/Cargo.toml
SF=/workspace/piebot_runs/main_48h_20260802T081500Z/elo_anchors/cycle83_sf16_2500_60plus05/bin/stockfish16
sha256sum "$SF"   # must equal 00628bd9c9855c1b7ff93d7f8d51b413586cdc6336fe637f9a14a61531a05aca
```

## 3. Baseline ladder (frozen anchor conditions, first and forever)

Conditions: SF16 sha above, TC 60+0.5, wall cap 900s (script default now),
Threads=1 both sides, clamp rejection active. Rungs 1320,1500 first (±400 rule;
the local Mac signal suggested ~1900-2100 on SF18 — if 1320/1500 both exceed
85%, add 1800,2000 and re-pool):

```bash
/venv/main/bin/python scripts/uci_elo_ladder.py \
  --piebot-command PieBot/target/release/uci \
  --piebot-nnue /workspace/piebot_runs/main_72h_self_teacher_repair_v1/cycles/cycle_000098/nnue_quant.nnue \
  --piebot-blend <blend-from-step-1> \
  --stockfish-command "$SF" \
  --rungs 1320,1500 --games 100 --seed 20260807 \
  --out-dir /workspace/battery_v1/ladder_baseline
```

Also one PST-only rung run (`--piebot-blend 0`) at the nearest sensible rung —
decides INITIAL_ACTIVE_MODEL_BLEND_PERCENT for v2.

## 4. Compute split + per-cycle bytes (F5, D)

```bash
for C in 159 160 161 162 163 164 165 166; do
  D=$(printf '/workspace/piebot_runs/main_72h_self_teacher_repair_v1/cycles/cycle_%06d' "$C")
  S=$(jq -r --argjson c "$C" '(.completed_cycles[]|select(.cycle==$c)|.started_at|floor)' "$PIEBOT_STATE")
  SP=$(stat -c %Y "$D/selfplay_jsonl/.piebot_stage_complete.json" 2>/dev/null || echo 0)
  RL=$(stat -c %Y "$D/jsonl_relabel/.piebot_stage_complete.json" 2>/dev/null || echo 0)
  CK=$(stat -c %Y "$D/train/checkpoint.json" 2>/dev/null || echo 0)
  echo "cycle=$C selfplay=$((SP-S))s relabel=$((RL-SP))s train=$((CK-RL))s"
  du -sb "$D/selfplay_jsonl" "$D/jsonl_relabel" "$D" 2>/dev/null
done
```

Retention math: `retain = min(8, floor(0.6 * verified_free / per_cycle_bytes))`,
`replay = min(6, retain)`.

## 5. Teacher node-cost distribution (sizes RELABEL_MAX_NODES)

Relabel a retained cycle-165 shard at depth 5 uncapped with per-position node
counts (add `--max-records 2000`); take p95 of nodes as the depth-7 cap.
Achieved-depth stamping is already in the branch, so a capped depth-7 pass on
the same shard reports its achieved-depth distribution directly.

## 6. Teacher-agreement tripwire baseline

Freeze the 10k-FEN probe first (from cycle-165/166 shards, stratified by ply
bucket; commit to `evidence/probe_10k.fen`), then:

```bash
/venv/main/bin/python scripts/teacher_agreement.py \
  --piebot-command "PieBot/target/release/uci" \
  --model-a cycles/cycle_000098/nnue_quant.nnue \
  --model-b <latest-checkpoint-quant> \
  --blend <incumbent-blend> --depth 5 \
  --probe evidence/probe_10k.fen --limit 2000 \
  --out /workspace/battery_v1/teacher_agreement_baseline.json
```

Near-total agreement (>97% best-move) confirms the fixed point (expected).

## 7. S1 confirmation (1000 games, box)

If the Mac 400-game screen passed: `compare_play --games 1000 --movetime 150
--noise-plies 12 --noise-topk 5 --threads 1 --paired-openings --parallel-games 12`
with NNUE blend = incumbent on both sides, `--json-out /workspace/battery_v1/s1_confirm.json`.
Promote alphabeta_temp -> alphabeta only on paired-bootstrap 95% LCB > 0
(compute from pair outcomes; autopilot._paired_gate_statistics has the method).

## 8. Decision-tree branches (from the plan §7)

- Ladder baseline sets rung placement and blend for everything after.
- Compute split sizes Pilot C actor budget (raise is cheap when relabel dominates).
- Per-cycle bytes fix retention/replay for the v2 mint.
- p95 node cost fixes RELABEL_MAX_NODES.
- Agreement baseline arms the tripwire threshold (re-run every ~10 cycles in v2).
