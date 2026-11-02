#!/bin/bash
# Mac-side backup of the w2048 lineage (the Vast box's /workspace is not a
# persistent volume). Every INTERVAL seconds it pulls:
#   - the watcher's manifest and every gate result and log,
#   - the exported nets at multiples of 100 superbatches,
#   - the two newest checkpoints that pass the watcher's check, plus every
#     checkpoint at a multiple of 500 (weights and optimiser state, for resuming),
# then verifies each pulled net and checkpoint quantised.bin against the hash
# recorded in the box's manifest. Nothing on the box is modified.
#
#   nohup scripts/w2048_backup_pull.sh ~/piebot_backups/w2048_20261030 > pull.log 2>&1 &
#
# LINEAGE (default w2048) picks the run, e.g. LINEAGE=w2048t for the threats run:
# /workspace/$LINEAGE, /workspace/${LINEAGE}_nets, /workspace/${LINEAGE}_gates.
set -uo pipefail
DEST=${1:?usage: w2048_backup_pull.sh DEST_DIR}
HOST=${HOST:-root@222.239.174.10}
PORT=${PORT:-40194}
INTERVAL=${INTERVAL:-1800}
MIN_FREE_GB=${MIN_FREE_GB:-8}
L=${LINEAGE:-w2048}
SSH="ssh -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=30"
mkdir -p "$DEST/nets" "$DEST/gates" "$DEST/ckpt"

while true; do
  echo "== $(date -u +%FT%TZ)"
  free_gb=$(df -g "$DEST" | awk 'NR==2 {print $4}')
  if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "only ${free_gb} GB free; skipping this round"
  else
    rsync -a -e "$SSH" "$HOST:/workspace/${L}_nets/manifest.jsonl" "$DEST/nets/" 2>/dev/null
    rsync -a -e "$SSH" "$HOST:/workspace/${L}_gates/" "$DEST/gates/" 2>/dev/null
    rsync -a -e "$SSH" --include='*00.nnue' --exclude='*' "$HOST:/workspace/${L}_nets/" "$DEST/nets/" 2>/dev/null
    want=$($SSH "$HOST" "python3 -c '
import json
recs=[json.loads(l) for l in open(\"/workspace/${L}_nets/manifest.jsonl\") if l.strip()]
good=[r for r in recs if r.get(\"ok\")]
keep={r[\"n\"] for r in good[-2:]} | {r[\"n\"] for r in good if r[\"n\"] % 500 == 0}
print(\" \".join(str(n) for n in sorted(keep)))
'" 2>/dev/null)
    for n in $want; do
      rsync -a -e "$SSH" "$HOST:/workspace/$L/$L-$n" "$DEST/ckpt/" 2>/dev/null
    done
    # Drop pulled checkpoints that are neither among the newest two nor at a multiple of 500.
    for d in "$DEST"/ckpt/"$L"-*; do
      [ -d "$d" ] || continue
      n=${d##*-}
      case " $want " in *" $n "*) ;; *) [ $((n % 500)) -eq 0 ] || rm -rf "$d" ;; esac
    done
    python3 - "$DEST" "$L" <<'PY'
import hashlib, json, sys
from pathlib import Path
dest, lineage = Path(sys.argv[1]), sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
bad = 0
man = dest / "nets/manifest.jsonl"
if not man.exists():
    print("verify: no manifest yet"); sys.exit(0)
for line in man.read_text().splitlines():
    r = json.loads(line)
    net = dest / "nets" / f"{lineage}-{r['n']}.nnue"
    if net.exists() and r.get("nnue_sha256") and sha(net) != r["nnue_sha256"]:
        print("MISMATCH", net); bad += 1
    q = dest / "ckpt" / f"{lineage}-{r['n']}" / "quantised.bin"
    if q.exists() and sha(q) != r["quantised_sha256"]:
        print("MISMATCH", q); bad += 1
print("verify:", "OK" if not bad else f"{bad} mismatches")
PY
  fi
  sleep "$INTERVAL"
done
