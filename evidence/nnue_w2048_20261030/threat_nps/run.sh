#!/bin/bash
# Threat-input NPS decision gate (run under flock /workspace/pcores.lock).
# One engine on P-core CPU 8 (5.8 GHz), its SMT sibling idle, nothing else on the P-cores.
set -euo pipefail
export PATH=/root/.cargo/bin:$PATH
OUT=/workspace/threat_nps
cd /workspace/piebot_threats/PieBot
taskset -c 16-31 nice -n 10 cargo build --release -j 6 2>&1 | tail -1
cp target/release/uci /workspace/bin/uci_threats
sha256sum /workspace/bin/uci_threats /workspace/bin/uci_w2048 > $OUT/binaries.txt
cd $OUT
latest=$(ls -v /workspace/w2048_nets/w2048-*.nnue | tail -1)
cp "$latest" $OUT/
L=$(basename "$latest" .nnue)
taskset -c 16-31 python3 - "$L" <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/piebot_threats/training/nnue")
import threats, train_v3 as tv3
for base in ["v3pw_bt4_1500", sys.argv[1]]:
    src = Path(f"/workspace/threat_nps/{base}.nnue")
    h = tv3.load_quant(src)["hidden"]
    zero = Path(f"/workspace/threat_nps/{base}_threats_zero.nnue")
    if not zero.exists():
        threats.write_threat_net(src, zero, np.zeros((threats.N_THREAT_FEATURES, h), "i1"))
    rnd = Path(f"/workspace/threat_nps/{base}_threats_rand.nnue")
    if not rnd.exists():
        rng = np.random.default_rng(1000)
        threats.write_threat_net(src, rnd, np.clip(np.rint(rng.normal(0, 2, (threats.N_THREAT_FEATURES, h))), -127, 127).astype("i1"))
PY
run() {  # engine net label
  echo "== $3"
  taskset -c 8 python3 /workspace/piebot_threats/scripts/experiments/uci_nps.py "$1" --opt UseNNUE=true --opt EvalBlend=100 \
    --opt Hash=64 --opt NNUEQuantFile="$2" --nodes 2000000 --rounds 2 2>&1 | grep "knps per position"
}
{
for base in v3pw_bt4_1500 "$L"; do
  run /workspace/bin/uci_w2048 $OUT/$base.nnue "$base plain, old binary (control)"
  run /workspace/bin/uci_threats $OUT/$base.nnue "$base plain, threats binary"
  run /workspace/bin/uci_threats $OUT/${base}_threats_zero.nnue "$base + threats, zero weights"
  run /workspace/bin/uci_threats $OUT/${base}_threats_rand.nnue "$base + threats, random weights (sd 2)"
done
} | tee $OUT/results.txt
echo NPS_DONE >> $OUT/results.txt
