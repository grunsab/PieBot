#!/bin/bash
# Copy the deployment kit to the Lichess bot host and run it there in the
# background (it waits for an idle moment, builds and tests on the host, and
# can take 30+ minutes), then follow its log until it finishes.
# Usage: out/lichess_deploy_67081c6/run_deploy.sh [ssh-target]
set -euo pipefail
HOST=${1:-ubuntu@vps-5d9b1da8.vps.ovh.ca}
KIT=$(cd "$(dirname "$0")" && pwd)
REMOTE=/home/ubuntu/lichess-bot/deploy_kits/67081c6
ssh -o ConnectTimeout=15 "$HOST" "mkdir -p $REMOTE"
scp -q "$KIT/source.tar.gz" "$KIT/source_manifest.json" "$KIT/deploy_remote.py" "$HOST:$REMOTE/"
ssh "$HOST" "cd $REMOTE && sha256sum source.tar.gz && nohup /home/ubuntu/lichess-bot/venv/bin/python -u deploy_remote.py > deploy.log 2>&1 < /dev/null & echo started"
until ssh -o ConnectTimeout=15 "$HOST" "grep -q '\"event\": \"done\"' $REMOTE/deploy.log"; do
  ssh -o ConnectTimeout=15 "$HOST" "tail -1 $REMOTE/deploy.log" || true
  sleep 60
done
ssh "$HOST" "tail -3 $REMOTE/deploy.log; cat /home/ubuntu/lichess-bot/deployment_67081c6/result.json"
scp -q "$HOST:/home/ubuntu/lichess-bot/deployment_67081c6/result.json" "$KIT/result.json"
