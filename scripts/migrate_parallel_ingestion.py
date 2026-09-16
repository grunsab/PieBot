#!/usr/bin/env python3
"""Safely migrate PieBot LCZero training pipeline to parallel ingestion and caching.

Performs:
1. Preflight validation of current process, state, and supervisor status.
2. Backups of lc0_state.json and source_git_commit.
3. Clean shutdown at chunk boundary via supervisorctl stop piebot_lc0.
4. Verification that all trainer processes have fully exited.
5. Checkout of target branch campaign/lc0-parallel-ingestion in /workspace/piebot_lc0_repo.
6. Verification and computation of new source identity digest.
7. Atomic update of lc0_state.json (identity.source.commit and identity.source.digest).
8. Atomic update of /workspace/piebot_lc0_20260907/source_git_commit.
9. Clean restart via supervisorctl start piebot_lc0.
10. Verification of running status, absence of identity errors, and training resumption.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

OPS_DIR = Path("/workspace/piebot_ops/parallel_ingestion_20260916")
STATE_PATH = Path("/workspace/piebot_lc0_20260907/training/lc0_state.json")
PIN_PATH = Path("/workspace/piebot_lc0_20260907/source_git_commit")
REPO_PATH = Path("/workspace/piebot_lc0_repo")
LOG_PATH = Path("/workspace/piebot_lc0_supervisor.log")
ERR_PATH = Path("/workspace/piebot_lc0_supervisor.err")

TARGET_BRANCH = "campaign/lc0-parallel-ingestion"
EXPECTED_SCHEMA = "piebot-lc0-autopilot-v1"


def log(msg: str) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    print(f"[{now_str}] {msg}", flush=True)


def run_cmd(args: list[str], check: bool = True, timeout: float = 60.0, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=timeout, cwd=cwd)


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def preflight() -> tuple[dict, str]:
    log("Running preflight checks...")
    OPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check supervisor status
    res = run_cmd(["supervisorctl", "status", "piebot_lc0"], check=False)
    if "RUNNING" not in res.stdout and "STOPPED" not in res.stdout:
        raise RuntimeError(f"Unexpected supervisor status for piebot_lc0: {res.stdout.strip()}")
    log(f"Supervisor status: {res.stdout.strip()}")

    # Check state file
    if not STATE_PATH.is_file():
        raise RuntimeError(f"State file missing: {STATE_PATH}")
    state = json.loads(STATE_PATH.read_text())
    if state.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError(f"State schema mismatch: {state.get('schema')}")
    log(f"Current state: completed_chunks={state.get('completed_chunks')}, pass={state.get('pass_number')}, status={state.get('status')}")

    # Check repo has target branch/commit
    rev_res = run_cmd(["git", "rev-parse", TARGET_BRANCH], check=False, cwd=REPO_PATH)
    if rev_res.returncode != 0:
        raise RuntimeError(f"Target branch {TARGET_BRANCH} not found in {REPO_PATH}")
    target_commit = rev_res.stdout.strip()
    log(f"Target commit: {target_commit}")

    # Check source pin
    if not PIN_PATH.is_file():
        raise RuntimeError(f"Source pin missing: {PIN_PATH}")
    current_pin = PIN_PATH.read_text().strip()
    log(f"Current source pin: {current_pin}")

    # Backups
    state_bak = OPS_DIR / "lc0_state.json.pre_migration.bak"
    pin_bak = OPS_DIR / "source_git_commit.pre_migration.bak"
    if not state_bak.exists():
        atomic_write(state_bak, STATE_PATH.read_bytes(), mode=0o644)
    if not pin_bak.exists():
        atomic_write(pin_bak, PIN_PATH.read_bytes(), mode=0o644)
    log(f"Preflight passed. Backups preserved in {OPS_DIR}")
    return state, target_commit


def stop_piebot() -> dict:
    status_res = run_cmd(["supervisorctl", "status", "piebot_lc0"], check=False)
    if "STOPPED" not in status_res.stdout:
        log("Requesting clean stop of piebot_lc0 at chunk boundary (waiting up to 360s)...")
        res = run_cmd(["supervisorctl", "stop", "piebot_lc0"], timeout=400.0)
        log(f"supervisorctl stop result: {res.stdout.strip()}")

    # Poll to confirm no lingering deploy or autopilot processes
    deadline = time.time() + 60.0
    while time.time() < deadline:
        pgrep = run_cmd(["pgrep", "-af", "training\\.nnue\\.lc0_(deploy|autopilot)"], check=False)
        if not pgrep.stdout.strip():
            break
        log(f"Waiting for lingering trainer processes to exit: {pgrep.stdout.strip()}")
        time.sleep(2.0)
    else:
        raise RuntimeError("Trainer processes did not exit cleanly!")

    # Verify stopped status
    status_res = run_cmd(["supervisorctl", "status", "piebot_lc0"], check=False)
    if "STOPPED" not in status_res.stdout:
        raise RuntimeError(f"piebot_lc0 is not STOPPED: {status_res.stdout.strip()}")

    # Read and inspect stopped state
    state = json.loads(STATE_PATH.read_text())
    log(f"Stopped state: status={state.get('status')}, completed_chunks={state.get('completed_chunks')}, in_progress={state.get('in_progress')}")
    return state


def update_source_and_state(state: dict, target_commit: str) -> dict:
    log(f"Checking out target commit {target_commit} in {REPO_PATH}...")
    run_cmd(["git", "checkout", target_commit], cwd=REPO_PATH)
    status_res = run_cmd(["git", "status"], cwd=REPO_PATH)
    log(f"Git status:\n{status_res.stdout.strip()}")

    # Verify source identity using repository module
    sys.path.insert(0, str(REPO_PATH))
    from training.nnue.lc0_autopilot import _source_identity
    source_id = _source_identity(target_commit)
    log(f"Computed source identity: {source_id}")
    if source_id["commit"] != target_commit:
        raise RuntimeError(f"Commit mismatch: {source_id['commit']} != {target_commit}")

    # Update state
    log(f"Updating lc0_state.json source commit and digest...")
    state["identity"]["source"]["commit"] = target_commit
    state["identity"]["source"]["digest"] = source_id["digest"]
    state["status"] = "paused"
    state["in_progress"] = None

    state_bytes = json.dumps(state, indent=2).encode("utf-8")
    atomic_write(STATE_PATH, state_bytes, mode=0o664)
    log("Updated lc0_state.json atomically.")

    # Update source pin
    log(f"Updating source_git_commit to {target_commit}...")
    atomic_write(PIN_PATH, f"{target_commit}\n".encode("utf-8"), mode=0o644)
    log("Updated source_git_commit atomically.")
    return source_id


def restart_and_verify(old_chunks: int, target_commit: str, source_id: dict) -> dict:
    log("Starting piebot_lc0 with parallel ingestion and caching...")
    start_res = run_cmd(["supervisorctl", "start", "piebot_lc0"])
    log(f"start result: {start_res.stdout.strip()}")

    log("Waiting 15 seconds for startup and preflight...")
    time.sleep(15.0)

    # Check supervisor status
    status_res = run_cmd(["supervisorctl", "status", "piebot_lc0"])
    log(f"status: {status_res.stdout.strip()}")
    if "RUNNING" not in status_res.stdout:
        raise RuntimeError(f"piebot_lc0 failed to enter RUNNING state: {status_res.stdout}")

    # Check processes
    pgrep = run_cmd(["pgrep", "-af", "training\\.nnue\\.lc0_(deploy|autopilot)"], check=False)
    log(f"Processes:\n{pgrep.stdout.strip()}")

    # Check error log tail
    err_tail = run_cmd(["tail", "-n", "20", str(ERR_PATH)], check=False).stdout
    if "identity mismatch" in err_tail:
        raise RuntimeError(f"Identity mismatch in error log:\n{err_tail}")

    # Check state file
    state = json.loads(STATE_PATH.read_text())
    log(f"Live state: status={state.get('status')}, completed_chunks={state.get('completed_chunks')}, source_commit={state.get('identity', {}).get('source', {}).get('commit')}")
    if state.get("identity", {}).get("source", {}).get("commit") != target_commit:
        raise RuntimeError("State source commit does not match target!")

    receipt = {
        "transition": "piebot-lc0-parallel-ingestion-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "target_commit": target_commit,
        "source_digest": source_id["digest"],
        "completed_chunks_at_stop": old_chunks,
        "live_chunks": state.get("completed_chunks"),
        "status": state.get("status")
    }
    receipt_path = OPS_DIR / "transition.json"
    atomic_write(receipt_path, json.dumps(receipt, indent=2).encode("utf-8"), mode=0o644)
    log(f"Transition receipt saved to {receipt_path}")
    return receipt


def main() -> int:
    try:
        initial_state, target_commit = preflight()
        stopped_state = stop_piebot()
        old_chunks = stopped_state.get("completed_chunks", 0)
        source_id = update_source_and_state(stopped_state, target_commit)
        receipt = restart_and_verify(old_chunks, target_commit, source_id)
        log(f"SUCCESS: LCZero training successfully updated and running on commit {target_commit[:7]}!")
        return 0
    except Exception as exc:
        log(f"FATAL ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
