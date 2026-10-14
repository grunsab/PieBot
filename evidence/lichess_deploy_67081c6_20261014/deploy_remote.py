"""Deploy PieBot main 67081c6 (in-search repetition draws) to the Lichess bot.

Runs on the bot host with the bot's own Python. Steps, each checked:
  1. stage the source from source.tar.gz and verify every file against the
     manifest; link the deployed net and verify its sha256;
  2. wait until no game is in progress, then stop the runner (SIGINT, never a
     forced kill), so compiling and testing cannot slow a live game;
  3. build and test the candidate: library tests, the game-derived and
     repetition regressions, UCI tests, matein3 at depth 7 on 1 and 2 threads
     (91/91, 7087288 nodes on one thread) and the forced perpetual scored 0;
  4. switch engine.dir/working_dir in config.yml (backup kept, every other
     setting verified unchanged) and start the runner again.
If any step before 4 fails, the runner is restarted on the unchanged config.

Usage: python deploy_remote.py [--rehearse]
--rehearse works on a copy of config.yml, never touches processes or the
Lichess API, and only builds and tests (for a dry run on another machine).
Environment: DEPLOY_BASE (default /home/ubuntu/lichess-bot).
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import yaml

COMMIT = "67081c6"
NET_SHA = "56434c1bacf5165baaebc6a5a06d5542d69828dc7b1ff09a934fb8263ea917d0"
EXPECTED_NODES_1T = 7087288
PERPETUAL = "6k1/6p1/5p2/7Q/8/1r6/r1q3PP/7K w - - 0 1"

BASE = pathlib.Path(os.environ.get("DEPLOY_BASE", "/home/ubuntu/lichess-bot"))
KIT = pathlib.Path(__file__).resolve().parent
STAGE = BASE / f"engines/PieBot-{COMMIT}"
STATE = BASE / f"deployment_{COMMIT}"


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def log(event, **fields):
    print(json.dumps({"event": event, "at": now().isoformat(), **fields}), flush=True)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def live(pid):
    try:
        return "\nState:\tZ" not in pathlib.Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False


def runners():
    found = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = [a.decode() for a in (p / "cmdline").read_bytes().split(b"\0") if a]
            if any(pathlib.Path(a).name == "lichess-bot.py" for a in args) and os.readlink(p / "cwd") == str(BASE):
                if live(int(p.name)):
                    found.append(int(p.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    return found


def engines():
    found = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            exe = os.readlink(p / "exe")
            if "/lichess-bot/engines/" in exe and exe.endswith("/uci") and live(int(p.name)):
                found.append(int(p.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    return found


def ongoing(cfg):
    import requests

    r = requests.get("https://lichess.org/api/account/playing",
                     headers={"Authorization": "Bearer " + cfg["token"]}, timeout=20)
    r.raise_for_status()
    return [g["gameId"] for g in r.json().get("nowPlaying", [])]


def stop_when_idle(config):
    deadline = time.monotonic() + 5400
    while time.monotonic() < deadline:
        cfg = yaml.safe_load(config.read_text())
        busy_engines, games = engines(), ongoing(cfg)
        if busy_engines or games:
            log("waiting_for_game_to_finish", engine_pids=busy_engines, games=games)
            time.sleep(15)
            continue
        pids = runners()
        assert len(pids) <= 1, ("more than one runner", pids)
        for pid in pids:
            os.kill(pid, signal.SIGINT)
        end = time.monotonic() + 60
        while runners() and time.monotonic() < end:
            time.sleep(0.5)
        assert not runners(), "runner did not exit; refusing a forced kill"
        assert not engines(), "an engine process remains"
        log("runner_stopped", pids=pids)
        return pids
    raise RuntimeError("no idle window within 90 minutes; runner left running")


def start_runner(config):
    assert not runners(), "refusing a duplicate runner"
    cfg = yaml.safe_load(config.read_text())
    threads = cfg["engine"]["uci_options"]["Threads"]
    out = BASE / f"nohup.{COMMIT}.{threads}t.{now().strftime('%Y%m%dT%H%M%SZ')}.log"
    with out.open("ab", buffering=0) as f:
        child = subprocess.Popen(
            ["nohup", str(BASE / "venv/bin/python"), "-u", str(BASE / "lichess-bot.py"), "--config", str(config)],
            cwd=BASE, env=dict(os.environ, RAYON_NUM_THREADS=str(threads)),
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        time.sleep(1)
        assert child.poll() is None, f"runner exited; inspect {out}"
        if "Engine configuration OK" in out.read_text(errors="replace"):
            break
    text = out.read_text(errors="replace")
    assert "Engine configuration OK" in text, f"engine check not passed within 60 s; inspect {out}"
    log("runner_started", pid=child.pid, log=str(out), threads=threads)
    return {"pid": child.pid, "log": str(out)}


def stage_source():
    assert not STAGE.exists(), f"{STAGE} exists; inspect before retrying"
    STAGE.mkdir(parents=True)
    with tarfile.open(KIT / "source.tar.gz") as t:
        t.extractall(STAGE, filter="data")
    manifest = json.loads((KIT / "source_manifest.json").read_text())
    bad = [p for p, h in manifest["files"].items() if sha256(STAGE / p) != h]
    assert not bad, ("source differs from manifest", bad[:5])
    (STAGE / "models").mkdir()
    net = BASE / "engines/PieBot/models/lc0_chunk_00034965.nnue"
    (STAGE / "models/lc0_chunk_00034965.nnue").symlink_to(net)
    assert sha256(STAGE / "models/lc0_chunk_00034965.nnue") == NET_SHA, "deployed net changed"
    log("staged", stage=str(STAGE), files=len(manifest["files"]))


def run(cmd, name, env, cwd):
    path = STATE / f"{name}.log"
    started = time.time()
    with path.open("wb") as f:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log("step", name=name, returncode=r.returncode, seconds=round(time.time() - started, 1), log=str(path))
    if r.returncode:
        raise RuntimeError(f"{name} failed; see {path}")
    return path.read_text(errors="replace")


def build_and_test(jobs):
    cargo_home = pathlib.Path.home() / ".cargo/bin"
    env = dict(os.environ, PATH=f"{cargo_home}:{os.environ['PATH']}", CARGO_BUILD_JOBS=str(jobs))
    crate = STAGE / "PieBot"
    run(["cargo", "build", "--release", "--locked", "--bin", "uci", "--bin", "accept", "--bin", "accept_temp"],
        "build", env, crate)
    run(["cargo", "test", "--release", "--locked", "--lib"], "test_lib", env, crate)
    run(["cargo", "test", "--release", "--locked", "--test", "repetition_in_tree", "--test", "lichess_smp_blunders",
         "--test", "lazy_smp", "--test", "search_core_temp_regressions", "--test", "uci_moves", "--test", "uci_stop"],
        "test_regressions", env, crate)
    for threads in (1, 2):
        acc_env = dict(env, PIEBOT_SUITE_FILE=str(crate / "src/suites/matein3.txt"), PIEBOT_TEST_THREADS=str(threads),
                       PIEBOT_TEST_START_DEPTH="7", PIEBOT_TEST_MAX_DEPTH="7")
        out = run([str(crate / "target/release/accept")], f"matein3_{threads}t", acc_env, crate)
        summary = [l for l in out.splitlines() if l.startswith("summary:")][-1]
        assert "cases=91" in summary, summary
        if threads == 1:
            assert f"nodes={EXPECTED_NODES_1T}" in summary, ("unexpected node signature", summary)
    # Wait for bestmove before quitting: `quit` during a search stops it.
    eng = subprocess.Popen([str(crate / "target/release/uci")], cwd=STAGE, text=True,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    eng.stdin.write(f"uci\nsetoption name NNUEQuantFile value {STAGE / 'models/lc0_chunk_00034965.nnue'}\n"
                    f"isready\nposition fen {PERPETUAL}\ngo depth 10\n")
    eng.stdin.flush()
    lines, deadline = [], time.monotonic() + 120
    while time.monotonic() < deadline:
        line = eng.stdout.readline()
        if not line:
            break
        lines.append(line.strip())
        if line.startswith("bestmove"):
            break
    eng.stdin.write("quit\n")
    eng.stdin.flush()
    eng.wait(timeout=10)
    info = [l for l in lines if l.startswith("info depth")]
    assert info and " score cp 0 " in info[-1] + " ", ("perpetual not scored as a draw", info[-1:])
    binary = crate / "target/release/uci"
    log("verified", binary=str(binary), binary_sha256=sha256(binary), perpetual=info[-1])
    return binary


def switch_config(config, binary):
    text = config.read_text()
    cfg = yaml.safe_load(text)
    backup = STATE / f"config.before_{COMMIT}.{now().strftime('%Y%m%dT%H%M%SZ')}.yml"
    backup.write_text(text)
    backup.chmod(0o600)
    expected = yaml.safe_load(text)
    expected["engine"]["dir"] = str(binary.parent) + "/"
    expected["engine"]["working_dir"] = str(STAGE)
    expected["engine"]["name"] = "uci"
    for key, value in (("dir", expected["engine"]["dir"]), ("working_dir", str(STAGE))):
        text, n = re.subn(r"(?m)^(  " + key + r":).*$", lambda m: m[1] + ' "' + value + '"', text)
        assert n == 1, (key, n)
    assert yaml.safe_load(text) == expected, "unexpected config edit"
    tmp = config.with_suffix(".yml.deploy-tmp")
    tmp.write_text(text)
    tmp.chmod(0o600)
    tmp.replace(config)
    log("config_switched", backup=str(backup), engine_dir=expected["engine"]["dir"],
        uci_options=cfg["engine"]["uci_options"])
    return backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    config = BASE / "config.yml"
    with (STATE / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = {"commit": COMMIT, "started_at": now().isoformat(), "rehearse": args.rehearse}
        stage_source()
        stopped = False
        backup = None
        try:
            if not args.rehearse:
                record["stopped_pids"] = stop_when_idle(config)
                stopped = True
            binary = build_and_test(args.jobs)
            record["binary_sha256"] = sha256(binary)
            if args.rehearse:
                rehearsal = STATE / "config.rehearsal.yml"
                shutil.copy(config, rehearsal)
                record["backup"] = str(switch_config(rehearsal, binary))
            else:
                backup = switch_config(config, binary)
                record["backup"] = str(backup)
                record["runner"] = start_runner(config)
        except BaseException as error:
            record["error"] = repr(error)
            if stopped:
                for pid in runners():
                    os.kill(pid, signal.SIGINT)
                end = time.monotonic() + 60
                while runners() and time.monotonic() < end:
                    time.sleep(0.5)
                if backup is not None:
                    tmp = config.with_suffix(".yml.rollback-tmp")
                    tmp.write_text(backup.read_text())
                    tmp.chmod(0o600)
                    tmp.replace(config)
                    record["config_restored_from"] = str(backup)
                if not runners():
                    record["restarted_old_runner"] = start_runner(config)
            raise
        finally:
            record["finished_at"] = now().isoformat()
            (STATE / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            log("done", **{k: v for k, v in record.items() if k not in ("started_at",)})


if __name__ == "__main__":
    main()
