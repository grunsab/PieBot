#!/usr/bin/env python3
"""Sampling profiler without perf: run a UCI engine as a child of gdb, SIGINT it
periodically, and count the search thread's frames (self, self lines, inclusive).
Usage: gdb_profile.py ENGINE CMDS_FILE N_SAMPLES   (CMDS_FILE holds uci/setoption/go
lines; build the engine with CARGO_PROFILE_RELEASE_DEBUG=line-tables-only)."""
import collections, os, queue, re, signal, subprocess, sys, threading, time

engine, cmds, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
g = subprocess.Popen(["gdb", "-q", "-nx", "--args", engine], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l) for l in g.stdout], daemon=True).start()
def send(c): g.stdin.write(c + "\n"); g.stdin.flush()
def drain_until(marker, timeout):
    out, t = [], time.time()
    while time.time() - t < timeout:
        try: line = q.get(timeout=0.1)
        except queue.Empty: continue
        out.append(line)
        if marker in line: break
    return out
send("set pagination off"); send("set confirm off"); send("set print thread-events off")
send(f"run < <(cat {cmds}; sleep 900) > /dev/null")
time.sleep(4)
# the inferior is gdb's child
def find_inferior():
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == g.pid: continue
        try:
            exe = os.readlink(f"/proc/{d}/exe")
            ppid = int(open(f"/proc/{d}/stat").read().split(")")[1].split()[1])
        except OSError:
            continue
        if exe == os.path.realpath(engine) and ppid == g.pid:
            return int(d)
    return None
pid = find_inferior()
if pid is None:
    print("".join(drain_until("@@never@@", 2))); sys.exit("no inferior")
selfc, incl, linec = collections.Counter(), collections.Counter(), collections.Counter()
for i in range(n):
    time.sleep(0.02)
    os.kill(pid, signal.SIGINT)
    drain_until("SIGINT", 3)
    send("thread apply all bt 30"); send("echo @@END@@\\n")
    text = "".join(drain_until("@@END@@", 5))
    if i == 3: open("raw.txt","w").write(text)
    send("continue")
    for block in text.split("\nThread "):
        if "uci-sear" not in block.split("\n")[0] and "piebot-smp" not in block.split("\n")[0]:
            continue
        frames = re.findall(r"#\d+\s+(?:0x[0-9a-f]+ in )?(\S+?) \(.*? at (\S+)", block)
        lines0 = " <- ".join(f + "@" + l.split("/")[-1] for f, l in frames[:4]) if frames else None
        frames = [f for f, _ in frames]
        frames = [re.sub(r"::h[0-9a-f]{16}$", "", f) for f in frames]
        if frames:
            selfc[frames[0]] += 1
            linec[lines0] += 1
            for f in set(frames): incl[f] += 1
send("kill"); send("quit")
tot = max(sum(selfc.values()), 1)
print("samples", tot)
print("== self"); [print(f"{c*100/tot:5.1f}% {f}") for f, c in selfc.most_common(30)]
print("== self lines"); [print(f"{c*100/tot:5.1f}% {f}") for f, c in linec.most_common(30)]
print("== inclusive"); [print(f"{c*100/tot:5.1f}% {f}") for f, c in incl.most_common(40)]
