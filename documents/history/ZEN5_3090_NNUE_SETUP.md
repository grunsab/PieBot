# PieBot NNUE Ubuntu Setup (Zen5 9755 + RTX 3090)

Yes: this pipeline runs on Ubuntu out of the box once Rust, Python deps, and NVIDIA driver are installed.

## 1) Prerequisites
Ubuntu 22.04/24.04, RTX 3090, 64 GB+ RAM, NVMe storage.

## 2) Install OS tools
```bash
sudo apt update
sudo apt install -y \
  build-essential clang lld pkg-config git curl ca-certificates \
  python3 python3-venv python3-pip \
  tmux htop nvtop jq
```

## 3) Install NVIDIA driver
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi
```

## 4) Install Rust
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install stable
rustup default stable
```

## 5) Clone repo + Python env
```bash
cd /opt
sudo git clone <YOUR_REPO_URL> piebot_rust
sudo chown -R "$USER":"$USER" /opt/piebot_rust
sudo install -d -o "$USER" -g "$USER" /opt/piebot_runs
cd /opt/piebot_rust
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r training/nnue/requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

`/opt/piebot_runs` must be owned and writable by the account that runs the
training process (and by the `User=` configured for systemd). Verify it before
starting an unattended job:

```bash
test -w /opt/piebot_runs
stat -c '%U:%G %A %n' /opt/piebot_runs
```

## 6) Verify PyTorch CUDA
```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment")
PY
```

## 7) Build required binaries
```bash
cd /opt/piebot_rust
cargo build --locked --release --manifest-path PieBot/Cargo.toml \
  --bin uci --bin selfplay --bin relabel_jsonl --bin compare_play \
  --bin accept --bin accept_temp
```

The normal `--bin uci` build is the production Cozy Chess engine and includes
the NNUE-capable UCI path. The old Pleco adapter is compatibility-only and is
opted into explicitly with `--features board-pleco`; do not enable it for the
NNUE training deployment.

## 8) Smoke test (1 tiny cycle)
```bash
cd /opt/piebot_rust
source .venv/bin/activate
python -m training.nnue.autopilot --out-root /opt/piebot_runs/smoke --max-cycles 1 --selfplay-games 2 --selfplay-depth 1 --teacher-relabel-depth 2 --teacher-relabel-every 2 --epochs 1 --batch-size 64 --trainer-backend torch --trainer-device cuda
```

## 9) Production run (7 days)
```bash
cd /opt/piebot_rust
source .venv/bin/activate
python -m training.nnue.autopilot --out-root /opt/piebot_runs/zen5_7d --profile zen5_9755_7d --hours 168 --warm-start --warm-start-learning-rate 0.001 --trainer-backend torch --trainer-device cuda
```

## 10) Set-and-forget via systemd
```bash
sudo tee /etc/systemd/system/piebot-autopilot.service >/dev/null <<'EOF'
[Unit]
Description=PieBot NNUE Autopilot
After=network-online.target
[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/piebot_rust
Environment=PATH=/opt/piebot_rust/.venv/bin:/home/YOUR_USER/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/piebot_rust/.venv/bin/python -m training.nnue.autopilot --out-root /opt/piebot_runs/zen5_7d --profile zen5_9755_7d --hours 168 --warm-start --warm-start-learning-rate 0.001 --trainer-backend torch --trainer-device cuda
Restart=on-failure
RestartSec=15
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now piebot-autopilot.service
```

Replace both `YOUR_USER` placeholders with the account that installed Rust and
owns `/opt/piebot_rust` and `/opt/piebot_runs`.

## 11) Operate and monitor
```bash
journalctl -u piebot-autopilot.service -f
watch -n 2 nvidia-smi
jq . /opt/piebot_runs/zen5_7d/autopilot_state.json
```

## 12) Notes
- Self-play and relabel are CPU-heavy; training is GPU-heavy.
- A failed process resumes automatically; a clean exit after the 168-hour deadline stays stopped.
- Resume is automatic if the process fails or the machine restarts before the deadline.
- Main artifacts are in `/opt/piebot_runs/zen5_7d` (state file and per-cycle folders).
