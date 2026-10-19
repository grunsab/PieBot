#!/usr/bin/env python3
"""Train PieBot's arch-v3 NNUE on records made by ``lc0_to_v3bin.py``.

Architecture (``PIENNQ03``):

* Inputs per perspective: the perspective's king square is oriented (ranks
  flipped for Black) and mirrored to files a-d; it picks one of
  ``N_KING_BUCKETS`` buckets (``KING_BUCKET_LAYOUT``). Every piece, kings
  included, contributes ``bucket * 768 + (relative_color * 6 + type) * 64 +
  square`` with the same orientation and mirror applied to the square.
* One shared feature transformer (inputs -> ``hidden``) per perspective,
  SCReLU (``clamp(x, 0, 1) ** 2``), side-to-move half first.
* ``N_OUTPUT_BUCKETS`` output heads, chosen by piece count
  (``(pieces - 2) // 4``), each a linear layer over both halves.
* Output is a logit in units of ``cp / EVAL_SCALE``; the loss is BCE against
  ``0.8 * (1 + Q)/2 + 0.2 * (1 + result)/2`` (Q alone when the outcome is not
  valid), matching the LCZero lineage's target.

Quantisation mirrors the v2 net: feature weights and biases times ``QA``
(i16), output weights times ``QB`` (i8), output biases in the
``QA * QA * QB`` domain (i32); cp = ``(sum + b2) * EVAL_SCALE / (QA*QA*QB)``.
``quant_eval_fen`` is a pure-Python integer reference of the engine's
evaluation, used by the Rust parity test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from pathlib import Path

import numpy as np

MAGIC = b"PIENNQ03"
FILE_VERSION_SINGLE = 1   # feature transformer + bucketed linear head
FILE_VERSION_HIDDEN = 2   # + bucketed hidden stack 2h -> L1 -> L2 -> 1
FILE_VERSION_EXPERTS = 3  # linear head; input layer split into material experts
L1_IN_SCALE = 254         # SCReLU output 0..1 -> u8 0..254 ((acc clamped)^2 >> 8)
L1_QB = 64                # L1 weights as i8
N_KING_BUCKETS = 16
# Indexed by rank * 4 + file for the mirrored half (files a-d).
KING_BUCKET_LAYOUT = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
]
N_OUTPUT_BUCKETS = 8
N_FEATURES = N_KING_BUCKETS * 768
# --buckets 32: one bucket per oriented, mirrored king square.
KING_BUCKET_LAYOUT_32 = list(range(32))
KING_BUCKET_LAYOUT_16 = list(KING_BUCKET_LAYOUT)

# Mixture of experts on the input layer (idea: Triumviratus' "Consilium"
# network): the feature transformer holds one full set of king-bucketed
# weights per material band, and the piece count picks which set a position
# uses. Exactly one expert is active, so the layer has N times the weights at
# almost the cost of one. Routing is a fixed rule, not a learned gate.
N_EXPERTS = 1
EXPERT_OF_PIECES = [0] * 33


def experts_table(n: int) -> list:
    """Piece count (0..32) -> expert. 4 experts: 32-25, 24-17, 16-9, 8-2 pieces."""
    if n == 1:
        return [0] * 33
    if n == 4:
        return [0 if p >= 25 else 1 if p >= 17 else 2 if p >= 9 else 3 for p in range(33)]
    raise ValueError("experts must be 1 or 4")


def set_experts(n: int) -> None:
    global N_EXPERTS, EXPERT_OF_PIECES, N_FEATURES
    N_EXPERTS = n
    EXPERT_OF_PIECES = experts_table(n)
    N_FEATURES = N_EXPERTS * N_KING_BUCKETS * 768


def set_king_buckets(n: int) -> None:
    """Switch the module to 16 (grouped) or 32 (one per square) king buckets."""
    global N_KING_BUCKETS, KING_BUCKET_LAYOUT, N_FEATURES
    if n == 32:
        KING_BUCKET_LAYOUT = KING_BUCKET_LAYOUT_32
    elif n == 16:
        KING_BUCKET_LAYOUT = KING_BUCKET_LAYOUT_16
    else:
        raise ValueError("king buckets must be 16 or 32")
    N_KING_BUCKETS = n
    N_FEATURES = N_EXPERTS * n * 768
QA, QB, EVAL_SCALE = 255, 64, 400
W2_LIMIT = 127 / QB  # keep output weights representable as i8


def output_bucket(pieces: int) -> int:
    return min(max((pieces - 2) // 4, 0), N_OUTPUT_BUCKETS - 1)


def piece_features(squares_codes, king_white: int, king_black: int):
    """Feature indices for both perspectives from (square, code) pairs.

    Reference implementation in plain Python (the GPU path must agree).
    """
    out = []
    expert = EXPERT_OF_PIECES[len(squares_codes)]
    for persp in (0, 1):
        k = king_white if persp == 0 else king_black ^ 56
        mirror = (k & 7) >= 4
        if mirror:
            k ^= 7
        bucket = expert * N_KING_BUCKETS + KING_BUCKET_LAYOUT[(k >> 3) * 4 + (k & 7)]
        feats = []
        for sq, code in squares_codes:
            color, ptype = divmod(code, 6)
            s = sq if persp == 0 else sq ^ 56
            if mirror:
                s ^= 7
            rel = color if persp == 0 else 1 - color
            feats.append(bucket * 768 + (rel * 6 + ptype) * 64 + s)
        out.append(feats)
    return out


# ---------------------------------------------------------------------------
# Training (torch imported lazily so the reference helpers work without it)
# ---------------------------------------------------------------------------

def build_model(hidden: int, l1: int = 0, l2: int = 32):
    import torch

    class V3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = l1
            self.l2 = l2
            self.ft = torch.nn.Parameter(torch.empty(N_FEATURES + 1, hidden))  # last row: padding
            self.ft_bias = torch.nn.Parameter(torch.full((hidden,), 0.5))
            self.out_w = torch.nn.Parameter(torch.empty(N_OUTPUT_BUCKETS, 2 * hidden))
            self.out_b = torch.nn.Parameter(torch.zeros(N_OUTPUT_BUCKETS))
            with torch.no_grad():
                torch.nn.init.normal_(self.ft, 0.0, 0.5 / math.sqrt(30.0))
                self.ft[N_FEATURES].zero_()
                torch.nn.init.uniform_(self.out_w, -1.0 / math.sqrt(2 * hidden), 1.0 / math.sqrt(2 * hidden))
            if l1 > 0:
                nb = N_OUTPUT_BUCKETS
                self.h1_w = torch.nn.Parameter(torch.empty(nb * l1, 2 * hidden))
                self.h1_b = torch.nn.Parameter(torch.zeros(nb * l1))
                self.h2_w = torch.nn.Parameter(torch.empty(nb * l2, l1))
                self.h2_b = torch.nn.Parameter(torch.zeros(nb * l2))
                self.h3_w = torch.nn.Parameter(torch.empty(nb, l2))
                self.h3_b = torch.nn.Parameter(torch.zeros(nb))
                with torch.no_grad():
                    for w, fan in ((self.h1_w, 2 * hidden), (self.h2_w, l1), (self.h3_w, l2)):
                        torch.nn.init.uniform_(w, -1.0 / math.sqrt(fan), 1.0 / math.sqrt(fan))

        def forward(self, idx_stm, idx_nstm, bucket):
            f = torch.nn.functional.embedding_bag
            a = f(idx_stm, self.ft, mode="sum", padding_idx=N_FEATURES) + self.ft_bias
            b = f(idx_nstm, self.ft, mode="sum", padding_idx=N_FEATURES) + self.ft_bias
            h = torch.cat([a, b], dim=1).clamp(0.0, 1.0).pow(2)
            if self.l1 == 0:
                w = self.out_w[bucket]
                return (h * w).sum(1) + self.out_b[bucket]
            # Every bucket's layer at once, then keep the position's bucket.
            B = h.shape[0]
            ar = torch.arange(B, device=h.device)
            y1 = (h @ self.h1_w.t() + self.h1_b).view(B, -1, self.l1)[ar, bucket].clamp(0.0, 1.0)
            y2 = (y1 @ self.h2_w.t() + self.h2_b).view(B, -1, self.l2)[ar, bucket].clamp(0.0, 1.0)
            return (y2 * self.h3_w[bucket]).sum(1) + self.h3_b[bucket]

    return V3()


def gpu_features(occ, pcs, stm, device):
    """Batch feature indices (B, 32) for stm/nstm, output buckets, from raw records."""
    import torch

    occ = torch.as_tensor(occ.astype(np.int64), device=device)
    pcs = torch.as_tensor(pcs, device=device).long()
    stm = torch.as_tensor(stm, device=device).long()
    B = occ.shape[0]
    sq = torch.arange(64, device=device)
    mask = ((occ.unsqueeze(1) >> sq) & 1).bool()                      # (B, 64)
    rank = torch.cumsum(mask.long(), 1) - 1                            # nibble index per square
    nib = torch.gather(pcs, 1, (rank.clamp(min=0) // 2))               # (B, 64)
    code = (nib >> (4 * (rank.clamp(min=0) % 2))) & 0xF
    code = torch.where(mask, code, torch.full_like(code, -1))
    color = torch.div(code, 6, rounding_mode="floor")
    ptype = code % 6
    n_pieces = mask.sum(1)
    kw = torch.argmax((code == 5).long(), 1)
    kb = torch.argmax((code == 11).long(), 1)
    sqb = sq.unsqueeze(0).expand(B, 64)
    layout = torch.tensor(KING_BUCKET_LAYOUT, device=device)
    expert = torch.tensor(EXPERT_OF_PIECES, device=device)[n_pieces]

    def persp(p):
        k = kw if p == 0 else kb ^ 56
        mir = (k & 7) >= 4
        k = torch.where(mir, k ^ 7, k)
        bucket = expert * N_KING_BUCKETS + layout[(k >> 3) * 4 + (k & 7)]
        s = sqb if p == 0 else sqb ^ 56
        s = torch.where(mir.unsqueeze(1), s ^ 7, s)
        rel = color if p == 0 else 1 - color
        f = bucket.unsqueeze(1) * 768 + (rel * 6 + ptype) * 64 + s
        f = torch.where(mask, f, torch.full_like(f, N_FEATURES))
        # Compact the (up to 32) real features to the front.
        order = torch.argsort((~mask).long(), dim=1, stable=True)[:, :32]
        return torch.gather(f, 1, order)

    fw, fb = persp(0), persp(1)
    stm_is_black = (stm == 1).unsqueeze(1)
    idx_stm = torch.where(stm_is_black, fb, fw)
    idx_nstm = torch.where(stm_is_black, fw, fb)
    bucket = ((n_pieces - 2) // 4).clamp(0, N_OUTPUT_BUCKETS - 1)
    return idx_stm, idx_nstm, bucket


def targets(q, result, flags, device, wdl_weight):
    import torch

    q = torch.as_tensor(q.astype(np.float32) / 32767.0, device=device)
    r = torch.as_tensor(result.astype(np.float32), device=device)
    valid = torch.as_tensor((flags & 1).astype(bool), device=device)
    pq = (1 + q) / 2
    pr = (1 + r) / 2
    return torch.where(valid, (1 - wdl_weight) * pq + wdl_weight * pr, pq)


class Shards:
    """Random slices from many record files, shuffled together."""

    def __init__(self, files, slice_len=65536, slices=48, seed=0):
        from lc0_to_v3bin import V3_DTYPE

        files = [f for f in files if Path(f).stat().st_size >= V3_DTYPE.itemsize]
        self.maps = [np.memmap(f, dtype=V3_DTYPE, mode="r") for f in files]
        self.maps = [m for m in self.maps if len(m) > 0]
        self.sizes = np.array([len(m) for m in self.maps], dtype=np.float64)
        self.slice_len = slice_len
        self.slices = slices
        self.rng = np.random.default_rng(seed)

    def total(self):
        return int(self.sizes.sum())

    def superbatch(self):
        picks = self.rng.choice(len(self.maps), size=self.slices, p=self.sizes / self.sizes.sum())
        parts = []
        for i in picks:
            m = self.maps[i]
            n = min(self.slice_len, len(m))
            start = int(self.rng.integers(0, len(m) - n + 1))
            parts.append(np.asarray(m[start:start + n]))
        arr = np.concatenate(parts)
        self.rng.shuffle(arr)
        return arr


def evaluate(model, data, device, wdl_weight, batch=16384):
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), batch):
            d = data[i:i + batch]
            a, b, bk = gpu_features(d["occ"], d["pcs"], d["stm"], device)
            p = targets(d["q"], d["result"], d["flags"], device, wdl_weight)
            out = model(a, b, bk)
            tot += torch.nn.functional.binary_cross_entropy_with_logits(out, p, reduction="sum").item()
            n += len(d)
    model.train()
    return tot / max(n, 1)


def export(model, path: Path, hidden: int) -> None:
    import torch

    l1 = getattr(model, "l1", 0)
    with torch.no_grad():
        ft = model.ft[:N_FEATURES].detach().cpu().numpy()
        fb = model.ft_bias.detach().cpu().numpy()
        ow = model.out_w.detach().cpu().numpy()
        ob = model.out_b.detach().cpu().numpy()
    w1 = np.clip(np.round(ft * QA), -32768, 32767).astype("<i2")
    b1 = np.clip(np.round(fb * QA), -32768, 32767).astype("<i2")
    w2 = np.clip(np.round(ow * QB), -128, 127).astype("i1")
    b2 = np.round(ob * QA * QA * QB).astype("<i4")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        if l1 and N_EXPERTS > 1:
            raise ValueError("experts with a hidden stack are not supported")
        version = FILE_VERSION_HIDDEN if l1 else FILE_VERSION_EXPERTS if N_EXPERTS > 1 else FILE_VERSION_SINGLE
        f.write(MAGIC)
        f.write(struct.pack("<II", version, N_KING_BUCKETS))
        f.write(bytes(KING_BUCKET_LAYOUT))
        if version == FILE_VERSION_EXPERTS:
            f.write(struct.pack("<I", N_EXPERTS))
            f.write(bytes(EXPERT_OF_PIECES))
        f.write(struct.pack("<II", hidden, N_OUTPUT_BUCKETS))
        f.write(struct.pack("<iii", QA, QB, EVAL_SCALE))
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        f.write(w2.tobytes())
        f.write(b2.tobytes())
        if l1:
            # Hidden stack: u32 l1, u32 l2; i8 h1_w[nb*l1][2h] (x L1_QB),
            # f32 h1_b[nb*l1], f32 h2_w[nb*l2][l1], f32 h2_b[nb*l2],
            # f32 h3_w[nb][l2], f32 h3_b[nb].
            with torch.no_grad():
                h1w = np.clip(np.round(model.h1_w.cpu().numpy() * L1_QB), -128, 127).astype("i1")
                parts = [model.h1_b, model.h2_w, model.h2_b, model.h3_w, model.h3_b]
                parts = [p.detach().cpu().numpy().astype("<f4") for p in parts]
            f.write(struct.pack("<II", model.l1, model.l2))
            f.write(h1w.tobytes())
            for p in parts:
                f.write(p.tobytes())
    tmp.replace(path)


def load_quant(path: Path) -> dict:
    raw = Path(path).read_bytes()
    assert raw[:8] == MAGIC, "not a PIENNQ03 file"
    off = 8
    version, nkb = struct.unpack_from("<II", raw, off); off += 8
    layout = list(raw[off:off + 32]); off += 32
    n_experts, expert_table = 1, [0] * 33
    if version == FILE_VERSION_EXPERTS:
        (n_experts,) = struct.unpack_from("<I", raw, off); off += 4
        expert_table = list(raw[off:off + 33]); off += 33
    hidden, nout = struct.unpack_from("<II", raw, off); off += 8
    qa, qb, scale = struct.unpack_from("<iii", raw, off); off += 12
    nf = n_experts * nkb * 768
    w1 = np.frombuffer(raw, "<i2", nf * hidden, off).reshape(nf, hidden); off += nf * hidden * 2
    b1 = np.frombuffer(raw, "<i2", hidden, off); off += hidden * 2
    w2 = np.frombuffer(raw, "i1", nout * 2 * hidden, off).reshape(nout, 2 * hidden); off += nout * 2 * hidden
    b2 = np.frombuffer(raw, "<i4", nout, off); off += nout * 4
    net = dict(version=version, layout=layout, hidden=hidden, nout=nout, qa=qa, qb=qb, scale=scale,
               w1=w1, b1=b1, w2=w2, b2=b2, l1=0, n_experts=n_experts, expert_table=expert_table,
               nkb=nkb)
    if version == FILE_VERSION_HIDDEN:
        l1, l2 = struct.unpack_from("<II", raw, off); off += 8
        net["l1"], net["l2"] = l1, l2
        net["h1_w"] = np.frombuffer(raw, "i1", nout * l1 * 2 * hidden, off).reshape(nout, l1, 2 * hidden)
        off += nout * l1 * 2 * hidden
        for name, shape in (("h1_b", (nout, l1)), ("h2_w", (nout, l2, l1)), ("h2_b", (nout, l2)),
                            ("h3_w", (nout, l2)), ("h3_b", (nout,))):
            n = int(np.prod(shape))
            net[name] = np.frombuffer(raw, "<f4", n, off).reshape(shape); off += n * 4
    assert off == len(raw), "trailing bytes"
    return net


def quant_eval_fen(net: dict, fen: str) -> int:
    """Integer evaluation (side-to-move centipawns) exactly as the engine does it."""
    import chess

    board = chess.Board(fen)
    pairs = [(sq, (0 if p.color else 1) * 6 + p.piece_type - 1) for sq, p in board.piece_map().items()]
    kw = board.king(chess.WHITE)
    kb = board.king(chess.BLACK)
    # Evaluate with the file's own configuration.
    saved = (N_EXPERTS, EXPERT_OF_PIECES, N_KING_BUCKETS, KING_BUCKET_LAYOUT, N_FEATURES)
    globals().update(N_EXPERTS=net["n_experts"], EXPERT_OF_PIECES=net["expert_table"],
                     N_KING_BUCKETS=net["nkb"], KING_BUCKET_LAYOUT=net["layout"],
                     N_FEATURES=net["n_experts"] * net["nkb"] * 768)
    try:
        fw, fbk = piece_features(pairs, kw, kb)
    finally:
        globals().update(N_EXPERTS=saved[0], EXPERT_OF_PIECES=saved[1], N_KING_BUCKETS=saved[2],
                         KING_BUCKET_LAYOUT=saved[3], N_FEATURES=saved[4])
    acc = []
    for feats in (fw, fbk):
        a = net["b1"].astype(np.int64).copy()
        for f in feats:
            a += net["w1"][f]
        acc.append(a)
    stm, nstm = (acc[0], acc[1]) if board.turn else (acc[1], acc[0])
    h = np.concatenate([np.clip(stm, 0, net["qa"]), np.clip(nstm, 0, net["qa"])])
    bucket = output_bucket(len(pairs))
    if net.get("l1"):
        # u8 activations: (clamped acc)^2 >> 8, i32 dot with i8 weights, then f32.
        x = (h * h) >> 8
        y1 = (x.astype(np.int64) @ net["h1_w"][bucket].astype(np.int64).T).astype(np.float32)
        y1 = np.clip(y1 / np.float32(L1_IN_SCALE * L1_QB) + net["h1_b"][bucket], 0, 1).astype(np.float32)
        y2 = np.clip(net["h2_w"][bucket] @ y1 + net["h2_b"][bucket], 0, 1).astype(np.float32)
        out = np.float32(np.dot(net["h3_w"][bucket], y2) + net["h3_b"][bucket])
        return int(np.float32(out * net["scale"]))
    s = int((h * h * net["w2"][bucket].astype(np.int64)).sum()) + int(net["b2"][bucket])
    num = s * net["scale"]
    den = net["qa"] * net["qa"] * net["qb"]
    # Rust integer division truncates toward zero.
    return int(num / den) if num >= 0 else -int(-num // den)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with train/ and holdout/ *.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--buckets", type=int, default=16, choices=(16, 32))
    ap.add_argument("--experts", type=int, default=1, choices=(1, 4),
                    help="material experts on the input layer (see set_experts)")
    ap.add_argument("--l1", type=int, default=0, help="hidden stack width (0: linear head)")
    ap.add_argument("--l2", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--final-lr-frac", type=float, default=0.03)
    ap.add_argument("--positions", type=float, default=2e9, help="total training positions")
    ap.add_argument("--wdl-weight", type=float, default=0.2)
    ap.add_argument("--eval-every", type=float, default=5e7)
    ap.add_argument("--holdout-size", type=int, default=1_000_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    set_king_buckets(a.buckets)
    set_experts(a.experts)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_files = sorted(Path(a.data, "train").glob("*.bin"))
    hold_files = sorted(Path(a.data, "holdout").glob("*.bin"))
    from lc0_to_v3bin import V3_DTYPE

    hold = np.concatenate([np.fromfile(f, dtype=V3_DTYPE) for f in hold_files])
    rng = np.random.default_rng(12345)
    hold = hold[rng.permutation(len(hold))[: a.holdout_size]]
    data = Shards(train_files, seed=a.seed)
    print(f"train positions {data.total():,} in {len(train_files)} files; holdout {len(hold):,}", flush=True)

    model = build_model(a.hidden, a.l1, a.l2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    seen = 0
    state_path = out / "state.pt"
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        seen = st["seen"]
        print(f"resumed at {seen:,} positions", flush=True)
    total = int(a.positions)
    next_eval = (seen // int(a.eval_every) + 1) * int(a.eval_every)
    log = open(out / "log.jsonl", "a")
    t0, seen0 = time.time(), seen
    while seen < total:
        sb = data.superbatch()
        for i in range(0, len(sb) - a.batch + 1, a.batch):
            d = sb[i:i + a.batch]
            frac = seen / total
            lr = a.lr * (a.final_lr_frac ** frac)  # exponential decay to final_lr_frac
            for g in opt.param_groups:
                g["lr"] = lr
            ia, ib, bk = gpu_features(d["occ"], d["pcs"], d["stm"], device)
            p = targets(d["q"], d["result"], d["flags"], device, a.wdl_weight)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ia, ib, bk), p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.out_w.clamp_(-W2_LIMIT, W2_LIMIT)
                if a.l1:
                    model.h1_w.clamp_(-127 / L1_QB, 127 / L1_QB)
                model.ft[N_FEATURES].zero_()
            seen += a.batch
            if seen >= next_eval or seen >= total:
                next_eval += int(a.eval_every)
                v = evaluate(model, hold, device, a.wdl_weight)
                rate = (seen - seen0) / (time.time() - t0)
                rec = {"seen": seen, "train_loss": round(loss.item(), 6), "holdout_loss": round(v, 6),
                       "lr": lr, "pos_per_s": round(rate)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "seen": seen}, state_path)
                export(model, out / f"v3_{seen // 1_000_000:06d}M.nnue", a.hidden)
            if seen >= total:
                break
    export(model, out / "v3_final.nnue", a.hidden)


if __name__ == "__main__":
    main()
