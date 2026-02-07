#!/usr/bin/env python3
"""
Tiny NNUE trainer (bootstrap implementation).

Reads JSONL training data with {"fen": str, "result": int}, computes HalfKP-style
active feature indices, and trains a small one-hidden-layer scalar network
(ReLU + linear head) with minibatch SGD on MSE targets.

Usage:
  python training/nnue/train_stub.py \
    --jsonl-dir data/nnue_jsonl/test80 --batch-size 4096 --max-samples 500000 \
    --epochs 8 --val-split 0.1 --learning-rate 0.05 --out out/nnue_stub_train

Requires:
  - python-chess
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

try:
    import chess  # type: ignore
except Exception as e:
    raise SystemExit("python-chess is required: pip install python-chess")

try:
    from .dataloader import TrainingRecord, jsonl_to_training_samples, read_jsonl_dir
except Exception:
    from dataloader import TrainingRecord, jsonl_to_training_samples, read_jsonl_dir  # type: ignore


def featureize_fen_counts(fen: str) -> List[int]:
    """Simple placeholder features: counts of 12 piece types (white/black x 6).
    Order: [P,N,B,R,Q,K, p,n,b,r,q,k]
    """
    board = chess.Board(fen)
    order = [
        (chess.WHITE, chess.PAWN),
        (chess.WHITE, chess.KNIGHT),
        (chess.WHITE, chess.BISHOP),
        (chess.WHITE, chess.ROOK),
        (chess.WHITE, chess.QUEEN),
        (chess.WHITE, chess.KING),
        (chess.BLACK, chess.PAWN),
        (chess.BLACK, chess.KNIGHT),
        (chess.BLACK, chess.BISHOP),
        (chess.BLACK, chess.ROOK),
        (chess.BLACK, chess.QUEEN),
        (chess.BLACK, chess.KING),
    ]
    feats = []
    for color, piece in order:
        bb = board.pieces(piece, color)
        feats.append(len(bb))
    return feats


HALFKP_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
]
HALFKP_DIM = 2 * 64 * len(HALFKP_PIECE_ORDER) * 64


def _active_halfkp_indices(fen: str) -> List[int]:
    board = chess.Board(fen)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return []
    out: List[int] = []
    for side_off, (color, ksq) in enumerate(((chess.WHITE, wk), (chess.BLACK, bk))):
        for piece_idx, piece in enumerate(HALFKP_PIECE_ORDER):
            for sq in board.pieces(piece, color):
                idx = (((side_off * 64 + int(ksq)) * len(HALFKP_PIECE_ORDER) + piece_idx) * 64) + int(sq)
                out.append(idx)
    return out


def iterate_samples(jsonl_dir: Path, max_samples: int) -> Iterator[Tuple[List[int], TrainingRecord]]:
    count = 0
    for record in jsonl_to_training_samples(read_jsonl_dir(str(jsonl_dir))):
        feats = _active_halfkp_indices(record.fen)
        yield feats, record
        count += 1
        if max_samples and count >= max_samples:
            break


def _result_to_target_cp(result: int, target_cp: float) -> float:
    if result > 0:
        return float(target_cp)
    if result < 0:
        return -float(target_cp)
    return 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _target_cp_for_record(
    record: TrainingRecord,
    *,
    target_cp: float,
    teacher_mix: float,
    max_teacher_cp: float,
    outcome_decay: float = 1.0,
) -> float:
    result_q = float(record.result_q)
    if not math.isfinite(result_q):
        result_q = float(record.result)
    result_q = _clamp(result_q, -1.0, 1.0)
    outcome_cp = result_q * float(target_cp)
    if record.ply is not None and record.ply > 0 and outcome_decay < 0.999999:
        outcome_cp *= float(outcome_decay) ** int(record.ply)

    if record.value_cp is None or not math.isfinite(float(record.value_cp)):
        return outcome_cp

    teacher_cp = _clamp(float(record.value_cp), -float(max_teacher_cp), float(max_teacher_cp))
    mix = _clamp(float(teacher_mix), 0.0, 1.0)
    return mix * teacher_cp + (1.0 - mix) * outcome_cp


def _eval_split(
    w1: List[float],
    b1: List[float],
    w2: List[float],
    b2: float,
    input_dim: int,
    xs: Sequence[Sequence[int]],
    ys: Sequence[float],
) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    loss_sum = 0.0
    correct = 0
    hidden_dim = len(b1)
    for i in range(len(xs)):
        act = xs[i]
        y = ys[i]
        hpre = [0.0] * hidden_dim
        for j in range(hidden_dim):
            off = j * input_dim
            s = b1[j]
            for idx in act:
                s += w1[off + idx]
            hpre[j] = s
        h = [v if v > 0.0 else 0.0 for v in hpre]
        pred = b2
        for j in range(hidden_dim):
            pred += w2[j] * h[j]
        diff = pred - y
        loss_sum += diff * diff
        pred_label = 1 if pred > 1e-6 else (-1 if pred < -1e-6 else 0)
        true_label = 1 if y > 1e-6 else (-1 if y < -1e-6 else 0)
        if pred_label == true_label:
            correct += 1
    n = float(len(xs))
    return loss_sum / n, correct / n


def train_model(
    *,
    jsonl_dir: Path,
    batch_size: int = 4096,
    max_samples: int = 200000,
    epochs: int = 8,
    val_split: float = 0.1,
    learning_rate: float = 0.05,
    hidden_dim: int = 16,
    target_cp: float = 100.0,
    teacher_mix: float = 0.7,
    max_teacher_cp: float = 1500.0,
    outcome_decay: float = 1.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 5.0,
    seed: int = 1,
    out_dir: Path,
) -> Dict[str, object]:
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    val_split = min(0.9, max(0.0, float(val_split)))
    lr = max(1e-6, float(learning_rate))
    hidden_dim = max(1, int(hidden_dim))
    target_cp = max(1.0, float(target_cp))
    teacher_mix = _clamp(float(teacher_mix), 0.0, 1.0)
    max_teacher_cp = max(1.0, float(max_teacher_cp))
    outcome_decay = _clamp(float(outcome_decay), 0.0, 1.0)
    adam_beta1 = _clamp(float(adam_beta1), 0.0, 0.9999)
    adam_beta2 = _clamp(float(adam_beta2), 0.0, 0.99999)
    adam_eps = max(1e-12, float(adam_eps))
    grad_clip = max(0.0, float(grad_clip))

    xs: List[List[int]] = []
    ys: List[float] = []
    best_move_available = 0
    teacher_value_available = 0
    for feats, record in iterate_samples(jsonl_dir, max_samples):
        xs.append(feats)
        ys.append(
            _target_cp_for_record(
                record,
                target_cp=target_cp,
                teacher_mix=teacher_mix,
                max_teacher_cp=max_teacher_cp,
                outcome_decay=outcome_decay,
            )
        )
        if record.best_move:
            best_move_available += 1
        if record.value_cp is not None:
            teacher_value_available += 1

    if not xs:
        raise ValueError("no training samples were loaded")

    dim = HALFKP_DIM
    rng = random.Random(seed)
    order = list(range(len(xs)))
    rng.shuffle(order)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    val_count = int(len(xs) * val_split)
    if val_split > 0.0:
        val_count = max(1, val_count)
        val_count = min(val_count, len(xs) - 1)
    else:
        val_count = 0
    train_count = len(xs) - val_count

    train_x = xs[:train_count]
    train_y = ys[:train_count]
    val_x = xs[train_count:]
    val_y = ys[train_count:]

    # Small random init.
    w1 = [(rng.random() - 0.5) * 0.01 for _ in range(hidden_dim * dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [(rng.random() - 0.5) * 0.01 for _ in range(hidden_dim)]
    b2 = 0.0
    best_w1 = list(w1)
    best_b1 = list(b1)
    best_w2 = list(w2)
    best_b2 = b2

    # Adam moments.
    m_w1 = [0.0 for _ in range(hidden_dim * dim)]
    v_w1 = [0.0 for _ in range(hidden_dim * dim)]
    m_b1 = [0.0 for _ in range(hidden_dim)]
    v_b1 = [0.0 for _ in range(hidden_dim)]
    m_w2 = [0.0 for _ in range(hidden_dim)]
    v_w2 = [0.0 for _ in range(hidden_dim)]
    m_b2 = 0.0
    v_b2 = 0.0
    adam_t = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        idx = list(range(train_count))
        rng.shuffle(idx)

        for start in range(0, train_count, batch_size):
            batch_idx = idx[start:start + batch_size]
            if not batch_idx:
                continue

            gw1 = [0.0 for _ in range(hidden_dim * dim)]
            gb1 = [0.0 for _ in range(hidden_dim)]
            gw2 = [0.0 for _ in range(hidden_dim)]
            gb2 = 0.0

            for i in batch_idx:
                act = train_x[i]
                target = train_y[i]
                hpre = [0.0 for _ in range(hidden_dim)]
                for j in range(hidden_dim):
                    off = j * dim
                    s = b1[j]
                    for f in act:
                        s += w1[off + f]
                    hpre[j] = s
                h = [v if v > 0.0 else 0.0 for v in hpre]
                pred = b2
                for j in range(hidden_dim):
                    pred += w2[j] * h[j]
                diff = pred - target
                dloss_dpred = 2.0 * diff

                gb2 += dloss_dpred
                for j in range(hidden_dim):
                    gw2[j] += dloss_dpred * h[j]
                for j in range(hidden_dim):
                    if hpre[j] <= 0.0:
                        continue
                    dpre = dloss_dpred * w2[j]
                    gb1[j] += dpre
                    off = j * dim
                    for f in act:
                        gw1[off + f] += dpre

            scale = 1.0 / float(len(batch_idx))
            if scale != 1.0:
                for j in range(hidden_dim * dim):
                    gw1[j] *= scale
                for j in range(hidden_dim):
                    gb1[j] *= scale
                    gw2[j] *= scale
                gb2 *= scale

            if grad_clip > 0.0:
                norm2 = gb2 * gb2
                for j in range(hidden_dim):
                    norm2 += gb1[j] * gb1[j]
                    norm2 += gw2[j] * gw2[j]
                for j in range(hidden_dim * dim):
                    norm2 += gw1[j] * gw1[j]
                norm = math.sqrt(norm2)
                if norm > grad_clip:
                    gscale = grad_clip / (norm + 1e-12)
                    for j in range(hidden_dim * dim):
                        gw1[j] *= gscale
                    for j in range(hidden_dim):
                        gb1[j] *= gscale
                        gw2[j] *= gscale
                    gb2 *= gscale

            adam_t += 1
            bc1 = 1.0 - (adam_beta1 ** adam_t)
            bc2 = 1.0 - (adam_beta2 ** adam_t)

            for j in range(hidden_dim * dim):
                g = gw1[j]
                m_w1[j] = adam_beta1 * m_w1[j] + (1.0 - adam_beta1) * g
                v_w1[j] = adam_beta2 * v_w1[j] + (1.0 - adam_beta2) * g * g
                mhat = m_w1[j] / bc1
                vhat = v_w1[j] / bc2
                w1[j] -= lr * mhat / (math.sqrt(vhat) + adam_eps)

            for j in range(hidden_dim):
                g1 = gb1[j]
                m_b1[j] = adam_beta1 * m_b1[j] + (1.0 - adam_beta1) * g1
                v_b1[j] = adam_beta2 * v_b1[j] + (1.0 - adam_beta2) * g1 * g1
                mhat1 = m_b1[j] / bc1
                vhat1 = v_b1[j] / bc2
                b1[j] -= lr * mhat1 / (math.sqrt(vhat1) + adam_eps)

                g2 = gw2[j]
                m_w2[j] = adam_beta1 * m_w2[j] + (1.0 - adam_beta1) * g2
                v_w2[j] = adam_beta2 * v_w2[j] + (1.0 - adam_beta2) * g2 * g2
                mhat2 = m_w2[j] / bc1
                vhat2 = v_w2[j] / bc2
                w2[j] -= lr * mhat2 / (math.sqrt(vhat2) + adam_eps)

            m_b2 = adam_beta1 * m_b2 + (1.0 - adam_beta1) * gb2
            v_b2 = adam_beta2 * v_b2 + (1.0 - adam_beta2) * gb2 * gb2
            b2 -= lr * (m_b2 / bc1) / (math.sqrt(v_b2 / bc2) + adam_eps)

        train_loss, train_acc = _eval_split(w1, b1, w2, b2, dim, train_x, train_y)
        if val_count > 0:
            val_loss, val_acc = _eval_split(w1, b1, w2, b2, dim, val_x, val_y)
        else:
            val_loss, val_acc = train_loss, train_acc

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_w1 = list(w1)
            best_b1 = list(b1)
            best_w2 = list(w2)
            best_b2 = b2

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "piebot-halfkp-mse-v2",
        "input_dim": dim,
        "hidden_dim": hidden_dim,
        "w1": best_w1,
        "b1": best_b1,
        "w2": best_w2,
        "b2": best_b2,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
    }
    metrics = {
        "train_samples": train_count,
        "val_samples": val_count,
        "input_dim": dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_eps": adam_eps,
        "grad_clip": grad_clip,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "records_with_best_move": best_move_available,
        "records_with_teacher_value": teacher_value_available,
        "records_total": len(xs),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("out/nnue_stub_train"))
    args = ap.parse_args(argv)

    metrics = train_model(
        jsonl_dir=args.jsonl_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        target_cp=args.target_cp,
        teacher_mix=args.teacher_mix,
        max_teacher_cp=args.max_teacher_cp,
        outcome_decay=args.outcome_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        out_dir=args.out,
    )
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Val samples: {metrics['val_samples']}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val loss: {metrics['best_val_loss']:.6f}")
    print(f"Wrote: {(args.out / 'checkpoint.json').as_posix()}")
    print(f"Wrote: {(args.out / 'metrics.json').as_posix()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
