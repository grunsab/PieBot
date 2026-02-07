#!/usr/bin/env python3
"""Torch NNUE trainer (EmbeddingBag + ReLU + linear head)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from . import train_stub
except Exception:
    import train_stub  # type: ignore


def torch_available() -> bool:
    return torch is not None


def cuda_available() -> bool:
    return torch is not None and bool(torch.cuda.is_available())


def _select_device(req: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is not installed")
    r = (req or "auto").lower()
    if r == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested cuda device but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if r == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchNnue(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.EmbeddingBag(input_dim, hidden_dim, mode="sum", sparse=False)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, flat_idx: "torch.Tensor", offsets: "torch.Tensor") -> "torch.Tensor":
        h = self.embed(flat_idx, offsets) + self.b1
        h = torch.relu(h)
        return self.out(h).squeeze(1)


def _pack_batch(
    batch_feats: Sequence[Sequence[int]],
    batch_targets: Sequence[float],
    device: "torch.device",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    flat: List[int] = []
    offsets: List[int] = []
    ofs = 0
    for feats in batch_feats:
        offsets.append(ofs)
        flat.extend(int(x) for x in feats)
        ofs += len(feats)
    if not flat:
        flat = [0]
        offsets = [0]
    flat_t = torch.tensor(flat, dtype=torch.long, device=device)
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
    targets_t = torch.tensor(batch_targets, dtype=torch.float32, device=device)
    return flat_t, offsets_t, targets_t


def _eval_split(
    model: TorchNnue,
    xs: Sequence[Sequence[int]],
    ys: Sequence[float],
    batch_size: int,
    device: "torch.device",
) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mse = torch.nn.MSELoss(reduction="mean")
    model.eval()
    loss_sum = 0.0
    n = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(xs), batch_size):
            bx = xs[start:start + batch_size]
            by = ys[start:start + batch_size]
            flat, offs, tgt = _pack_batch(bx, by, device)
            pred = model(flat, offs)
            loss = mse(pred, tgt)
            bs = len(by)
            loss_sum += float(loss.item()) * bs
            n += bs
            pred_lbl = torch.sign(pred).to(torch.int32)
            tgt_lbl = torch.sign(tgt).to(torch.int32)
            correct += int((pred_lbl == tgt_lbl).sum().item())
    return loss_sum / float(max(1, n)), float(correct) / float(max(1, n))


def train_model(
    *,
    jsonl_dir: Path,
    batch_size: int = 4096,
    max_samples: int = 200000,
    epochs: int = 8,
    val_split: float = 0.1,
    learning_rate: float = 0.05,
    hidden_dim: int = 64,
    target_cp: float = 100.0,
    teacher_mix: float = 0.7,
    max_teacher_cp: float = 1500.0,
    outcome_decay: float = 1.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 5.0,
    seed: int = 1,
    out_dir: Path = Path("out/nnue_torch_train"),
    device: str = "auto",
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("torch backend requested but torch is not installed")
    dev = _select_device(device)

    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    hidden_dim = max(1, int(hidden_dim))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    xs: List[List[int]] = []
    ys: List[float] = []
    best_move_available = 0
    teacher_value_available = 0
    for feats, record in train_stub.iterate_samples(jsonl_dir, max_samples):
        xs.append(feats)
        ys.append(
            train_stub._target_cp_for_record(
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

    input_dim = train_stub.HALFKP_DIM
    model = TorchNnue(input_dim=input_dim, hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_eps),
    )
    mse = torch.nn.MSELoss(reduction="mean")

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []

    for ep in range(epochs):
        idx = list(range(train_count))
        rng.shuffle(idx)
        model.train()
        for start in range(0, train_count, batch_size):
            bidx = idx[start:start + batch_size]
            if not bidx:
                continue
            bx = [train_x[i] for i in bidx]
            by = [train_y[i] for i in bidx]
            flat, offs, tgt = _pack_batch(bx, by, dev)
            pred = model(flat, offs)
            loss = mse(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

        tr_loss, tr_acc = _eval_split(model, train_x, train_y, batch_size, dev)
        if val_count > 0:
            va_loss, va_acc = _eval_split(model, val_x, val_y, batch_size, dev)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        train_loss_history.append(tr_loss)
        val_loss_history.append(va_loss)
        train_acc_history.append(tr_acc)
        val_acc_history.append(va_acc)
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.embed.weight.detach().cpu()  # [input, hidden]
    w1 = emb.transpose(0, 1).contiguous().view(-1).tolist()  # row-major [hidden][input]
    b1 = model.b1.detach().cpu().view(-1).tolist()
    w2 = model.out.weight.detach().cpu().view(-1).tolist()  # [hidden]
    b2 = float(model.out.bias.detach().cpu().item())

    checkpoint = {
        "format": "piebot-halfkp-mse-v2-torch",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "target_cp": target_cp,
        "teacher_mix": teacher_mix,
        "max_teacher_cp": max_teacher_cp,
        "outcome_decay": outcome_decay,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "device": str(dev),
    }
    metrics = {
        "train_samples": train_count,
        "val_samples": val_count,
        "input_dim": input_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": float(learning_rate),
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
        "best_val_loss": best_val,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "records_with_best_move": best_move_available,
        "records_with_teacher_value": teacher_value_available,
        "records_total": len(xs),
        "backend": "torch",
        "device": str(dev),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("out/nnue_torch_train"))
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
        device=args.device,
    )
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Val samples: {metrics['val_samples']}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val loss: {metrics['best_val_loss']:.6f}")
    print(f"Wrote: {(args.out / 'checkpoint.json').as_posix()}")
    print(f"Wrote: {(args.out / 'metrics.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
