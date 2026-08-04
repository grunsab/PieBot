#!/usr/bin/env python3
"""Torch NNUE trainer (EmbeddingBag + ReLU + linear head)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_DIRECT_CHECKPOINT_FORMATS = {
    "piebot-halfkp-mse-v2",
    "piebot-halfkp-mse-v2-torch",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_jsonl_source(path: Path) -> str:
    """Hash a fixed validation source, including stable relative file names."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    digest = hashlib.sha256()
    for file_path in files:
        name = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _count_jsonl_source_records(path: Path) -> int:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    records = 0
    for file_path in files:
        with file_path.open("rb") as handle:
            records += sum(1 for line in handle if line.strip())
    return records


def _load_initial_checkpoint(
    model: TorchNnue,
    checkpoint_path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    device: "torch.device",
    objective: Dict[str, Any],
    weights_only: bool = False,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"initial checkpoint does not exist: {path}")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid initial checkpoint JSON: {path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a JSON object")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in _DIRECT_CHECKPOINT_FORMATS:
        raise ValueError(f"unsupported initial checkpoint format: {checkpoint_format!r}")
    if int(checkpoint.get("input_dim", 0)) != int(input_dim):
        raise ValueError(
            f"initial checkpoint input_dim mismatch: expected {input_dim}, "
            f"got {checkpoint.get('input_dim')}"
        )
    if int(checkpoint.get("hidden_dim", 0)) != int(hidden_dim):
        raise ValueError(
            f"initial checkpoint hidden_dim mismatch: expected {hidden_dim}, "
            f"got {checkpoint.get('hidden_dim')}"
        )
    checkpoint_feature_set = checkpoint.get("feature_set")
    if checkpoint_feature_set != train_stub.FEATURE_SET:
        raise ValueError(
            f"initial checkpoint feature_set mismatch: expected {train_stub.FEATURE_SET!r}, "
            f"got {checkpoint_feature_set!r}"
        )
    checkpoint_target_schema = checkpoint.get("target_schema")
    if checkpoint_target_schema != train_stub.TARGET_SCHEMA:
        raise ValueError(
            f"initial checkpoint target_schema mismatch: expected "
            f"{train_stub.TARGET_SCHEMA!r}, got {checkpoint_target_schema!r}"
        )
    checkpoint_objective = checkpoint.get("objective")
    if not isinstance(checkpoint_objective, dict):
        raise ValueError("initial checkpoint objective metadata is missing or invalid")
    objective_transition = checkpoint_objective != objective
    if objective_transition and not weights_only:
        raise ValueError("initial checkpoint objective does not match this training run")

    expected_lengths = {
        "w1": input_dim * hidden_dim,
        "b1": hidden_dim,
        "w2": hidden_dim,
    }
    values: Dict[str, List[float]] = {}
    for key, expected_len in expected_lengths.items():
        raw = checkpoint.get(key)
        if not isinstance(raw, list) or len(raw) != expected_len:
            raise ValueError(
                f"initial checkpoint {key} size mismatch: expected {expected_len}"
            )
        try:
            converted = [float(value) for value in raw]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"initial checkpoint {key} contains non-numeric values") from exc
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"initial checkpoint {key} contains non-finite values")
        values[key] = converted
    try:
        b2 = float(checkpoint["b2"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("initial checkpoint b2 is missing or non-numeric") from exc
    if not math.isfinite(b2):
        raise ValueError("initial checkpoint b2 contains a non-finite value")

    try:
        w1_tensor = torch.tensor(
            values["w1"], dtype=model.embed.weight.dtype, device=device
        )
        b1_tensor = torch.tensor(values["b1"], dtype=model.b1.dtype, device=device)
        w2_tensor = torch.tensor(
            values["w2"], dtype=model.out.weight.dtype, device=device
        )
        b2_tensor = torch.tensor([b2], dtype=model.out.bias.dtype, device=device)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(
            "initial checkpoint weights cannot be represented as finite model tensors"
        ) from exc
    tensors = {
        "w1": w1_tensor,
        "b1": b1_tensor,
        "w2": w2_tensor,
        "b2": b2_tensor,
    }
    for key, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f"initial checkpoint {key} contains non-finite model tensor values"
            )

    with torch.no_grad():
        # Serialized w1 is row-major [hidden][input], while EmbeddingBag stores
        # [input][hidden]. The transpose is required for an exact warm start.
        model.embed.weight.copy_(
            w1_tensor.view(hidden_dim, input_dim).transpose(0, 1)
        )
        model.b1.copy_(b1_tensor)
        model.out.weight.copy_(w2_tensor.view(1, hidden_dim))
        model.out.bias.copy_(b2_tensor)

    optimizer_state = checkpoint.get("optimizer_state")
    optimizer_state_sha256 = (
        optimizer_state.get("sha256") if isinstance(optimizer_state, dict) else None
    )

    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "format": str(checkpoint_format),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "feature_set": checkpoint_feature_set,
        "target_schema": checkpoint_target_schema,
        # ``objective`` remains the requested objective as a compatibility
        # alias. The explicit source/requested fields make transitions
        # auditable without changing existing provenance consumers.
        "objective": copy.deepcopy(objective),
        "source_objective": copy.deepcopy(checkpoint_objective),
        "requested_objective": copy.deepcopy(objective),
        "mode": "weights-only" if weights_only else "strict",
        "weights_only": bool(weights_only),
        "objective_transition": bool(objective_transition),
        "weights_only_objective_transition": bool(
            weights_only and objective_transition
        ),
        "optimizer_state_sha256": optimizer_state_sha256,
    }


_OPTIMIZER_FORMAT = "piebot-torch-adam-v3"


def _parameter_shapes(model: TorchNnue) -> Dict[str, List[int]]:
    return {name: list(parameter.shape) for name, parameter in model.named_parameters()}


def _model_parameters_sha256(model: TorchNnue) -> str:
    """Return a canonical digest of the exact float32 model parameters."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        shape = [int(dimension) for dimension in parameter.shape]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        values = (
            parameter.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _to_cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        torch.save(value, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch releases
        return torch.load(path, map_location="cpu")


def _load_initial_optimizer_state(
    optimizer: "torch.optim.Optimizer",
    model: TorchNnue,
    optimizer_path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    device: "torch.device",
    objective: Dict[str, Any],
    expected_sha256: Optional[str],
) -> Dict[str, Any]:
    path = Path(optimizer_path)
    if not path.is_file():
        raise ValueError(f"initial optimizer state does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("initial checkpoint does not bind an optimizer SHA-256")
    if actual_sha256 != expected_sha256:
        raise ValueError("initial optimizer SHA-256 does not match the initial checkpoint")
    try:
        payload = _torch_load(path)
    except Exception as exc:
        raise ValueError(f"invalid initial optimizer state: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != _OPTIMIZER_FORMAT:
        raise ValueError("unsupported initial optimizer state format")
    if int(payload.get("input_dim", 0)) != int(input_dim):
        raise ValueError(
            f"initial optimizer input_dim mismatch: expected {input_dim}, "
            f"got {payload.get('input_dim')}"
        )
    if int(payload.get("hidden_dim", 0)) != int(hidden_dim):
        raise ValueError(
            f"initial optimizer hidden_dim mismatch: expected {hidden_dim}, "
            f"got {payload.get('hidden_dim')}"
        )
    if payload.get("feature_set") != train_stub.FEATURE_SET:
        raise ValueError("initial optimizer feature_set does not match the model")
    if payload.get("target_schema") != train_stub.TARGET_SCHEMA:
        raise ValueError("initial optimizer target schema does not match the model")
    if payload.get("objective") != objective:
        raise ValueError("initial optimizer objective does not match this training run")
    expected_shapes = _parameter_shapes(model)
    if payload.get("parameter_shapes") != expected_shapes:
        raise ValueError("initial optimizer parameter shapes do not match the model")
    model_parameters_sha256 = _model_parameters_sha256(model)
    if payload.get("model_parameters_sha256") != model_parameters_sha256:
        raise ValueError(
            "initial optimizer model parameters do not match the initial checkpoint"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("initial optimizer state_dict is missing")
    serialized_groups = state_dict.get("param_groups")
    if not isinstance(serialized_groups, list) or len(serialized_groups) != len(
        optimizer.param_groups
    ):
        raise ValueError("initial optimizer parameter groups are incompatible")
    for serialized, requested in zip(serialized_groups, optimizer.param_groups):
        if not isinstance(serialized, dict):
            raise ValueError("initial optimizer parameter group is invalid")
        serialized_betas = serialized.get("betas")
        requested_betas = requested.get("betas")
        try:
            restored_betas = tuple(float(value) for value in serialized_betas)
            expected_betas = tuple(float(value) for value in requested_betas)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial optimizer Adam betas are invalid") from exc
        if restored_betas != expected_betas:
            raise ValueError(
                "initial optimizer Adam betas mismatch: "
                f"expected {expected_betas}, got {restored_betas}"
            )
        try:
            restored_eps = float(serialized["eps"])
            expected_eps = float(requested["eps"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial optimizer Adam epsilon is invalid") from exc
        if restored_eps != expected_eps:
            raise ValueError(
                "initial optimizer Adam epsilon mismatch: "
                f"expected {expected_eps}, got {restored_eps}"
            )
    try:
        optimizer.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("initial optimizer state_dict is incompatible") from exc

    # load_state_dict normally follows the parameters' device, but make that
    # guarantee explicit so resumed CUDA training never retains CPU moments.
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            if value.ndim > 0 and tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"initial optimizer tensor {key!r} shape does not match parameter"
                )
            state[key] = value.to(device=device)

    return {
        "path": path.resolve().as_posix(),
        "sha256": actual_sha256,
        "format": _OPTIMIZER_FORMAT,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "feature_set": train_stub.FEATURE_SET,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": objective,
        "model_parameters_sha256": model_parameters_sha256,
        "adam_betas": list(expected_betas),
        "adam_eps": expected_eps,
    }


def _save_optimizer_state(
    state_dict: Dict[str, Any],
    model: TorchNnue,
    path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    best_epoch: int,
    objective: Dict[str, Any],
) -> Dict[str, Any]:
    model_parameters_sha256 = _model_parameters_sha256(model)
    payload = {
        "format": _OPTIMIZER_FORMAT,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "feature_set": train_stub.FEATURE_SET,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": copy.deepcopy(objective),
        "parameter_shapes": _parameter_shapes(model),
        "model_parameters_sha256": model_parameters_sha256,
        "best_epoch": int(best_epoch),
        "state_dict": _to_cpu_tree(state_dict),
    }
    _atomic_torch_save(payload, path)
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "format": _OPTIMIZER_FORMAT,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "feature_set": train_stub.FEATURE_SET,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": copy.deepcopy(objective),
        "model_parameters_sha256": model_parameters_sha256,
        "best_epoch": int(best_epoch),
    }


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
    flat_t = torch.tensor(flat, dtype=torch.long, device=device)
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
    targets_t = torch.tensor(batch_targets, dtype=torch.float32, device=device)
    return flat_t, offsets_t, targets_t


def _objective_loss(
    pred_cp: "torch.Tensor",
    target_cp: "torch.Tensor",
    target_wdl: "torch.Tensor",
    *,
    loss_kind: str,
    huber_delta_cp: float,
    wdl_scale_cp: float,
) -> "torch.Tensor":
    if loss_kind == "mse":
        return torch.nn.functional.mse_loss(pred_cp, target_cp, reduction="mean")
    if loss_kind == "huber":
        return torch.nn.functional.huber_loss(
            pred_cp,
            target_cp,
            reduction="mean",
            delta=float(huber_delta_cp),
        )
    if loss_kind == "wdl":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            pred_cp / float(wdl_scale_cp),
            target_wdl,
            reduction="mean",
        )
    raise ValueError(f"unsupported loss_kind: {loss_kind!r}")


def _eval_split(
    model: TorchNnue,
    xs: Sequence[Sequence[int]],
    ys_cp: Sequence[float],
    ys_wdl: Sequence[float],
    batch_size: int,
    device: "torch.device",
    *,
    loss_kind: str,
    huber_delta_cp: float,
    wdl_scale_cp: float,
) -> Tuple[float, float, float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    model.eval()
    objective_sum = 0.0
    cp_squared_error_sum = 0.0
    n = 0
    correct = 0
    prediction_abs_sum = 0.0
    prediction_max_abs = 0.0
    with torch.no_grad():
        for start in range(0, len(xs), batch_size):
            bx = xs[start:start + batch_size]
            by_cp = ys_cp[start:start + batch_size]
            by_wdl = ys_wdl[start:start + batch_size]
            flat, offs, tgt_cp = _pack_batch(bx, by_cp, device)
            tgt_wdl = torch.tensor(by_wdl, dtype=torch.float32, device=device)
            pred = model(flat, offs)
            objective = _objective_loss(
                pred,
                tgt_cp,
                tgt_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
            bs = len(by_cp)
            objective_sum += float(objective.item()) * bs
            cp_squared_error_sum += float(
                torch.sum(torch.square(pred - tgt_cp)).item()
            )
            prediction_abs = torch.abs(pred)
            prediction_abs_sum += float(torch.sum(prediction_abs).item())
            prediction_max_abs = max(
                prediction_max_abs,
                float(torch.max(prediction_abs).item()),
            )
            n += bs
            pred_lbl = torch.sign(pred).to(torch.int32)
            tgt_lbl = torch.sign(tgt_cp).to(torch.int32)
            correct += int((pred_lbl == tgt_lbl).sum().item())
    denominator = float(max(1, n))
    return (
        objective_sum / denominator,
        cp_squared_error_sum / denominator,
        float(correct) / denominator,
        prediction_abs_sum / denominator,
        prediction_max_abs,
    )


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
    initial_checkpoint: Optional[Path] = None,
    initial_checkpoint_weights_only: bool = False,
    loss_kind: str = "mse",
    huber_delta_cp: float = 100.0,
    wdl_scale_cp: float = 400.0,
    min_teacher_depth: int = 0,
    primary_sample_fraction: float = 0.5,
    teacher_sample_fraction: float = 0.5,
    validation_jsonl_dir: Optional[Path] = None,
    max_validation_samples: int = 100000,
    validation_seed: int = 20_260_802,
    validation_require_teacher: bool = False,
    initial_optimizer_state: Optional[Path] = None,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("torch backend requested but torch is not installed")
    initial_checkpoint_weights_only = bool(initial_checkpoint_weights_only)
    if initial_checkpoint_weights_only and initial_optimizer_state is not None:
        raise ValueError(
            "weights-only initial checkpoint mode categorically forbids "
            "initial optimizer state restore"
        )
    if initial_checkpoint_weights_only and initial_checkpoint is None:
        raise ValueError(
            "initial_checkpoint_weights_only requires an initial checkpoint"
        )
    if initial_optimizer_state is not None and initial_checkpoint is None:
        raise ValueError("initial optimizer state requires an initial checkpoint")
    dev = _select_device(device)

    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    hidden_dim = max(1, int(hidden_dim))
    val_split = min(0.9, max(0.0, float(val_split)))
    target_cp = max(1.0, float(target_cp))
    teacher_mix = min(1.0, max(0.0, float(teacher_mix)))
    max_teacher_cp = max(1.0, float(max_teacher_cp))
    outcome_decay = min(1.0, max(0.0, float(outcome_decay)))
    loss_kind = str(loss_kind).lower()
    if loss_kind not in {"mse", "huber", "wdl"}:
        raise ValueError("loss_kind must be one of: mse, huber, wdl")
    huber_delta_cp = float(huber_delta_cp)
    if not math.isfinite(huber_delta_cp) or huber_delta_cp <= 0.0:
        raise ValueError("huber_delta_cp must be finite and positive")
    wdl_scale_cp = float(wdl_scale_cp)
    if not math.isfinite(wdl_scale_cp) or wdl_scale_cp <= 0.0:
        raise ValueError("wdl_scale_cp must be finite and positive")
    min_teacher_depth = max(0, int(min_teacher_depth))
    primary_sample_fraction = min(1.0, max(0.0, float(primary_sample_fraction)))
    teacher_sample_fraction = min(1.0, max(0.0, float(teacher_sample_fraction)))
    max_validation_samples = int(max_validation_samples)
    validation_seed = int(validation_seed)
    validation_require_teacher = bool(validation_require_teacher)
    if validation_jsonl_dir is not None:
        train_stub._assert_validation_source_disjoint(
            Path(validation_jsonl_dir),
            Path(jsonl_dir),
        )
    objective = train_stub.objective_metadata(
        loss_kind=loss_kind,
        target_cp=target_cp,
        teacher_mix=teacher_mix,
        max_teacher_cp=max_teacher_cp,
        outcome_decay=outcome_decay,
        min_teacher_depth=min_teacher_depth,
        huber_delta_cp=huber_delta_cp,
        wdl_scale_cp=wdl_scale_cp,
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    xs: List[List[int]] = []
    ys_cp: List[float] = []
    ys_wdl: List[float] = []
    # Keep only compact partition metadata. Retaining all 700k TrainingRecord
    # objects costs several GB, while these identities are sufficient to keep
    # every game/duplicate wholly on one side of the aligned holdout.
    validation_group_identities: List[str] = []
    validation_teacher_flags: List[bool] = []
    best_move_available = 0
    teacher_value_available = 0
    raw_teacher_value_available = 0
    for feats, record in train_stub.iterate_samples(
        jsonl_dir,
        max_samples,
        seed=seed,
        primary_sample_fraction=primary_sample_fraction,
        teacher_sample_fraction=teacher_sample_fraction,
        min_teacher_depth=min_teacher_depth,
    ):
        xs.append(feats)
        validation_group_identities.append(
            train_stub._validation_group_identity(record)
        )
        validation_teacher_flags.append(
            train_stub._teacher_available(record, min_teacher_depth)
        )
        cp, probability = train_stub._targets_for_record(
            record,
            loss_kind=loss_kind,
            target_cp=target_cp,
            teacher_mix=teacher_mix,
            max_teacher_cp=max_teacher_cp,
            outcome_decay=outcome_decay,
            min_teacher_depth=min_teacher_depth,
            wdl_scale_cp=wdl_scale_cp,
        )
        ys_cp.append(cp)
        ys_wdl.append(probability)
        if record.best_move:
            best_move_available += 1
        if record.value_cp is not None:
            raw_teacher_value_available += 1
        if train_stub._teacher_available(record, min_teacher_depth):
            teacher_value_available += 1
    if not xs:
        raise ValueError("no training samples were loaded")
    requested_teacher_samples = int(round(len(xs) * teacher_sample_fraction))
    teacher_sampling_satisfied = teacher_value_available == requested_teacher_samples
    if (
        max_samples > 0
        and 0 < teacher_value_available < len(xs)
        and not teacher_sampling_satisfied
    ):
        raise ValueError(
            "unable to satisfy teacher_sample_fraction while preserving source quotas: "
            f"requested {requested_teacher_samples}/{len(xs)}, "
            f"selected {teacher_value_available}/{len(xs)}"
        )

    order = list(range(len(xs)))
    rng.shuffle(order)
    xs = [xs[i] for i in order]
    ys_cp = [ys_cp[i] for i in order]
    ys_wdl = [ys_wdl[i] for i in order]
    validation_group_identities = [
        validation_group_identities[i] for i in order
    ]
    validation_teacher_flags = [validation_teacher_flags[i] for i in order]

    fixed_validation = validation_jsonl_dir is not None
    validation_source = None
    validation_teacher_value_available: Optional[int] = None
    validation_raw_teacher_value_available: Optional[int] = None
    validation_sample_sha256: Optional[str] = None
    train_indices, validation_indices = (
        train_stub._internal_validation_partition_from_metadata(
            validation_group_identities,
            validation_teacher_flags,
            val_split,
            validation_seed=validation_seed,
        )
    )
    train_x = [xs[idx] for idx in train_indices]
    train_y_cp = [ys_cp[idx] for idx in train_indices]
    train_y_wdl = [ys_wdl[idx] for idx in train_indices]
    val_x = [xs[idx] for idx in validation_indices]
    val_y_cp = [ys_cp[idx] for idx in validation_indices]
    val_y_wdl = [ys_wdl[idx] for idx in validation_indices]
    train_count = len(train_x)
    val_count = len(val_x)
    train_teacher_count = sum(
        validation_teacher_flags[idx] for idx in train_indices
    )
    primary_validation_teacher_count = sum(
        validation_teacher_flags[idx] for idx in validation_indices
    )
    train_groups = {
        validation_group_identities[idx] for idx in train_indices
    }
    validation_groups = {
        validation_group_identities[idx] for idx in validation_indices
    }
    internal_validation_record_overlap = len(
        train_groups.intersection(validation_groups)
    )

    reference_val_x: List[List[int]] = []
    reference_val_y_cp: List[float] = []
    reference_val_y_wdl: List[float] = []
    if fixed_validation:
        validation_teacher_value_available = 0
        validation_raw_teacher_value_available = 0
        validation_path = Path(validation_jsonl_dir)  # type: ignore[arg-type]
        validation_source_before = {
            "path": validation_path.resolve().as_posix(),
            "sha256": _sha256_jsonl_source(validation_path),
            "records": _count_jsonl_source_records(validation_path),
            "max_samples": max_validation_samples,
            "seed": validation_seed,
        }
        validation_digest = hashlib.sha256()
        for feats, record in train_stub.iterate_fixed_validation_samples(
            validation_path,
            max_validation_samples,
            seed=validation_seed,
            min_teacher_depth=min_teacher_depth,
            require_teacher=validation_require_teacher,
        ):
            validation_digest.update(
                train_stub._record_identity(record).encode("utf-8")
            )
            validation_digest.update(b"\0")
            if record.value_cp is not None:
                validation_raw_teacher_value_available += 1
            if train_stub._teacher_available(record, min_teacher_depth):
                validation_teacher_value_available += 1
            reference_val_x.append(feats)
            cp, probability = train_stub._targets_for_record(
                record,
                loss_kind=loss_kind,
                target_cp=target_cp,
                teacher_mix=teacher_mix,
                max_teacher_cp=max_teacher_cp,
                outcome_decay=outcome_decay,
                min_teacher_depth=min_teacher_depth,
                wdl_scale_cp=wdl_scale_cp,
            )
            reference_val_y_cp.append(cp)
            reference_val_y_wdl.append(probability)
        if not reference_val_x:
            raise ValueError("no fixed reference validation samples were loaded")
        validation_source = {
            "path": validation_path.resolve().as_posix(),
            "sha256": _sha256_jsonl_source(validation_path),
            "records": _count_jsonl_source_records(validation_path),
            "max_samples": max_validation_samples,
            "seed": validation_seed,
        }
        if validation_source != validation_source_before:
            raise ValueError("fixed validation source changed while trainer was reading it")
        validation_sample_sha256 = validation_digest.hexdigest()
    reference_val_count = len(reference_val_x)

    input_dim = train_stub.HALFKP_DIM
    model = TorchNnue(input_dim=input_dim, hidden_dim=hidden_dim).to(dev)
    initialized_from = None
    if initial_checkpoint is not None:
        initialized_from = _load_initial_checkpoint(
            model,
            Path(initial_checkpoint),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=dev,
            objective=objective,
            weights_only=initial_checkpoint_weights_only,
        )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_eps),
    )
    optimizer_initialized_from = None
    if initial_optimizer_state is not None:
        optimizer_initialized_from = _load_initial_optimizer_state(
            opt,
            model,
            Path(initial_optimizer_state),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=dev,
            objective=objective,
            expected_sha256=(
                initialized_from.get("optimizer_state_sha256")
                if isinstance(initialized_from, dict)
                else None
            ),
        )
        # Retain the parent's moments and step counters, but honor this cycle's
        # explicitly requested learning-rate schedule. Adam betas and epsilon
        # were validated above because changing them would reinterpret the
        # restored moments and bias-correction step counters.
        for param_group in opt.param_groups:
            param_group["lr"] = float(learning_rate)

    best_state = None
    best_optimizer_state = None
    best_val = float("inf")
    best_epoch = 0
    initial_train_loss = None
    initial_train_cp_mse = None
    initial_train_acc = None
    initial_val_loss = None
    initial_val_cp_mse = None
    initial_val_acc = None
    initial_train_prediction_mean_abs = None
    initial_val_prediction_mean_abs = None
    initial_train_prediction_max_abs = None
    initial_val_prediction_max_abs = None
    initial_reference_val_loss = None
    initial_reference_val_cp_mse = None
    initial_reference_val_acc = None
    initial_reference_val_prediction_mean_abs = None
    initial_reference_val_prediction_max_abs = None
    best_reference_val_loss = None
    best_reference_val_cp_mse = None
    best_reference_val_acc = None
    best_reference_val_prediction_mean_abs = None
    best_reference_val_prediction_max_abs = None
    if initialized_from is not None:
        (
            initial_train_loss,
            initial_train_cp_mse,
            initial_train_acc,
            initial_train_prediction_mean_abs,
            initial_train_prediction_max_abs,
        ) = _eval_split(
            model,
            train_x,
            train_y_cp,
            train_y_wdl,
            batch_size,
            dev,
            loss_kind=loss_kind,
            huber_delta_cp=huber_delta_cp,
            wdl_scale_cp=wdl_scale_cp,
        )
        if val_count > 0:
            (
                initial_val_loss,
                initial_val_cp_mse,
                initial_val_acc,
                initial_val_prediction_mean_abs,
                initial_val_prediction_max_abs,
            ) = _eval_split(
                model,
                val_x,
                val_y_cp,
                val_y_wdl,
                batch_size,
                dev,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            initial_val_loss = initial_train_loss
            initial_val_cp_mse = initial_train_cp_mse
            initial_val_acc = initial_train_acc
            initial_val_prediction_mean_abs = initial_train_prediction_mean_abs
            initial_val_prediction_max_abs = initial_train_prediction_max_abs
        if reference_val_count > 0:
            (
                initial_reference_val_loss,
                initial_reference_val_cp_mse,
                initial_reference_val_acc,
                initial_reference_val_prediction_mean_abs,
                initial_reference_val_prediction_max_abs,
            ) = _eval_split(
                model,
                reference_val_x,
                reference_val_y_cp,
                reference_val_y_wdl,
                batch_size,
                dev,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
            best_reference_val_loss = initial_reference_val_loss
            best_reference_val_cp_mse = initial_reference_val_cp_mse
            best_reference_val_acc = initial_reference_val_acc
            best_reference_val_prediction_mean_abs = (
                initial_reference_val_prediction_mean_abs
            )
            best_reference_val_prediction_max_abs = (
                initial_reference_val_prediction_max_abs
            )
        best_val = float(initial_val_loss)
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        best_optimizer_state = _to_cpu_tree(opt.state_dict())
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_cp_mse_history: List[float] = []
    val_cp_mse_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []
    train_prediction_mean_abs_history: List[float] = []
    val_prediction_mean_abs_history: List[float] = []
    train_prediction_max_abs_history: List[float] = []
    val_prediction_max_abs_history: List[float] = []
    reference_val_loss_history: List[float] = []
    reference_val_cp_mse_history: List[float] = []
    reference_val_acc_history: List[float] = []
    reference_val_prediction_mean_abs_history: List[float] = []
    reference_val_prediction_max_abs_history: List[float] = []
    reference_val_checkpoint_eligible_history: List[bool] = []

    for ep in range(epochs):
        idx = list(range(train_count))
        rng.shuffle(idx)
        model.train()
        for start in range(0, train_count, batch_size):
            bidx = idx[start:start + batch_size]
            if not bidx:
                continue
            bx = [train_x[i] for i in bidx]
            by_cp = [train_y_cp[i] for i in bidx]
            by_wdl = [train_y_wdl[i] for i in bidx]
            flat, offs, tgt_cp = _pack_batch(bx, by_cp, dev)
            tgt_wdl = torch.tensor(by_wdl, dtype=torch.float32, device=dev)
            pred = model(flat, offs)
            loss = _objective_loss(
                pred,
                tgt_cp,
                tgt_wdl,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

        (
            tr_loss,
            tr_cp_mse,
            tr_acc,
            tr_prediction_mean_abs,
            tr_prediction_max_abs,
        ) = _eval_split(
            model,
            train_x,
            train_y_cp,
            train_y_wdl,
            batch_size,
            dev,
            loss_kind=loss_kind,
            huber_delta_cp=huber_delta_cp,
            wdl_scale_cp=wdl_scale_cp,
        )
        if val_count > 0:
            (
                va_loss,
                va_cp_mse,
                va_acc,
                va_prediction_mean_abs,
                va_prediction_max_abs,
            ) = _eval_split(
                model,
                val_x,
                val_y_cp,
                val_y_wdl,
                batch_size,
                dev,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            va_loss, va_cp_mse, va_acc = tr_loss, tr_cp_mse, tr_acc
            va_prediction_mean_abs = tr_prediction_mean_abs
            va_prediction_max_abs = tr_prediction_max_abs
        if reference_val_count > 0:
            (
                reference_va_loss,
                reference_va_cp_mse,
                reference_va_acc,
                reference_va_prediction_mean_abs,
                reference_va_prediction_max_abs,
            ) = _eval_split(
                model,
                reference_val_x,
                reference_val_y_cp,
                reference_val_y_wdl,
                batch_size,
                dev,
                loss_kind=loss_kind,
                huber_delta_cp=huber_delta_cp,
                wdl_scale_cp=wdl_scale_cp,
            )
        else:
            reference_va_loss = None
            reference_va_cp_mse = None
            reference_va_acc = None
            reference_va_prediction_mean_abs = None
            reference_va_prediction_max_abs = None
        reference_checkpoint_eligible = True
        if (
            reference_va_loss is not None
            and initial_reference_val_loss is not None
        ):
            reference_limit = float(initial_reference_val_loss) * (
                1.0
                + train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
            )
            reference_checkpoint_eligible = (
                math.isfinite(float(reference_va_loss))
                and float(reference_va_loss) <= reference_limit + 1e-12
            )
        train_loss_history.append(tr_loss)
        val_loss_history.append(va_loss)
        train_cp_mse_history.append(tr_cp_mse)
        val_cp_mse_history.append(va_cp_mse)
        train_acc_history.append(tr_acc)
        val_acc_history.append(va_acc)
        train_prediction_mean_abs_history.append(tr_prediction_mean_abs)
        val_prediction_mean_abs_history.append(va_prediction_mean_abs)
        train_prediction_max_abs_history.append(tr_prediction_max_abs)
        val_prediction_max_abs_history.append(va_prediction_max_abs)
        if reference_va_loss is not None:
            reference_val_loss_history.append(reference_va_loss)
            reference_val_cp_mse_history.append(reference_va_cp_mse)
            reference_val_acc_history.append(reference_va_acc)
            reference_val_prediction_mean_abs_history.append(
                reference_va_prediction_mean_abs
            )
            reference_val_prediction_max_abs_history.append(
                reference_va_prediction_max_abs
            )
        reference_val_checkpoint_eligible_history.append(
            reference_checkpoint_eligible
        )
        if va_loss < best_val and reference_checkpoint_eligible:
            best_val = va_loss
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = _to_cpu_tree(opt.state_dict())
            best_reference_val_loss = reference_va_loss
            best_reference_val_cp_mse = reference_va_cp_mse
            best_reference_val_acc = reference_va_acc
            best_reference_val_prediction_mean_abs = (
                reference_va_prediction_mean_abs
            )
            best_reference_val_prediction_max_abs = (
                reference_va_prediction_max_abs
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_optimizer_state is None:
        best_optimizer_state = _to_cpu_tree(opt.state_dict())

    selected_eval_x = val_x if val_count > 0 else train_x
    selected_eval_cp = val_y_cp if val_count > 0 else train_y_cp
    selected_eval_wdl = val_y_wdl if val_count > 0 else train_y_wdl
    (
        selected_val_loss,
        selected_val_cp_mse,
        selected_val_acc,
        selected_val_prediction_mean_abs,
        selected_val_prediction_max_abs,
    ) = _eval_split(
        model,
        selected_eval_x,
        selected_eval_cp,
        selected_eval_wdl,
        batch_size,
        dev,
        loss_kind=loss_kind,
        huber_delta_cp=huber_delta_cp,
        wdl_scale_cp=wdl_scale_cp,
    )
    selected_reference_val_loss = None
    selected_reference_val_cp_mse = None
    selected_reference_val_acc = None
    selected_reference_val_prediction_mean_abs = None
    selected_reference_val_prediction_max_abs = None
    if reference_val_count > 0:
        (
            selected_reference_val_loss,
            selected_reference_val_cp_mse,
            selected_reference_val_acc,
            selected_reference_val_prediction_mean_abs,
            selected_reference_val_prediction_max_abs,
        ) = _eval_split(
            model,
            reference_val_x,
            reference_val_y_cp,
            reference_val_y_wdl,
            batch_size,
            dev,
            loss_kind=loss_kind,
            huber_delta_cp=huber_delta_cp,
            wdl_scale_cp=wdl_scale_cp,
        )
        best_reference_val_loss = selected_reference_val_loss
        best_reference_val_cp_mse = selected_reference_val_cp_mse
        best_reference_val_acc = selected_reference_val_acc
        best_reference_val_prediction_mean_abs = (
            selected_reference_val_prediction_mean_abs
        )
        best_reference_val_prediction_max_abs = (
            selected_reference_val_prediction_max_abs
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer_state = _save_optimizer_state(
        best_optimizer_state,
        model,
        out_dir / "optimizer.pt",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        best_epoch=best_epoch,
        objective=objective,
    )
    emb = model.embed.weight.detach().cpu()  # [input, hidden]
    w1 = emb.transpose(0, 1).contiguous().view(-1).tolist()  # row-major [hidden][input]
    b1 = model.b1.detach().cpu().view(-1).tolist()
    w2 = model.out.weight.detach().cpu().view(-1).tolist()  # [hidden]
    b2 = float(model.out.bias.detach().cpu().item())

    checkpoint = {
        "format": "piebot-halfkp-mse-v2-torch",
        "feature_set": train_stub.FEATURE_SET,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": objective,
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
        "initialized_from": initialized_from,
        "initial_checkpoint_weights_only": initial_checkpoint_weights_only,
        "loss_kind": loss_kind,
        "huber_delta_cp": huber_delta_cp,
        "wdl_scale_cp": wdl_scale_cp,
        "min_teacher_depth": min_teacher_depth,
        "primary_sample_fraction": primary_sample_fraction,
        "teacher_sample_fraction": teacher_sample_fraction,
        "sampling_schema": train_stub.SAMPLING_SCHEMA,
        "validation_sampling_schema": train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
        "reference_validation_sampling_schema": (
            train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA
        ),
        "checkpoint_selection_schema": train_stub.CHECKPOINT_SELECTION_SCHEMA,
        "reference_validation_max_relative_loss_regression": (
            train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_hash_namespace": (
            train_stub.PRIMARY_VALIDATION_HASH_NAMESPACE
        ),
        "validation_seed": validation_seed,
        "validation_require_teacher": validation_require_teacher,
        "validation_source": validation_source,
        "optimizer_state": optimizer_state,
        "optimizer_initialized_from": optimizer_initialized_from,
        "initialized_optimizer_state": optimizer_initialized_from,
        "optimizer_state_restored": optimizer_initialized_from is not None,
    }
    metrics = {
        "train_samples": train_count,
        "val_samples": val_count,
        "input_dim": input_dim,
        "feature_set": train_stub.FEATURE_SET,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": objective,
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
        "loss_kind": loss_kind,
        "huber_delta_cp": huber_delta_cp,
        "wdl_scale_cp": wdl_scale_cp,
        "min_teacher_depth": min_teacher_depth,
        "primary_sample_fraction": primary_sample_fraction,
        "teacher_sample_fraction": teacher_sample_fraction,
        "sampling_schema": train_stub.SAMPLING_SCHEMA,
        "validation_sampling_schema": train_stub.PRIMARY_VALIDATION_SAMPLING_SCHEMA,
        "reference_validation_sampling_schema": (
            train_stub.FIXED_VALIDATION_SAMPLING_SCHEMA
        ),
        "checkpoint_selection_schema": train_stub.CHECKPOINT_SELECTION_SCHEMA,
        "reference_validation_max_relative_loss_regression": (
            train_stub.REFERENCE_VALIDATION_MAX_RELATIVE_LOSS_REGRESSION
        ),
        "primary_validation_hash_namespace": (
            train_stub.PRIMARY_VALIDATION_HASH_NAMESPACE
        ),
        "fixed_validation": fixed_validation,
        "validation_jsonl_dir": (
            Path(validation_jsonl_dir).resolve().as_posix()
            if validation_jsonl_dir is not None
            else None
        ),
        "max_validation_samples": max_validation_samples,
        "validation_seed": validation_seed,
        "validation_require_teacher": validation_require_teacher,
        "validation_source": validation_source,
        "reference_val_samples": reference_val_count,
        "train_records_with_teacher_value": train_teacher_count,
        "primary_validation_records_with_teacher_value": (
            primary_validation_teacher_count
        ),
        "primary_validation_teacher_sample_fraction": (
            float(primary_validation_teacher_count) / float(val_count)
            if val_count
            else 0.0
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "selected_val_loss": selected_val_loss,
        "selected_val_cp_mse": selected_val_cp_mse,
        "selected_val_acc": selected_val_acc,
        "selected_val_prediction_mean_abs": selected_val_prediction_mean_abs,
        "selected_val_prediction_max_abs": selected_val_prediction_max_abs,
        "selected_reference_val_loss": selected_reference_val_loss,
        "selected_reference_val_cp_mse": selected_reference_val_cp_mse,
        "selected_reference_val_acc": selected_reference_val_acc,
        "selected_reference_val_prediction_mean_abs": (
            selected_reference_val_prediction_mean_abs
        ),
        "selected_reference_val_prediction_max_abs": (
            selected_reference_val_prediction_max_abs
        ),
        "best_reference_val_loss": best_reference_val_loss,
        "best_reference_val_cp_mse": best_reference_val_cp_mse,
        "best_reference_val_acc": best_reference_val_acc,
        "best_reference_val_prediction_mean_abs": (
            best_reference_val_prediction_mean_abs
        ),
        "best_reference_val_prediction_max_abs": (
            best_reference_val_prediction_max_abs
        ),
        "initial_train_loss": initial_train_loss,
        "initial_train_cp_mse": initial_train_cp_mse,
        "initial_train_acc": initial_train_acc,
        "initial_val_loss": initial_val_loss,
        "initial_val_cp_mse": initial_val_cp_mse,
        "initial_val_acc": initial_val_acc,
        "initial_train_prediction_mean_abs": initial_train_prediction_mean_abs,
        "initial_val_prediction_mean_abs": initial_val_prediction_mean_abs,
        "initial_train_prediction_max_abs": initial_train_prediction_max_abs,
        "initial_val_prediction_max_abs": initial_val_prediction_max_abs,
        "initial_reference_val_loss": initial_reference_val_loss,
        "initial_reference_val_cp_mse": initial_reference_val_cp_mse,
        "initial_reference_val_acc": initial_reference_val_acc,
        "initial_reference_val_prediction_mean_abs": (
            initial_reference_val_prediction_mean_abs
        ),
        "initial_reference_val_prediction_max_abs": (
            initial_reference_val_prediction_max_abs
        ),
        "initialized_from": initialized_from,
        "initial_checkpoint_weights_only": initial_checkpoint_weights_only,
        "optimizer_state": optimizer_state,
        "optimizer_initialized_from": optimizer_initialized_from,
        "initialized_optimizer_state": optimizer_initialized_from,
        "optimizer_state_restored": optimizer_initialized_from is not None,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_cp_mse_history": train_cp_mse_history,
        "val_cp_mse_history": val_cp_mse_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "train_prediction_mean_abs_history": train_prediction_mean_abs_history,
        "val_prediction_mean_abs_history": val_prediction_mean_abs_history,
        "train_prediction_max_abs_history": train_prediction_max_abs_history,
        "val_prediction_max_abs_history": val_prediction_max_abs_history,
        "reference_val_loss_history": reference_val_loss_history,
        "reference_val_cp_mse_history": reference_val_cp_mse_history,
        "reference_val_acc_history": reference_val_acc_history,
        "reference_val_prediction_mean_abs_history": (
            reference_val_prediction_mean_abs_history
        ),
        "reference_val_prediction_max_abs_history": (
            reference_val_prediction_max_abs_history
        ),
        "reference_val_checkpoint_eligible_history": (
            reference_val_checkpoint_eligible_history
        ),
        "records_with_best_move": best_move_available,
        "records_with_teacher_value": teacher_value_available,
        "records_with_raw_teacher_value": raw_teacher_value_available,
        "records_total": len(xs),
        "validation_records_with_teacher_value": validation_teacher_value_available,
        "validation_records_with_raw_teacher_value": validation_raw_teacher_value_available,
        "validation_sample_sha256": validation_sample_sha256,
        "internal_validation_record_overlap": internal_validation_record_overlap,
        "actual_teacher_sample_fraction": (
            float(teacher_value_available) / float(len(xs)) if xs else 0.0
        ),
        "requested_teacher_samples": requested_teacher_samples,
        "teacher_sampling_satisfied": teacher_sampling_satisfied,
        "train_target_cp_mean_abs": (
            sum(abs(value) for value in train_y_cp) / float(len(train_y_cp))
            if train_y_cp
            else 0.0
        ),
        "train_target_cp_max_abs": max(
            (abs(value) for value in train_y_cp),
            default=0.0,
        ),
        "val_target_cp_mean_abs": (
            sum(abs(value) for value in val_y_cp) / float(len(val_y_cp))
            if val_y_cp
            else 0.0
        ),
        "val_target_cp_max_abs": max(
            (abs(value) for value in val_y_cp),
            default=0.0,
        ),
        "reference_val_target_cp_mean_abs": (
            sum(abs(value) for value in reference_val_y_cp)
            / float(len(reference_val_y_cp))
            if reference_val_y_cp
            else 0.0
        ),
        "reference_val_target_cp_max_abs": max(
            (abs(value) for value in reference_val_y_cp),
            default=0.0,
        ),
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
    ap.add_argument("--initial-checkpoint", type=Path, default=None)
    ap.add_argument(
        "--initial-checkpoint-weights-only",
        action="store_true",
        help=(
            "load only model weights for an explicit objective transition; "
            "optimizer state restore is forbidden"
        ),
    )
    ap.add_argument("--initial-optimizer-state", type=Path, default=None)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--target-cp", type=float, default=100.0)
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--max-teacher-cp", type=float, default=1500.0)
    ap.add_argument("--outcome-decay", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--loss-kind", choices=("mse", "huber", "wdl"), default="mse")
    ap.add_argument("--huber-delta-cp", type=float, default=100.0)
    ap.add_argument("--wdl-scale-cp", type=float, default=400.0)
    ap.add_argument("--min-teacher-depth", type=int, default=0)
    ap.add_argument("--primary-sample-fraction", type=float, default=0.5)
    ap.add_argument("--teacher-sample-fraction", type=float, default=0.5)
    ap.add_argument("--validation-jsonl-dir", type=Path, default=None)
    ap.add_argument("--max-validation-samples", type=int, default=100000)
    ap.add_argument("--validation-seed", type=int, default=20_260_802)
    ap.add_argument("--validation-require-teacher", action="store_true")
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
        loss_kind=args.loss_kind,
        huber_delta_cp=args.huber_delta_cp,
        wdl_scale_cp=args.wdl_scale_cp,
        min_teacher_depth=args.min_teacher_depth,
        primary_sample_fraction=args.primary_sample_fraction,
        teacher_sample_fraction=args.teacher_sample_fraction,
        validation_jsonl_dir=args.validation_jsonl_dir,
        max_validation_samples=args.max_validation_samples,
        validation_seed=args.validation_seed,
        validation_require_teacher=args.validation_require_teacher,
        seed=args.seed,
        out_dir=args.out,
        device=args.device,
        initial_checkpoint=args.initial_checkpoint,
        initial_checkpoint_weights_only=args.initial_checkpoint_weights_only,
        initial_optimizer_state=args.initial_optimizer_state,
    )
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Val samples: {metrics['val_samples']}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best val loss: {metrics['best_val_loss']:.6f}")
    print(f"Wrote: {(args.out / 'checkpoint.json').as_posix()}")
    print(f"Wrote: {(args.out / 'metrics.json').as_posix()}")
    print(f"Wrote: {(args.out / 'optimizer.pt').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
