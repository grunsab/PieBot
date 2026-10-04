"""Rebuild a float arch-v2 checkpoint from a PIENNQ02 playing net.

A recovery tool for when the float checkpoint is lost and only the quantised
net survives. The result evaluates identically to the playing net (exporting
it reproduces the same file), but its weights sit on the quantisation grid:
w1 and b1 in steps of 1/QA, w2 in steps of 1/QB. It carries no optimizer
state, so it can only seed a weights-only start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from . import features_v2, train_stub

MAGIC = b"PIENNQ02"
_HEADER = struct.Struct("<8sIIIIiii")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_from_quant(quant_path: Path, *, objective: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    quant_path = Path(quant_path)
    data = quant_path.read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError("quantised net is truncated")
    magic, _version, input_dim, hidden_dim, output_dim, qa, qb, scale = _HEADER.unpack_from(data)
    if magic != MAGIC or output_dim != 1 or qa <= 0 or qb <= 0:
        raise ValueError("not a single-output PIENNQ02 net")
    w1_bytes, b1_bytes, w2_bytes = 2 * input_dim * hidden_dim, 2 * hidden_dim, 2 * hidden_dim
    if len(data) != _HEADER.size + w1_bytes + b1_bytes + w2_bytes + 4:
        raise ValueError("quantised net size does not match its header")
    offset = _HEADER.size
    w1_q = np.frombuffer(data, dtype="<i2", count=input_dim * hidden_dim, offset=offset)
    offset += w1_bytes
    b1_q = np.frombuffer(data, dtype="<i2", count=hidden_dim, offset=offset)
    offset += b1_bytes
    w2_q = np.frombuffer(data, dtype="<i1", count=2 * hidden_dim, offset=offset)
    offset += w2_bytes
    (b2_q,) = struct.unpack_from("<i", data, offset)
    # The file is feature-major [input][hidden]; checkpoints are [hidden][input].
    w1 = (w1_q.reshape(input_dim, hidden_dim).T.astype(np.float64) / qa).ravel()
    return {
        "format": "piebot-halfkp-dp-screlu-v1-torch",
        "feature_set": features_v2.FEATURE_SET_V2,
        "target_schema": train_stub.TARGET_SCHEMA,
        "objective": dict(objective),
        "arch": "v2", "quant_qa": int(qa), "quant_qb": int(qb),
        "input_dim": int(input_dim), "hidden_dim": int(hidden_dim),
        "wdl_scale_cp": float(scale),
        "w1": w1.tolist(),
        "b1": (b1_q.astype(np.float64) / qa).tolist(),
        "w2": (w2_q.astype(np.float64) / qb).tolist(),
        "b2": b2_q / (qa * qa * qb),
        "recovered_from_quant": {"sha256": hashlib.sha256(data).hexdigest(),
                                 "note": "weights lie on the quantisation grid; no optimizer state"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective-json", required=True,
                        help="Objective metadata to record, as written by the trainer")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"refusing to overwrite {args.out}")
    checkpoint = checkpoint_from_quant(args.quant, objective=json.loads(args.objective_json))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(checkpoint, stream)
    print(json.dumps({"out": str(args.out), "sha256": sha256(args.out),
                      "quant_sha256": checkpoint["recovered_from_quant"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
