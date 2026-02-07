"""
Simple NNUE exporter for the current Rust loader.

Format (little-endian):
- 8 bytes magic: b'PIENNUE1'
- u32 version
- u32 input_dim, u32 hidden_dim, u32 output_dim
- f32 w1[hidden_dim * input_dim]
- f32 b1[hidden_dim]
- f32 w2[output_dim * hidden_dim]
- f32 b2[output_dim]

This is a temporary dense-f32 format to bootstrap pipelines.
"""
import struct
from typing import Iterable

MAGIC = b'PIENNUE1'
Q_MAGIC = b'PIENNQ01'

def _pack_f32s(vals: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<f', float(v)) for v in vals)

def write_header_only(path: str, *, version: int = 1, input_dim: int = 0, hidden_dim: int = 0, output_dim: int = 1) -> None:
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<III', input_dim, hidden_dim, output_dim))

def write_dense_f32(
    path: str,
    *,
    version: int = 1,
    input_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    w1: Iterable[float],
    b1: Iterable[float],
    w2: Iterable[float],
    b2: Iterable[float],
) -> None:
    w1 = list(w1)
    b1 = list(b1)
    w2 = list(w2)
    b2 = list(b2)
    assert len(w1) == hidden_dim * input_dim, f"w1 size {len(w1)} != {hidden_dim}*{input_dim}"
    assert len(b1) == hidden_dim, f"b1 size {len(b1)} != {hidden_dim}"
    assert len(w2) == output_dim * hidden_dim, f"w2 size {len(w2)} != {output_dim}*{hidden_dim}"
    assert len(b2) == output_dim, f"b2 size {len(b2)} != {output_dim}"
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<III', input_dim, hidden_dim, output_dim))
        f.write(_pack_f32s(w1))
        f.write(_pack_f32s(b1))
        f.write(_pack_f32s(w2))
        f.write(_pack_f32s(b2))

def write_bias_only(
    path: str,
    *,
    input_dim: int = 12,
    hidden_dim: int = 4,
    bias: float = 0.0,
    version: int = 1,
) -> None:
    w1 = [0.0] * (hidden_dim * input_dim)
    b1 = [0.0] * hidden_dim
    w2 = [0.0] * (1 * hidden_dim)
    b2 = [float(bias)]
    write_dense_f32(
        path,
        version=version,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        w1=w1, b1=b1, w2=w2, b2=b2,
    )

if __name__ == '__main__':
    # Tiny demo files in the local working directory
    write_header_only('nnue_header_only.nnue')
    write_bias_only('nnue_bias50.nnue', bias=50.0)

def write_quant_simple(
    path: str,
    *,
    version: int = 1,
    input_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    w1_scale: float = 1.0,
    w2_scale: float = 1.0,
    w1: Iterable[int],  # int8 values in [-128,127]
    b1: Iterable[int],  # int16 values
    w2: Iterable[int],
    b2: Iterable[int],
) -> None:
    w1 = bytes([(int(v) + 256) % 256 for v in w1])
    w2 = bytes([(int(v) + 256) % 256 for v in w2])
    b1_vals = list(int(v) for v in b1)
    b2_vals = list(int(v) for v in b2)
    assert len(w1) == hidden_dim * input_dim
    assert len(b1_vals) == hidden_dim
    assert len(w2) == output_dim * hidden_dim
    assert len(b2_vals) == output_dim
    with open(path, 'wb') as f:
        f.write(Q_MAGIC)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<III', input_dim, hidden_dim, output_dim))
        f.write(struct.pack('<f', float(w1_scale)))
        f.write(struct.pack('<f', float(w2_scale)))
        f.write(w1)
        for v in b1_vals:
            f.write(struct.pack('<h', v))
        f.write(w2)
        for v in b2_vals:
            f.write(struct.pack('<h', v))
