"""Build a hidden-128 PIENNQ01 net that computes the same function as the h64 source.

Duplicate the 64 hidden units (w1 rows, b1), duplicate w2, double b2, halve w2_scale.
Output is bit-identical evals up to sub-centipawn f32 rounding, so search trees match
and any NPS delta is pure arithmetic cost.
"""
import struct, sys

src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
assert raw[:8] == b"PIENNQ01", raw[:8]
ver, = struct.unpack_from("<I", raw, 8)
inp, hid, out = struct.unpack_from("<III", raw, 12)
w1s, w2s = struct.unpack_from("<ff", raw, 24)
off = 32
w1 = raw[off:off + hid * inp]; off += hid * inp
b1 = raw[off:off + 2 * hid]; off += 2 * hid
w2 = raw[off:off + out * hid]; off += out * hid
b2 = list(struct.unpack_from(f"<{out}h", raw, off))
assert out == 1 and off + 2 * out == len(raw), (out, off, len(raw))

b2d = b2[0] * 2
assert -32768 <= b2d <= 32767, f"b2 doubling overflows: {b2[0]}"

with open(dst, "wb") as f:
    f.write(b"PIENNQ01")
    f.write(struct.pack("<I", ver))
    f.write(struct.pack("<III", inp, hid * 2, out))
    f.write(struct.pack("<ff", w1s, w2s / 2.0))
    f.write(w1); f.write(w1)          # hidden-major rows duplicated
    f.write(b1); f.write(b1)
    f.write(w2); f.write(w2)
    f.write(struct.pack("<h", b2d))

print(f"src: input={inp} hidden={hid} w1_scale={w1s} w2_scale={w2s} b2={b2[0]}")
print(f"dst: hidden={hid*2} w2_scale={w2s/2.0} b2={b2d}")
