# Emulate the planned PTX bit logic in numpy and compare with nvfp4's E2M1 grid.
import numpy as np
GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def prmt(a, b, sel):
    src = [(a >> (8*i)) & 0xFF for i in range(4)] + [(b >> (8*i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        out |= src[(sel >> (4*i)) & 0x7] << (8*i)
    return out
def bf16(bits16):
    return np.array([bits16 << 16], np.uint32).view(np.float32)[0]
bad = 0
for q in [int(x) for x in np.random.default_rng(0).integers(0, 2**32, 2000, dtype=np.uint64)]:
    vals = []
    for half, sel in ((0, 0x4140), (1, 0x4342)):
        t = prmt(q, 0, sel)
        lo = ((t << 6) & 0x01C001C0) | ((t << 12) & 0x80008000)
        hi = ((t << 2) & 0x01C001C0) | ((t << 8) & 0x80008000)
        lo &= 0xFFFFFFFF; hi &= 0xFFFFFFFF
        vals.append((lo, hi))
    for k in range(4):  # element k of the 4 packed bytes
        reg = vals[k // 2]
        sh = 16 * (k % 2)
        lo_v = bf16((reg[0] >> sh) & 0xFFFF) * 2.0**126
        hi_v = bf16((reg[1] >> sh) & 0xFFFF) * 2.0**126
        byte = (q >> (8*k)) & 0xFF
        if lo_v != GRID[byte & 0xF] or hi_v != GRID[byte >> 4]:
            bad += 1
print("mismatches", bad)

# Pair variant: output k = (lo nibble of byte k, hi nibble of byte k) as a bf16x2.
bad = 0
for q in [int(x) for x in np.random.default_rng(1).integers(0, 2**32, 2000, dtype=np.uint64)]:
    u = q >> 4
    for k in range(4):
        t = prmt(q, u, (k) | (0 << 4) | ((4 + k) << 8) | (0 << 12))
        r = (((t << 6) & 0x01C001C0) | ((t << 12) & 0x80008000)) & 0xFFFFFFFF
        lo_v = bf16(r & 0xFFFF) * 2.0**126
        hi_v = bf16(r >> 16) * 2.0**126
        byte = (q >> (8 * k)) & 0xFF
        bad += lo_v != GRID[byte & 0xF] or hi_v != GRID[byte >> 4]
print("pair mismatches", bad)
