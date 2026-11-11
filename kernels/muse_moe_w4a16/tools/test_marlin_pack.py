"""The CUDA GEMM's weight layout, checked on CPU: what each lane decodes (marlin_pack's
NumPy emulation of the kernel's bit operations) must equal nvfp4.dequantize at the k the
lane's fragment position stands for.

    ‹local› test_marlin_pack.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlin_pack as mp  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402


def main():
    gen = torch.Generator().manual_seed(7)
    e, r, k = 2, 32, 1024
    w, sf, _ = random_weights(r, k, generator=gen, device="cpu", leading=(e,))
    words = mp.pack_codes(w).numpy()              # [E, R/16, K/64, 32, 4]
    scales = mp.pack_scales(mp.recode_scales(sf.view(torch.uint8))).numpy()  # [E, R/16, K/512, 32, 8, 2]
    ref = dequantize(w, sf, 1.0).numpy()          # [E, R, K]
    worst = 0.0
    for ei in range(e):
        for t in range(r // 16):
            for ch in range(k // 64):
                got = mp.emulate_tile(words[ei, t, ch], scales[ei, t, ch // 8, :, ch % 8])
                want = ref[ei, 16 * t:16 * t + 16, 64 * ch:64 * ch + 64]
                assert not np.isnan(got).any(), "a fragment position was never written"
                worst = max(worst, float(np.abs(got - want).max()))
    print(f"max |decoded - dequantize| = {worst}")
    assert worst == 0.0
    print("ok")


if __name__ == "__main__":
    main()
