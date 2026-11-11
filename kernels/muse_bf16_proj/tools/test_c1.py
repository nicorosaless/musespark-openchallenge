"""CPU checks of c1.py: lossless round trip on the judge's weights and edge cases, and the
two kernels in the Triton interpreter (plain-Triton decode, no griddepcontrol) against FP64.

  TRITON_INTERPRET=1 ‹local› test_c1.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from triton.runtime import interpreter

if hasattr(interpreter, "_patch_lang_tensor"):
    _patch = interpreter._patch_lang_tensor

    def _patch_lang_tensor(tensor):
        _patch(tensor)
        tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])

    interpreter._patch_lang_tensor = _patch_lang_tensor

import c1
import reference


def main():
    ok = True
    weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cpu")
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[name]
        c = c1.compress_c1(w)
        exact = c is not None and torch.equal(c1.decompress_c1(c, *w.shape).view(torch.int16), w.view(torch.int16))
        ratio = (c[0].numel() + c[1].numel() + c[2].numel() * 2) / (w.numel() * 2)
        print(f"{name:9s} round trip {'OK' if exact else 'FAIL'}  bytes x{ratio:.4f}")
        ok &= exact
    # edge cases: +-0, subnormals, a tiny value, a raised block maximum
    gen = torch.Generator().manual_seed(3)
    w = (torch.randn((128, 256), generator=gen) / 90).to(torch.bfloat16)
    b = w.view(torch.int16)
    b[0, 0], b[0, 1], b[1, 2], b[1, 3] = 0, -32768, 1, -32767
    w[2, 4] = 1e-30
    w[3, 130] = -0.25
    c = c1.compress_c1(w)
    exact = c is not None and torch.equal(c1.decompress_c1(c, 128, 256).view(torch.int16), w.view(torch.int16))
    crowded = w.clone()
    crowded[0, 10:30] = 1e-30   # 20 corrections in one row > CAP
    print(f"edge cases round trip {'OK' if exact else 'FAIL'}, overflow refused {'OK' if c1.compress_c1(crowded) is None else 'FAIL'}")
    ok &= exact and c1.compress_c1(crowded) is None
    # kernels in the interpreter
    state = {"acc": torch.zeros(1 << 16)}
    for n, k, split in ((256, 1024, 2), (128, 512, 4)):
        w = (torch.randn((n, k), generator=gen) / 30).to(torch.bfloat16)
        w[5, 7] = 1e-30   # one correction
        w[9, 300 % k] = 0.0
        w[70, k - 3] = -1e-30   # a correction in a later tile and the last K block
        x = torch.randn((1, k), generator=gen).to(torch.bfloat16)
        c = c1.compress_c1(w)
        out = c1.c1_matvec(x, c, n, k, split, state, asm=False, gdc=False, prefetch=split % 2 == 0,
                           pipe=3 if split == 4 else 1, corr_in_main=split == 4, early=split == 2)
        ref = x.double() @ w.double().T
        err = float((out.double() - ref).norm() / ref.norm())
        clean = not state["acc"].any()
        print(f"kernel n={n} k={k} split={split}: rel err {err:.1e}, workspace {'clean' if clean else 'DIRTY'}")
        ok &= err < 5e-3 and clean
    # the corrections must land: x one-hot on corrected columns, so each output is one weight
    # (dropping a correction gives d0 = 2^-12 instead of 1e-4, or d0 instead of 0)
    w = (torch.randn((128, 512), generator=gen) / 30).to(torch.bfloat16)
    w[5, 0], w[5, 7] = 8.0, 1e-4
    w[70, 300], w[70, 509] = -8.0, 0.0
    c = c1.compress_c1(w)
    no_corr = (c[0], c[1], torch.zeros_like(c[2]))
    for col, val in ((7, 1.0), (509, 2.0)):
        x = torch.zeros((1, 512)).to(torch.bfloat16)
        x[0, col] = val
        ref = x.double() @ w.double().T
        for split, kw in ((1, {}), (2, {"early": True}), (4, {"corr_in_main": True}), (2, {"corr_in_main": True})):
            good = []
            for comp in (c, no_corr):
                out = c1.c1_matvec(x, comp, 128, 512, split, state, asm=False, gdc=False, **kw)
                good.append(torch.allclose(out.double(), ref, rtol=1e-2, atol=1e-30))
            print(f"corrections col={col} split={split} {kw}: {'OK' if good[0] else 'FAIL'}, "
                  f"without the table {'caught' if not good[1] else 'NOT CAUGHT'}")
            ok &= good[0] and not good[1]
    # stream-K: segments that start and end inside tiles, several tiles per CTA, empty CTAs
    for n, k, ctas in ((256, 1024, 5), (128, 512, 3), (192, 256, 7), (64, 256, 16)):
        w = (torch.randn((n, k), generator=gen) / 30).to(torch.bfloat16)
        w[5, 7] = 1e-30
        x = torch.randn((1, k), generator=gen).to(torch.bfloat16)
        c = c1.compress_c1(w)
        st = dict(state, num_sms=1)
        out = c1.c1sk_matvec(x, c, n, k, st, ctas_per_sm=ctas, asm=False, gdc=False)
        ref = x.double() @ w.double().T
        err = float((out.double() - ref).norm() / ref.norm())
        clean = not state["acc"].any()
        print(f"stream-K n={n} k={k} ctas={ctas}: rel err {err:.1e}, workspace {'clean' if clean else 'DIRTY'}")
        ok &= err < 5e-3 and clean
    for n, k, split in ((256, 1024, 2), (128, 512, 4)):
        w = (torch.randn((n, k), generator=gen) / 30).to(torch.bfloat16)
        w[5, 7] = 1e-30
        w[9, 300 % k] = 0.0
        x = torch.randn((1, k), generator=gen).to(torch.bfloat16)
        c = c1.compress_c1(w)
        out = c1.c1w_matvec(x, c, n, k, split, state, asm=False, gdc=False, pipe=3 if split == 4 else 1)
        ref = x.double() @ w.double().T
        err = float((out.double() - ref).norm() / ref.norm())
        clean = not state["acc"].any()
        print(f"word kernel n={n} k={k} split={split}: rel err {err:.1e}, workspace {'clean' if clean else 'DIRTY'}")
        ok &= err < 5e-3 and clean
    for n, k, split, m in ((256, 1024, 2, 1), (128, 512, 4, 13), (128, 512, 2, 32)):
        w = (torch.randn((n, k), generator=gen) / 30).to(torch.bfloat16)
        w[5, 7] = 1e-30
        x = torch.randn((m, k), generator=gen).to(torch.bfloat16)
        c = c1.compress_c1(w)
        out = c1.c1_matmul(x, c, n, k, split, state, block_m=16 if m <= 16 else 32, asm=False, gdc=False, dot_fp32=True,
                           pipe=3 if m == 13 else 1)
        ref = x.double() @ w.double().T
        err = float((out.double() - ref).norm() / ref.norm())
        clean = not state["acc"].any()
        print(f"MMA kernel n={n} k={k} m={m} split={split}: rel err {err:.1e}, workspace {'clean' if clean else 'DIRTY'}")
        ok &= err < 5e-3 and clean
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
