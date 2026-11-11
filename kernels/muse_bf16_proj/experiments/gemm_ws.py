"""Warp-specialized Hopper GEMM for m = 128 / 256, narrow N and long K (lab; not a submission yet).

out[M, N] = x[M, K] @ W[N, K]^T, BF16 in, FP32 accumulation. Swap-AB as in submission.py: a
BLOCK_N-row tile of W is the wgmma M side (one warpgroup per 64 rows), BLOCK_M tokens its N side.
A one-warp producer partition streams W and x tiles by TMA into a STAGES-deep shared-memory ring
(mbarriers ready / empty); the consumer warpgroups issue asynchronous wgmma and release each stage
when its MMA is done. K is split over the grid so pooler / qkv_gate (few output tiles, long K)
fill the SMs; partial tiles go into the FP32 workspace with relaxed reds and a PDL-launched kernel
converts (as _proj_kernel_red / _splitk_finish in submission.py).
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, tma, warpgroup_mma,
                                                              warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@gluon.jit
def _ws_producer(w_desc, x_desc, w_s, x_s, ready, empty, n0, m0, kb_start, n_iter, BLOCK_K: gl.constexpr,
                 STAGES: gl.constexpr):
    nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
    for i in range(n_iter):
        s = i % STAGES
        if i >= STAGES:
            mbarrier.wait(empty.index(s), ((i // STAGES) - 1) & 1)
        mbarrier.expect(ready.index(s), nbytes)
        k = (kb_start + i) * BLOCK_K
        tma.async_load(w_desc, [n0, k], ready.index(s), w_s.index(s))
        tma.async_load(x_desc, [m0, k], ready.index(s), x_s.index(s))


@gluon.jit
def _ws_consumer(w_s, x_s, ready, empty, n_iter, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr,
                 STAGES: gl.constexpr, NUM_WARPS: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NUM_WARPS, 1],
                                                  instr_shape=[16, BLOCK_M, 16])
    acc = gl.zeros([BLOCK_N, BLOCK_M], gl.float32, layout=mma)
    for i in range(n_iter):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(w_s.index(s), x_s.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=1, deps=[acc])  # block i - 1's MMA is done
        mbarrier.arrive(empty.index((i + STAGES - 1) % STAGES), pred=i > 0)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=[acc])
    return acc


@gluon.jit(do_not_specialize=["M", "SPLIT_K"])
def _ws_gemm(w_desc, x_desc, c_desc, M, N, K, SPLIT_K, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr,
             BLOCK_K: gl.constexpr, STAGES: gl.constexpr, GDC: gl.constexpr, DIRECT: gl.constexpr):
    """DIRECT (no K split): the BF16 tile goes out by TMA store. Otherwise the FP32 partial tile is
    added into the workspace by a TMA bulk reduction (c_desc: the workspace as [M, N] FP32)."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    if GDC:
        gdc_launch_dependents()
    pid_n = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_k = gl.program_id(2)
    num_kb = K // BLOCK_K
    kb_start = pid_k * num_kb // SPLIT_K
    n_iter = (pid_k + 1) * num_kb // SPLIT_K - kb_start
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    w_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_N, BLOCK_K], w_desc.layout)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_M, BLOCK_K], x_desc.layout)
    ready = mbarrier.allocate_mbarrier(batch=STAGES)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    acc = gl.warp_specialize([
        (_ws_consumer, (w_s, x_s, ready, empty, n_iter, BLOCK_N, BLOCK_M, STAGES, NUM_WARPS)),
        (_ws_producer, (w_desc, x_desc, w_s, x_s, ready, empty, n0, m0, kb_start, n_iter, BLOCK_K, STAGES)),
    ], [1], [24])
    c_s = gl.allocate_shared_memory(c_desc.dtype, [BLOCK_M, BLOCK_N], c_desc.layout)
    c_s.store(gl.permute(acc, (1, 0)).to(c_desc.dtype))
    fence_async_shared()
    if GDC:
        gdc_wait()
    if DIRECT:
        tma.async_store(c_desc, [m0, n0], c_s)
    else:
        tma.async_atomic_add(c_desc, [m0, n0], c_s)
    tma.store_wait(0)


@gluon.jit
def _ws_finish(acc_ptr, out_ptr, MN, BLOCK: gl.constexpr, GDC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [4], [0])
    if GDC:
        gdc_wait()
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < MN
    v = gl.load(acc_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    gl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=mask)
    gl.store(acc_ptr + offs, gl.zeros_like(v), mask=mask)


_SMEM = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
_SMEM32 = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)


def ws_gemm(x, w, acc, split, block_n=64, block_m=128, block_k=64, stages=4, gdc=True):
    """x [M, K] @ w[N, K]^T -> BF16 [M, N] through the FP32 workspace acc (zeros, left zero).
    block_n = 64 per consumer warpgroup (num_warps = block_n // 16)."""
    m, k = x.shape
    n = w.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    w_desc = TensorDescriptor.from_tensor(w, [block_n, block_k], _SMEM)
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k], _SMEM)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (n // block_n, triton.cdiv(m, block_m), split)
    direct = split == 1
    if direct:
        c_desc = TensorDescriptor.from_tensor(out, [block_m, block_n], _SMEM)
    else:
        c_desc = TensorDescriptor.from_tensor(acc[:m * n].view(m, n), [block_m, block_n], _SMEM32)
    _ws_gemm[grid](w_desc, x_desc, c_desc, m, n, k, split, BLOCK_N=block_n, BLOCK_M=block_m, BLOCK_K=block_k,
                   STAGES=stages, GDC=gdc and not direct, DIRECT=direct, num_warps=block_n // 16)
    if not direct:
        kw = {"launch_pdl": True} if gdc else {}
        _ws_finish[(triton.cdiv(m * n, 512),)](acc, out, m * n, BLOCK=512, GDC=gdc, num_warps=4, **kw)
    return out
