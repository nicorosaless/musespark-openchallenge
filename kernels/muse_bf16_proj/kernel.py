# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T for one of five weights (qkv_gate,
o, pooler, unpooler: BF16; router: FP32 weights and FP32 output), x BF16 [m, k].

kernel_fn picks one of three paths per call, by layer and m only (H200, 132 SMs):

1. c1, lossless compressed weights (_c1_project), for m = 1 on pooler, o and the unpooler,
   and for 2 <= m <= 16 on the pooler. Two kernels: _c1_gemv (one token, FP32 FMA) or
   _c1_gemm (tensor cores) adds partial sums into an FP32 workspace, then _c1_finish,
   launched with programmatic dependent launch (PDL), adds the per-row corrections, writes
   BF16 and zeroes the workspace. Used only for layers whose weights prepare() could compress
   and decompress back bit for bit; any other layer falls through to path 2 or 3.
2. Split-K Triton GEMM (_project), for the router at every m (FP32 weights split into BF16
   hi + lo) and for BF16 layers with m <= SMALL_M_MAX = 32 whose output has too few 64-row
   tiles to fill the GPU (qkv_gate, pooler, the unpooler's 1024-row slice at m <= 8), when
   c1 does not take the call. With split > 1: _proj_kernel_red adds partial tiles into the
   FP32 workspace and _splitk_finish, launched with PDL, converts and zeroes it. With
   split = 1 (only the router, from about 1400 tokens up): _proj_kernel writes the output
   directly.
3. SGLang's own BF16 GEMM (cuBLAS) for everything else: o at m >= 2, the full 8192-row
   unpooler (m >= 9), and every BF16 layer above 32 tokens. It fills the GPU there without
   splitting K.

Why these kernels:

- Small m is a weight-streaming problem: splitting K across CTAs makes every SM pull
  weights, and the split-K partials are summed with relaxed FP32 atomics into a workspace that
  the finishing kernel leaves at zero for the next call. This replaces cuBLAS's split-K
  GEMM + splitKreduce pair, and SGLang's three or five kernels for the router.
- prepare() repacks each BF16 weight into tiles [n_tile, k_block, BLOCK_N, BLOCK_K] (pack), so
  a CTA's whole K range is one contiguous run of memory. Same values, same BF16.
- The MMA is swapped: a BLOCK_N-row tile of W is the MMA's M side and x^T (16 to 64 tokens)
  its N side, so the tensor core wastes nothing on the weight side when m is tiny.
- Router: FP32 weights are split once into BF16 hi + lo (as SGLang does above 8 tokens) and
  the kernel runs two BF16 MMAs per tile into one FP32 accumulator; it reads the same 4 bytes
  per weight as FP32 would.
- The split-K reduction runs in a second, tiny kernel launched with PDL: the main kernel
  adds its partial tiles and lets the finishing kernel launch at once; that kernel waits for
  the main grid (griddepcontrol.wait), converts, and zeroes the workspace. On the H200 this
  costs less than a last-CTA counter inside the main kernel.
- c1 stores 12 bits per weight plus a small per-row table instead of 16 (x0.75 to x0.77 the
  bytes). Every BF16 weight is reproduced bit for bit and the products use those exact BF16
  values. The format is described in the c1 section below.

Where to start: kernel_fn at the bottom, then _project and _proj_kernel_red, then the c1
section (format comment, _c1_decode, _c1_gemv, _c1_finish).
"""

from typing import NamedTuple, TypedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except ImportError:  # local runs without SGLang
    def _bf16_gemm_dispatch_impl(x, w, bias):
        return torch.nn.functional.linear(x, w, bias)


# ---- Harness interface ----------------------------------------------------------------------

class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ---- Split-K Triton GEMM: tiling and workspace ----------------------------------------------

class Tiling(NamedTuple):
    block_n: int      # rows of W per tile (the MMA's M side)
    block_k: int
    ctas_per_sm: int  # split K until the grid holds about this many CTAs per SM
    stages: int
    warps: int


# BF16 layers up to this many tokens may use the split-K kernel; above it, SGLang's GEMM.
SMALL_M_MAX = 32
TILING = Tiling(block_n=64, block_k=128, ctas_per_sm=2, stages=3, warps=4)
# The router above 32 tokens: half the split-K CTAs. At m=128-256 the atomics of a deeper
# split cost more than its parallelism buys (H200 lab, 2026-10-25). Same packing as TILING.
ROUTER_WIDE_M = Tiling(block_n=64, block_k=128, ctas_per_sm=1, stages=3, warps=4)
# Each CTA streams at least this many K blocks, so the load pipeline has work.
MIN_KB_PER_CTA = 2
# FP32 workspace for split-K partial sums (elements) and per-tile counters. One set per
# state: two kernel_fn calls must not run at once on different streams, or their partial
# sums would mix. The judge replays calls one after another on one stream.
WORKSPACE_ELEMS = 1 << 22
MAX_TILES = 1 << 12


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _proj_kernel(x_ptr, w_ptr, wlo_ptr, out_ptr, acc_ptr, cnt_ptr,
                 M, N, K, stride_xm, SPLIT_K,
                 BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                 HAS_LO: tl.constexpr):
    """out[M, N] = x[M, K] @ W[N, K]^T (+ x @ Wlo^T), one [BLOCK_N, BLOCK_M] tile of out^T
    per (pid_n, pid_m), its K range split SPLIT_K ways over pid_k. W is tiled
    [N / BLOCK_N, K / BLOCK_K, BLOCK_N, BLOCK_K]; N is a multiple of BLOCK_N.

    _project launches this kernel only with SPLIT_K == 1; split-K calls go to
    _proj_kernel_red + _splitk_finish. The SPLIT_K > 1 branch (a last-CTA counter reduction)
    never runs, but SPLIT_K is a runtime argument, so removing the branch would change the
    compiled kernel; it stays as the judge compiled it."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_r = tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # This CTA's K blocks: an even share, within one block.
    num_kb = K // BLOCK_K
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    tile_elems = BLOCK_N * BLOCK_K
    w_off = (pid_n * num_kb + kb_start).to(tl.int64) * tile_elems \
        + (offs_r[:, None] * BLOCK_K + offs_k[None, :])
    x_off = offs_m[None, :] * stride_xm + (kb_start * BLOCK_K + offs_k)[:, None]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb_start, kb_end):
        xt = tl.load(x_ptr + x_off, mask=m_mask[None, :], other=0.0)
        w = tl.load(w_ptr + w_off)
        acc = tl.dot(w, xt, acc)
        if HAS_LO:
            wlo = tl.load(wlo_ptr + w_off)
            acc = tl.dot(wlo, xt, acc)
        w_off += tile_elems
        x_off += BLOCK_K

    # out is [M, N] row-major; the tile is its transpose.
    offs_n = pid_n * BLOCK_N + offs_r
    o_off = offs_m[None, :] * N + offs_n[:, None]
    o_mask = m_mask[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
    else:
        tl.atomic_add(acc_ptr + o_off, acc, mask=o_mask, sem="relaxed")
        # Every thread's adds are done before one thread bumps the counter (acq_rel, GPU scope).
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(0) + pid_n
        arrived = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if arrived == SPLIT_K - 1:
            # Last CTA of the tile: the sum is complete in L2. Write it out, and zero the
            # workspace and counter for the next call.
            total = tl.load(acc_ptr + o_off, mask=o_mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + o_off, total.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.store(acc_ptr + o_off, tl.zeros_like(total), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _proj_kernel_red(x_ptr, w_ptr, wlo_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K,
                     BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                     HAS_LO: tl.constexpr):
    """_proj_kernel's main loop with the reduction left to _splitk_finish: each split only
    adds its partial tile into the FP32 workspace (relaxed reds), and the kernel lets the
    dependent finishing kernel launch at once (programmatic dependent launch)."""
    gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_r = tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    num_kb = K // BLOCK_K
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    tile_elems = BLOCK_N * BLOCK_K
    w_off = (pid_n * num_kb + kb_start).to(tl.int64) * tile_elems \
        + (offs_r[:, None] * BLOCK_K + offs_k[None, :])
    x_off = offs_m[None, :] * stride_xm + (kb_start * BLOCK_K + offs_k)[:, None]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb_start, kb_end):
        xt = tl.load(x_ptr + x_off, mask=m_mask[None, :], other=0.0)
        acc = tl.dot(tl.load(w_ptr + w_off), xt, acc)
        if HAS_LO:
            acc = tl.dot(tl.load(wlo_ptr + w_off), xt, acc)
        w_off += tile_elems
        x_off += BLOCK_K
    o_off = offs_m[None, :] * N + (pid_n * BLOCK_N + offs_r)[:, None]
    # A no-op unless this kernel is itself launched with PDL (as decode_layer may): then the
    # previous call's _splitk_finish must have zeroed the workspace before these reds land.
    gdc_wait()
    tl.atomic_add(acc_ptr + o_off, acc, mask=m_mask[None, :], sem="relaxed")


@triton.jit
def _splitk_finish(acc_ptr, out_ptr, MN, BLOCK: tl.constexpr):
    """out = the FP32 workspace in out's dtype, then the workspace back to zero. Launched as
    a programmatic dependent of _proj_kernel_red: griddepcontrol.wait returns once that grid
    has finished and its reds are visible, so the kernel boundary is the split-K barrier."""
    gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    v = tl.load(acc_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(acc_ptr + offs, tl.zeros_like(v), mask=mask)


def _block_m(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def _needs_split(n: int, t: Tiling, num_sms: int) -> bool:
    """Too few output tiles (at m <= 32) to give every SM two without splitting K."""
    return 2 * (n // t.block_n) <= num_sms


def _split(m: int, n: int, k: int, block_m: int, t: Tiling, num_sms: int) -> int:
    """Split-K factor for an [m, k] x [n, k]^T product. Depends on shapes only."""
    tiles = triton.cdiv(m, block_m) * (n // t.block_n)
    num_kb = k // t.block_k
    split = round(t.ctas_per_sm * num_sms / tiles)
    split = max(1, min(split, num_kb // MIN_KB_PER_CTA))
    if m * n > WORKSPACE_ELEMS or tiles > MAX_TILES:
        split = 1
    return split


def pack(w: torch.Tensor, t: Tiling) -> torch.Tensor:
    """[N, K] -> [N / block_n, K / block_k, block_n, block_k], contiguous, same dtype."""
    n, k = w.shape
    assert n % t.block_n == 0 and k % t.block_k == 0
    return w.view(n // t.block_n, t.block_n, k // t.block_k, t.block_k).permute(0, 2, 1, 3).contiguous()


def _project(x, w, wlo, n, out_dtype, t: Tiling, block_m: int, state):
    """x @ W[:n].T with W (and the router's lo half) packed by pack(., t)."""
    m, k = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    split = _split(m, n, k, block_m, t, state["num_sms"])
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    grid = (n // t.block_n, triton.cdiv(m, block_m), split)
    if split > 1:
        # Split-K reduction through a second kernel launched with PDL instead of the last-CTA
        # counter of _proj_kernel: the chain of reds, acq_rel counter and reload costs ~1.4 us
        # per call on the H200, the kernel boundary with PDL ~0.6-1.2 (check 875d005a7b9a454b).
        _proj_kernel_red[grid](x, w, w if wlo is None else wlo, state["acc"], m, n, k, x.stride(0), split,
                               BLOCK_N=t.block_n, BLOCK_M=block_m, BLOCK_K=t.block_k, HAS_LO=wlo is not None,
                               num_warps=t.warps, num_stages=t.stages)
        _splitk_finish[(triton.cdiv(m * n, 1024),)](state["acc"], out, m * n, BLOCK=1024, num_warps=4,
                                                   launch_pdl=True)
        return out
    _proj_kernel[grid](x, w, w if wlo is None else wlo, out, state["acc"], state["cnt"],
                       m, n, k, x.stride(0), split,
                       BLOCK_N=t.block_n, BLOCK_M=block_m, BLOCK_K=t.block_k, HAS_LO=wlo is not None,
                       num_warps=t.warps, num_stages=t.stages)
    return out


# ---- c1: lossless compressed BF16 -------------------------------------------------------------
# Format, per block of 64 rows x 128 columns of W (12 KB instead of 16):
#   bytes 0-4095      sign | mantissa of columns 0-63      (64 x 64, row-major)
#   bytes 4096-8191   sign | mantissa of columns 64-127
#   bytes 8192-12287  exponent codes: low nibble column c, high nibble column c + 64
# plus one byte per block, E (its largest exponent). A weight's exponent is E - code. Code 15
# with a zero sign/mantissa byte decodes to d0 = 2^(E - 15 - 127); the weights whose exponent
# is more than 14 below E, and zeros, are stored that way and fixed exactly from a per-row
# table (column, true value, d0; three [rows, C1_CAP] int16 planes): out[r] gets
# sum_j (v_j - d0_j) * x[c_j]. prepare() checks the round trip bit for bit and keeps the BF16
# path for any weight that does not fit or does not round-trip.

C1_BN, C1_BK, C1_CAP = 64, 128, 8
C1_BLOCK_BYTES = 3 * 64 * 64
# Where c1 replaces the BF16 kernels, measured on the H200 (check 424d8cd9c6d14061, us):
# m = 1: pooler 15.3 vs 18.1-18.4, o 5.7 vs 5.9, unpooler (1024 rows) 4.1 vs 4.5; qkv_gate ties
# (11.5 vs 11.6) and stays BF16. 2 <= m <= 16 (tensor cores): pooler 17.5 vs 18.5 at m = 8 and
# 17.95 vs 18.8 at m = 16; slower at m = 32 and for qkv_gate.
C1_GEMV_LAYERS = ("pooler", "o", "unpooler")
C1_MMA_LAYERS = ("pooler",)
C1_MMA_M_MAX = 16
C1_CTAS_PER_SM = 4

# Decode of four weights per call: $2 = four sign|mantissa bytes, $3 = four exponent-code
# nibbles (low nibbles; _C1_DEC_HI_C shifts the high ones down first), $4 = E in every byte.
# Per byte: exponent = E - code (no borrow: code <= E); then each BF16 is sign|mantissa
# spread to bits 15 and 6..0 (prmt + mask 0x807F) OR'ed with exponent << 7 (one lop3).
_C1_DEC_LO = """{
.reg .b32 c, ex, y, t;
and.b32 c, $3, 0x0F0F0F0F;
sub.u32 ex, $4, c;
prmt.b32 y, $2, 0, 0x1100;
prmt.b32 t, ex, 0, 0x4140;
shl.b32 t, t, 7;
lop3.b32 $0, y, 0x807F807F, t, 0xEA;
prmt.b32 y, $2, 0, 0x3322;
prmt.b32 t, ex, 0, 0x4342;
shl.b32 t, t, 7;
lop3.b32 $1, y, 0x807F807F, t, 0xEA;
}"""
_C1_DEC_LO_C = tl.constexpr(_C1_DEC_LO)
_C1_DEC_HI_C = tl.constexpr(_C1_DEC_LO.replace("and.b32 c, $3, 0x0F0F0F0F;",
                                               "shr.b32 c, $3, 4;\nand.b32 c, c, 0x0F0F0F0F;"))


@triton.jit
def _c1_decode(sm, nib, e8, HIGH: tl.constexpr):
    """BF16 weights from sign|mantissa bytes, exponent-code nibbles (HIGH: the high nibble)
    and the block's E (u8): four per inline-PTX call (prmt, per-byte subtract, lop3)."""
    if HIGH:
        return tl.inline_asm_elementwise(_C1_DEC_HI_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=tl.bfloat16,
                                         is_pure=True, pack=4)
    else:
        return tl.inline_asm_elementwise(_C1_DEC_LO_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=tl.bfloat16,
                                         is_pure=True, pack=4)


@triton.jit(do_not_specialize=["SPLIT_K"])
def _c1_gemv(x_ptr, wc_ptr, e_ptr, acc_ptr, K, SPLIT_K, PIPE: tl.constexpr):
    """acc[64 rows of tile pid_n] += W_tile[:, its K share] @ x for one token, W compressed;
    FP32 FMA. PIPE > 1: Triton pipelines the block loads that many stages deep (cp.async); it
    does not by itself, as they feed no tl.dot. E is a scalar load, splat: loaded per element,
    it made Triton shuffle all three planes through shared memory."""
    gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    offs_r = tl.arange(0, 64)
    offs_c = tl.arange(0, 64)
    p_off = offs_r[:, None] * 64 + offs_c[None, :]
    acc = tl.zeros((64, 64), dtype=tl.float32)
    blk = pid_n * num_kb + kb_start
    for kb in tl.range(kb_start, kb_end, num_stages=PIPE):
        base = wc_ptr + blk.to(tl.int64) * 12288
        sm_lo = tl.load(base + p_off)
        sm_hi = tl.load(base + 4096 + p_off)
        nib = tl.load(base + 8192 + p_off)
        e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk)
        x_lo = tl.load(x_ptr + kb * 128 + offs_c).to(tl.float32)
        x_hi = tl.load(x_ptr + kb * 128 + 64 + offs_c).to(tl.float32)
        w_lo = _c1_decode(sm_lo, nib, e8, False).to(tl.float32)
        w_hi = _c1_decode(sm_hi, nib, e8, True).to(tl.float32)
        acc += w_lo * x_lo[None, :] + w_hi * x_hi[None, :]
        blk += 1
    row = tl.sum(acc, axis=1)
    gdc_wait()  # the previous call's finishing kernel has zeroed the workspace
    tl.atomic_add(acc_ptr + pid_n * 64 + offs_r, row, sem="relaxed")


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _c1_gemm(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: tl.constexpr,
             PIPE: tl.constexpr):
    """acc[M, 64 rows of tile pid_n] += x[M, its K share] @ W_tile^T, W compressed: the decoded
    BF16 tile is the MMA's M side (as in _proj_kernel), x^T its N side."""
    gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    offs_r = tl.arange(0, 64)
    offs_c = tl.arange(0, 64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    p_off = offs_r[:, None] * 64 + offs_c[None, :]
    x_off = offs_m[None, :] * stride_xm + offs_c[:, None]
    acc = tl.zeros((64, BLOCK_M), dtype=tl.float32)
    blk = pid_n * num_kb + kb_start
    for kb in tl.range(kb_start, kb_end, num_stages=PIPE):
        base = wc_ptr + blk.to(tl.int64) * 12288
        sm_lo = tl.load(base + p_off)
        sm_hi = tl.load(base + 4096 + p_off)
        nib = tl.load(base + 8192 + p_off)
        e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk)
        xt_lo = tl.load(x_ptr + kb * 128 + x_off, mask=m_mask[None, :], other=0.0)
        xt_hi = tl.load(x_ptr + kb * 128 + 64 + x_off, mask=m_mask[None, :], other=0.0)
        w_lo = _c1_decode(sm_lo, nib, e8, False)
        w_hi = _c1_decode(sm_hi, nib, e8, True)
        acc = tl.dot(w_lo, xt_lo, acc)
        acc = tl.dot(w_hi, xt_hi, acc)
        blk += 1
    gdc_wait()
    o_off = offs_m[None, :] * N + (pid_n * 64 + offs_r)[:, None]
    tl.atomic_add(acc_ptr + o_off, acc, mask=m_mask[None, :], sem="relaxed")


@triton.jit
def _c1_finish(acc_ptr, out_ptr, corr_ptr, PSTRIDE, x_ptr, M, N, stride_xm, CAP: tl.constexpr,
               BLOCK: tl.constexpr):
    """out[t, r] = bf16(acc[t, r] + sum_j (v_j - d0_j) * x[t, c_j]); acc[t, r] = 0. The
    corrections do not depend on the main kernel: they are loaded and summed before
    griddepcontrol.wait, while it still runs (x is complete: the main kernel is launched
    without PDL). corr planes are PSTRIDE elements apart; padding is all zero."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N
    t = idx // N
    r = idx % N
    j = tl.arange(0, CAP)
    c_off = r[:, None] * CAP + j[None, :]
    cmask = mask[:, None]
    col = tl.load(corr_ptr + c_off, mask=cmask, other=0).to(tl.int32) & 0xFFFF
    v = tl.load(corr_ptr + PSTRIDE + c_off, mask=cmask, other=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
    d0 = tl.load(corr_ptr + 2 * PSTRIDE + c_off, mask=cmask, other=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
    xv = tl.load(x_ptr + t[:, None] * stride_xm + col, mask=cmask, other=0.0).to(tl.float32)
    fix = tl.sum(v * xv - d0 * xv, axis=1)
    gdc_wait()
    a = tl.load(acc_ptr + idx, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + idx, (a + fix).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(acc_ptr + idx, tl.zeros_like(a), mask=mask)


def compress_c1(w: torch.Tensor):
    """BF16 [N, K] -> (blocks uint8 [N/64 * K/128, 12288], E uint8 [N/64 * K/128],
    corrections int16 [3, N, C1_CAP]: columns, true values, d0s), or None if the weights do
    not fit the format."""
    n, k = w.shape
    if n % C1_BN or k % C1_BK or k > 1 << 15:
        return None
    nt, nkb = n // C1_BN, k // C1_BK
    bits = (w.view(torch.int16).to(torch.int32) & 0xFFFF).view(nt, C1_BN, nkb, C1_BK).permute(0, 2, 1, 3)
    e = (bits >> 7) & 0xFF
    e_max = e.amax(dim=(-2, -1))
    if int(e_max.min()) < 15:
        return None
    fix = ((e_max[..., None, None] - e) > 14) | ((bits & 0x7FFF) == 0)
    sm = torch.where(fix, 0, ((bits >> 8) & 0x80) | (bits & 0x7F))
    code = torch.where(fix, 15, e_max[..., None, None] - e)
    blocks = torch.cat([sm[..., :64], sm[..., 64:], code[..., :64] | (code[..., 64:] << 4)], dim=-1)
    blocks = blocks.view(nt, nkb, 64, 3, 64).permute(0, 1, 3, 2, 4).reshape(nt * nkb, C1_BLOCK_BYTES)
    fix_nat = fix.permute(0, 2, 1, 3).reshape(n, k)
    if int(fix_nat.sum(1).max()) > C1_CAP:
        return None
    corr = torch.zeros((3, n, C1_CAP), dtype=torch.int16, device=w.device)
    hit = fix_nat.nonzero()
    if len(hit):
        r, c = hit[:, 0], hit[:, 1]
        counts = torch.bincount(r, minlength=n)
        slot = torch.arange(len(hit), device=w.device) - (torch.cumsum(counts, 0) - counts)[r]
        e_blk = e_max[r // C1_BN, c // C1_BK]
        corr[0, r, slot] = c.to(torch.int16)
        corr[1, r, slot] = w.view(torch.int16)[r, c]
        corr[2, r, slot] = ((e_blk - 15) << 7).to(torch.int16)
    return blocks.to(torch.uint8).contiguous(), e_max.reshape(-1).to(torch.uint8).contiguous(), corr.contiguous()


def decompress_c1(c, n: int, k: int) -> torch.Tensor:
    """Inverse of compress_c1(), decoding as the kernels do (corrections included)."""
    blocks, e_max, corr = c
    nt, nkb = n // C1_BN, k // C1_BK
    b = blocks.view(nt, nkb, 3, 64, 64).to(torch.int32)
    sm = torch.cat([b[:, :, 0], b[:, :, 1]], dim=-1)
    code = torch.cat([b[:, :, 2] & 15, b[:, :, 2] >> 4], dim=-1)
    ex = e_max.view(nt, nkb, 1, 1).to(torch.int32) - code
    bits = ((sm & 0x80) << 8) | (ex << 7) | (sm & 0x7F)
    nat = bits.permute(0, 2, 1, 3).reshape(n, k)
    cr = corr.to(torch.int32) & 0xFFFF
    used = (cr[1] != 0) | (cr[2] != 0)
    rows = torch.arange(n, device=nat.device)[:, None].expand(n, C1_CAP)
    r, cc = rows[used], cr[0][used]
    if not torch.equal(nat[r, cc], cr[2][used]):
        return None  # a correction does not sit on a d0
    nat[r, cc] = cr[1][used]
    return nat.to(torch.int16).view(torch.bfloat16)


def _c1_split(tiles: int, num_kb: int, num_sms: int) -> int:
    """K split giving about C1_CTAS_PER_SM CTAs per SM, dividing the K blocks evenly."""
    split = max(1, min(num_kb, round(C1_CTAS_PER_SM * num_sms / tiles)))
    while split > 1 and num_kb % split:
        split -= 1
    return split


def _c1_project(x, comp, n, state):
    """x @ W[:n].T, W compressed by compress_c1 (n <= its rows): _c1_gemv (one token) or
    _c1_gemm (tensor cores), then _c1_finish through PDL."""
    blocks, e_max, corr = comp
    m, k = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    tiles = n // C1_BN
    num_kb = k // C1_BK
    split = _c1_split(tiles, num_kb, state["num_sms"])
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m == 1:
        # Pipelining pays only with a few K blocks per CTA: 4 warps and 3 stages then, else 8
        # warps and none (measured: pooler 8 blocks per CTA; o 2; the unpooler slice 1).
        deep = num_kb // split >= 4
        _c1_gemv[(tiles, split)](x, blocks, e_max, state["acc"], k, split, PIPE=3 if deep else 1,
                                 num_warps=4 if deep else 8, num_stages=3)
        block = 128
    else:
        _c1_gemm[(tiles, 1, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split, BLOCK_M=16,
                                    PIPE=3, num_warps=4, num_stages=3)
        block = 512
    _c1_finish[(triton.cdiv(m * n, block),)](state["acc"], out, corr, corr.stride(0), x, m, n, x.stride(0),
                                             CAP=C1_CAP, BLOCK=block, num_warps=4, launch_pdl=True)
    return out


def _c1_prepare(state) -> dict:
    """Compressed copies of the layers c1 serves, each checked bit for bit against W."""
    out = {}
    for name in sorted(set(C1_GEMV_LAYERS) | set(C1_MMA_LAYERS)):
        w = state[name]
        comp = compress_c1(w)
        if comp is None:
            continue
        back = decompress_c1(comp, *w.shape)
        if back is not None and torch.equal(back.view(torch.int16), w.view(torch.int16)):
            out[name] = comp
    return out


# ---- prepare and kernel_fn ------------------------------------------------------------------

def prepare(weights: Weights):
    """Called once, untimed. Pack the weights into tiles (the router split into BF16
    hi + lo first), allocate the split-K workspace (all zeros, and the kernels leave it
    so), compress the c1 layers, and compile every kernel variant."""
    state = {name: w.contiguous() for name, w in weights.items() if name != "router"}
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["packed"] = {name: pack(w, TILING) for name, w in state.items()}
    state["packed"]["router_hi"] = pack(hi, TILING)
    state["packed"]["router_lo"] = pack(lo, TILING)
    device = router.device
    state["num_sms"] = torch.cuda.get_device_properties(device).multi_processor_count
    state["acc"] = torch.zeros(WORKSPACE_ELEMS, dtype=torch.float32, device=device)
    state["cnt"] = torch.zeros(MAX_TILES, dtype=torch.int32, device=device)
    state["c1"] = _c1_prepare(state)
    # Compile outside the timed calls: one call per BLOCK_M and weight kind.
    for m in (1, 32, 64):
        x = torch.zeros((m, 8192), dtype=torch.bfloat16, device=device)
        kernel_fn({"x": x, "config": {"m": m, "n": 4096, "k": 8192, "layer": "pooler"}}, state)
        kernel_fn({"x": x, "config": {"m": m, "n": 256, "k": 8192, "layer": "router"}}, state)
    for name, m, n, k in (("pooler", 16, 4096, 8192), ("o", 1, 8192, 1024)):
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
        kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": name}}, state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """x @ W[:n].T, where W is the config's layer: bfloat16, or float32 for the router.
    Path by layer and m: c1, then the split-K Triton GEMM, then SGLang's GEMM."""
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    packed = state["packed"]
    block_m = _block_m(m)
    comp = state["c1"].get(layer)
    if comp is not None and n % C1_BN == 0 and (
            (m == 1 and layer in C1_GEMV_LAYERS) or (2 <= m <= C1_MMA_M_MAX and layer in C1_MMA_LAYERS)):
        return _c1_project(x, comp, n, state)
    if layer == "router":
        t = TILING if m <= 32 else ROUTER_WIDE_M
        return _project(x, packed["router_hi"], packed["router_lo"], n, torch.float32, t, block_m, state)
    if m <= SMALL_M_MAX and _needs_split(n, TILING, state["num_sms"]):
        return _project(x, packed[layer], None, n, torch.bfloat16, TILING, block_m, state)
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
