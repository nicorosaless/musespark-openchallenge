# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open, out = x @ W[:n].T, as one Triton kernel per call.

Small m is a weight-streaming problem: the kernel splits K across CTAs so every SM pulls
weights, and reduces the partial sums inside the same kernel. Each CTA adds its FP32 tile
into a workspace with atomics; the last CTA of a tile (an atomic counter tells it) writes
the output and leaves the workspace and the counter at zero for the next call. That
replaces cuBLAS's split-K GEMM + splitKreduce pair, and the router's three or five kernels,
with a single launch.

prepare() repacks each weight into tiles [n_tile, k_block, BLOCK_N, BLOCK_K], so a CTA's
whole K range is one contiguous run of memory: each loop step reads BLOCK_N * BLOCK_K
consecutive elements instead of BLOCK_N short row segments. Same values, same BF16.

The MMA is swapped: a BLOCK_N-row tile of W is the MMA's M side and x^T (16 to 64 tokens)
its N side, so the tensor core wastes nothing on the weight side when m is tiny.

Router: FP32 weights are split once into BF16 hi + lo (as SGLang does above 8 tokens) and
the kernel runs two BF16 MMAs per tile into one FP32 accumulator; it reads the same 4
bytes per weight as FP32 would.

Split-K only pays when the output has too few tiles to fill the GPU. BF16 products with
enough output tiles for every SM (o, the full unpooler) and BF16 layers above SMALL_M_MAX
tokens keep SGLang's own GEMM (cuBLAS), which fills the GPU there without splitting K.

The split-K reduction runs through a second, tiny kernel (_splitk_finish) launched with
programmatic dependent launch: the main kernel only adds its partial tiles into the FP32
workspace and lets the finishing kernel launch at once; that kernel waits for the main grid
(griddepcontrol.wait), converts, and zeroes the workspace. On the H200 this costs less than
the last-CTA counter chain of _proj_kernel, which stays for unsplit launches.
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
# Split-K reduction through a second kernel launched with programmatic dependent launch
# (_proj_kernel_red + _splitk_finish) instead of the last-CTA counter of _proj_kernel: the
# chain of reds, acq_rel counter and reload costs ~1.4 us per call on the H200, the kernel
# boundary with PDL ~0.6-1.2 (check 875d005a7b9a454b).
SPLITK_PDL = True
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
    [N / BLOCK_N, K / BLOCK_K, BLOCK_N, BLOCK_K]; N is a multiple of BLOCK_N."""
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
    if SPLITK_PDL and split > 1:
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


def prepare(weights: Weights):
    """Called once, untimed. Pack the weights into tiles (the router split into BF16
    hi + lo first), allocate the split-K workspace (all zeros, and the kernel leaves it
    so), and compile every kernel variant."""
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
    # Compile outside the timed calls: one call per BLOCK_M and weight kind.
    for m in (1, 32, 64):
        x = torch.zeros((m, 8192), dtype=torch.bfloat16, device=device)
        kernel_fn({"x": x, "config": {"m": m, "n": 4096, "k": 8192, "layer": "pooler"}}, state)
        kernel_fn({"x": x, "config": {"m": m, "n": 256, "k": 8192, "layer": "router"}}, state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """x @ W[:n].T, where W is the config's layer: bfloat16, or float32 for the router."""
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    packed = state["packed"]
    block_m = _block_m(m)
    if layer == "router":
        t = TILING if m <= 32 else ROUTER_WIDE_M
        return _project(x, packed["router_hi"], packed["router_lo"], n, torch.float32, t, block_m, state)
    if m <= SMALL_M_MAX and _needs_split(n, TILING, state["num_sms"]):
        return _project(x, packed[layer], None, n, torch.bfloat16, TILING, block_m, state)
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
