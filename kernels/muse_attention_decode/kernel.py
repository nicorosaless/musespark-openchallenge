# okc: problem=muse_attention_decode
"""Gated attention decode for Muse Spark on H200: one decode step (q_len 1) or one 8-token
speculative verify block over a paged FP8 E4M3 KV cache, 16 query heads and 2 KV heads (TP8).

What kernel_fn computes, per request b and block token j at position p:
    q, k, v, gate = split(qkv[t])            # the fused projection output, bfloat16
    q = rmsnorm(q);  k = rmsnorm(k)           # QK-norm, no weight
    q, k = rope(q, p), rope(k, p)             # sliding-window layers only
    cache[cache_loc[t]] = e4m3(k), e4m3(v)    # written in place
    a = softmax(q K^T * sm_scale * qk_scale_factor) V   # keys up to p: a 2,048-key window or all
    out = rmsnorm(a) * sigmoid(gate)          # bfloat16 [B, L, Hq * 64]

Every path is split-KV. Keys are cut into 64-key blocks counted back from the newest token, and
the blocks into `splits` contiguous ranges, one per CTA. Block 0 holds the step's own tokens and
belongs to exactly one split: that split writes the new keys and values to the cache and reads
them back (from the cache after a barrier, or from shared memory), so no other CTA depends on the
write. Each split produces a normalized partial output and its base-2 log-sum-exp; a combine
merges them and applies the norm-gate. The split count depends only on shapes and on the compiled
kernel's occupancy (`num_splits`, `_ctas_per_sm`, `_cufa_ctas`), never on input values, so the
call can be captured in a CUDA graph.

Kernels in launch order, by shape range (TUNING, kernel_fn):
- Sliding decode, batch <= 2: one CUDA kernel, `full_fa2` built with CL = 16. A cluster of 16
  CTAs per (request, KV head) takes the 16 splits, merges them through distributed shared memory
  and stores the gated output. No combine kernel.
- Sliding decode, batch > 2: the Triton `_attn_kernel` (both KV heads of a pair packed into one
  block-diagonal 16-row tile, one masked key loop), then the Triton `_combine_kernel`, launched
  with PDL so it loads the gates while kernel A drains. With one split, `_attn_kernel` applies
  the norm-gate itself and there is no combine.
- Full decode: CUDA `full_fa2` (both KV heads per CTA, a cp.async ring with the page-table entries
  2 * STAGES - 1 blocks ahead, four warps splitting each key block), then `_combine_kernel`.
- Verify (q_len 8), sliding and full: CUDA `verify_ws` (warp-specialized: one producer warp
  gathers K/V rows with cp.async and mbarriers, one consumer warpgroup runs QK^T with wgmma),
  then `_combine_kernel`.
- If a CUDA kernel does not compile or load (or the GPU is not Hopper), the same step runs on the
  Triton `_attn_kernel` with the Triton tuning of its TUNING entry. The CUDA kernels are compiled
  in-process with NVRTC on the first call of each shape (a warm-up, outside graph capture) and
  launched with cuLaunchKernelEx.

Precision. The CUDA kernels round q (unit RMS after the norm) to E4M3, as FA3 does, and run
Q K^T on the cache's E4M3 keys as stored, with the score scale applied to S. The Triton kernel
does the same on verify tiles (64 rows); on decode tiles (16 rows, mma.sync) it keeps q in FP16
with the scale folded in and converts the keys to FP16 (exact). Everywhere P is FP16, V goes from
E4M3 to FP16 (exact), and accumulation is FP32. Cache scales are read from their tensors, the
page size from config, and RoPE's cos and sin from a table built from config.

Where to start: kernel_fn (the dispatch), then KERNEL_A_CUDA_SOURCE (full_fa2) for decode and
KERNEL_A_WS_CUDA_SOURCE (verify_ws) for verify. The Triton kernels are the readable reference
version of the same split-KV scheme. MAP.md lists every symbol with its lines.
"""

import ctypes
import os
from typing import TypedDict

import torch
import triton
import triton.language as tl

try:  # programmatic dependent launch, Triton >= 3.4 (the judge has 3.8)
    from triton.language.extra.cuda import gdc
except ImportError:
    gdc = None


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


# ---- Constants and tuning ----

RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
MAX_SPLITS = 128      # most splits per (request, KV-head group); each one costs the combine a step
CUDA_STAGES = 3       # cp.async pipeline depth of full_fa2 (tiles in flight + 1)
CUDA_WS_STAGES = 4    # the same, for verify_ws
CUDA_DECODE_MINB = 1  # decode CTAs per SM full_fa2's registers must fit (__launch_bounds__)
WS_KVP = 1            # KV heads per verify_ws CTA (one consumer warpgroup each)
COMBINE_SP = 128      # combine: splits merged per step (fewer serial L2 round trips with many splits)
COMBINE_WARPS = 4     # ... and its warps when there are more than 32 splits

# Tuning by shape range, measured on the bench (H100) and confirmed with H200 checks
# (evidence/runs/muse_attention_decode/). Layer "sliding" (a window) or "full"; tile "decode" (L x Hq/Hkv <= 16
# rows per KV head) or "verify" (64 rows per KV head).
#   cuda            kernel A in CUDA: "fa2" (full_fa2), "cl" (full_fa2 with clusters: the whole
#                   step in one kernel, a cluster of `cl` CTAs per (request, KV head) merging its
#                   splits through DSMEM, then the norm-gate) or "ws" (verify_ws)
#   cl              cluster size for cuda="cl" (the split count; 9-16 need non-portable clusters)
#   cuda_max_batch  the CUDA kernel A only up to this many requests, the Triton kernel above
#   cuda_kvp        KV heads per full_fa2 CTA on decode tiles (default 2, the pair)
# The rest is the Triton kernel A's tuning (and block_n, min_chunk, per_sm, waves for every path):
#   block_n   keys per tile; min_chunk: fewest keys a split gets
#   edges     masks only on the edge key blocks (a masked copy of the tile code per edge), else
#             one masked loop over every block
#   stages    software-pipeline stages of the key loop
#   maxnreg   register cap (ptxas .maxnreg). With 4 warps, 128 registers fit 4 CTAs per SM and
#             137 only 3, which spills a full layer's programs into one more wave
#   per_sm    sliding layers: programs per SM the split count aims for (one short wave)
#   waves     full layers: the split count fills this many whole waves of resident CTAs (SMs x
#             CTAs per SM, from the compiled kernel), so no last wave runs part-empty
TUNING = {
    ("sliding", "decode"): dict(cuda="cl", cl=16, cuda_max_batch=2, cuda_kvp=1, block_n=64, min_chunk=128,
                                edges=False, stages=3, maxnreg=None, warps=4, per_sm=2),
    ("full", "decode"): dict(cuda="fa2", block_n=64, min_chunk=128, edges=False, stages=3, maxnreg=128,
                             warps=4, waves=1),
    ("sliding", "verify"): dict(cuda="ws", block_n=64, min_chunk=128, edges=True, stages=2, maxnreg=None,
                                warps=4, per_sm=2),
    ("full", "verify"): dict(cuda="ws", block_n=64, min_chunk=128, edges=True, stages=2, maxnreg=None,
                             warps=4, waves=1),
}
ON_GPU = os.environ.get("TRITON_INTERPRET") != "1"  # off in the CPU interpreter (local logic tests)
# The combine kernel launches while kernel A drains and waits on griddepcontrol.wait.
PDL = tl.constexpr(ON_GPU and gdc is not None)


# ---- Triton kernel A (split-KV attention) and the combine kernel ----

@triton.jit
def _rms(x, D: tl.constexpr, eps):
    """Row-wise RMSNorm of a [R, D] float32 tile, no weight."""
    return x * tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None]


@triton.jit
def _load_heads(qkv_ptr, pos_ptr, rope_ptr, tok, col, part, rows_ok, eps,
                QKV_STRIDE: tl.constexpr, D: tl.constexpr, W: tl.constexpr, ROPE: tl.constexpr):
    """RMSNorm (and interleaved RoPE) of head slices of the fused projection as [R, W] rows:
    row i holds token tok[i]'s columns col[i] .. col[i] + D at columns part[i] * D .. + D, and
    zeros elsewhere (W = D, or W = PACK * D for a block-diagonal tile of packed heads)."""
    c = tl.arange(0, W)
    dd = c % D
    keep = rows_ok[:, None] & ((c // D)[None, :] == part[:, None])
    base = qkv_ptr + tok[:, None].to(tl.int64) * QKV_STRIDE + col[:, None]
    x = tl.load(base + dd[None, :], mask=keep, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None]
    x = x * inv_rms
    if ROPE:
        # Pairs (2i, 2i+1) rotate by pos * inv_freq[i]: even' = e c - o s, odd' = o c + e s.
        swapped = tl.load(base + (dd ^ 1)[None, :], mask=keep, other=0.0).to(tl.float32) * inv_rms
        # cos and sin of pos * inv_freq[i] come from a table (rope_ptr: [positions, D], cos | sin):
        # one gather instead of a range-reduced cos and sin per element.
        pos = tl.load(pos_ptr + tok, mask=rows_ok, other=0).to(tl.int64)
        at = rope_ptr + pos[:, None] * D + (dd // 2)[None, :]
        cos = tl.load(at, mask=keep, other=0.0)
        sin = tl.load(at + D // 2, mask=keep, other=0.0)
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        x = x * cos + sign[None, :] * swapped * sin
    return x


@triton.jit
def _attend(q, k, v, allowed, m, l, acc, s_scale, QK_FP8: tl.constexpr, MASKED: tl.constexpr):
    """One online-softmax step (base 2) of a [M, W] query tile over a [N, W] key/value tile.
    With QK_FP8, q and k are E4M3 and the scores take the query scale s_scale afterwards.
    Unmasked tiles (every key visible to every row) skip the mask and the all-masked guard."""
    s = tl.dot(q, tl.trans(k))
    if QK_FP8:
        s = s * s_scale
    if MASKED:
        s = tl.where(allowed, s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, 1))
    m_safe = m_new
    if MASKED:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m - m_safe)
    p = tl.exp2(s - m_safe[:, None])
    l = l * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l, acc


@triton.jit
def _block(i, hi, lo, b, kv0, p_row, q, m, l, acc, q_scale, k_cache_ptr, v_cache_ptr, page_table_ptr, pt_stride,
           PAGE: tl.constexpr, WINDOW: tl.constexpr, ROW: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
           BLOCK_N: tl.constexpr, QK_FP8: tl.constexpr, MASKED: tl.constexpr):
    """Key block i (the BLOCK_N keys ending i * BLOCK_N before hi): gather its page-table entries
    and its key and value rows, then one _attend step. MASKED blocks apply the causal and window
    masks and the block's clip at lo."""
    q_dtype: tl.constexpr = q.dtype
    n = tl.arange(0, BLOCK_N)
    c = tl.arange(0, W)
    t = hi - (i + 1) * BLOCK_N + n
    allowed = 0  # unused on unmasked blocks
    if MASKED:
        t_ok = t >= lo
        page = tl.load(page_table_ptr + b * pt_stride + t // PAGE, mask=t_ok, other=0)
        allowed = t_ok[None, :] & (t[None, :] <= p_row[:, None])
        if WINDOW > 0:
            allowed = allowed & (t[None, :] > p_row[:, None] - WINDOW)
    else:
        t_ok = n < BLOCK_N
        page = tl.load(page_table_ptr + b * pt_stride + t // PAGE)
    row = (page.to(tl.int64) * PAGE + t % PAGE) * ROW + kv0 * D
    k = tl.load(k_cache_ptr + row[:, None] + c[None, :], mask=t_ok[:, None], other=0.0).to(q_dtype)
    v = tl.load(v_cache_ptr + row[:, None] + c[None, :], mask=t_ok[:, None], other=0.0).to(tl.float16)
    return _attend(q, k, v, allowed, m, l, acc, q_scale, QK_FP8, MASKED)


@triton.jit
def _gate(tok, head, rows_ok, qkv_ptr, QKV_STRIDE: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
          D: tl.constexpr):
    """sigmoid(gate logits) for [R, D] rows (token tok[i], query head head[i]), float32."""
    d = tl.arange(0, D)
    gate = tl.load(qkv_ptr + tok[:, None].to(tl.int64) * QKV_STRIDE + ((HQ + 2 * HKV) * D + head * D)[:, None]
                   + d[None, :], mask=rows_ok[:, None], other=0.0).to(tl.float32)
    return tl.sigmoid(gate)


@triton.jit
def _norm_gate_store(o, tok, head, rows_ok, qkv_ptr, out_ptr, eps,
                     QKV_STRIDE: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    """rmsnorm(o) * sigmoid(gate) for [R, D] rows (token tok[i], query head head[i]), stored
    as bfloat16."""
    _store_gated(o, _gate(tok, head, rows_ok, qkv_ptr, QKV_STRIDE, HQ, HKV, D), tok, head, rows_ok, out_ptr, eps,
                 HQ, D)


@triton.jit
def _store_gated(o, gate, tok, head, rows_ok, out_ptr, eps, HQ: tl.constexpr, D: tl.constexpr):
    """rmsnorm(o) * gate (already through the sigmoid), stored as bfloat16."""
    d = tl.arange(0, D)
    o = _rms(o, D, eps)
    y = o * gate
    tl.store(out_ptr + tok[:, None].to(tl.int64) * (HQ * D) + (head * D)[:, None] + d[None, :],
             y.to(tl.bfloat16), mask=rows_ok[:, None])


@triton.jit
def _finish(acc, l, m, v_scale, s, num_splits, tok, head, part, rows_ok, qkv_ptr, out_ptr, ws_o_ptr, ws_lse_ptr,
            eps, QKV_STRIDE: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
            PACK: tl.constexpr, ONE_SPLIT: tl.constexpr):
    """Kernel A's epilogue for one tl.dot group: the output itself with one split, else this
    split's normalized output and log-sum-exp (base 2) for kernel B. Empty splits store -inf
    and 0."""
    has = l > 0
    o = tl.where(has[:, None], acc / l[:, None], 0.0) * v_scale
    if PACK > 1:
        # Row i's own head is columns part[i] * D .. + D of the packed tile.
        M: tl.constexpr = o.shape[0]
        c = tl.arange(0, PACK * D)
        o = tl.where((c // D)[None, :] == part[:, None], o, 0.0)
        o = tl.sum(tl.reshape(o, [M, PACK, D]), 1)
    if ONE_SPLIT:
        _norm_gate_store(o, tok, head, rows_ok, qkv_ptr, out_ptr, eps, QKV_STRIDE, HQ, HKV, D)
    else:
        d = tl.arange(0, D)
        slot = (tok.to(tl.int64) * HQ + head) * num_splits + s
        tl.store(ws_o_ptr + slot[:, None] * D + d[None, :], o, mask=rows_ok[:, None])
        tl.store(ws_lse_ptr + slot, tl.where(has, m + tl.log2(l), float("-inf")), mask=rows_ok)


@triton.jit(do_not_specialize=["num_splits"])  # one binary for every split count (see _ctas_per_sm)
def _attn_kernel(qkv_ptr, pos_ptr, loc_ptr, seq_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
                 page_table_ptr, rope_ptr, out_ptr, ws_o_ptr, ws_lse_ptr,
                 pt_stride, num_splits, q_mult, eps,
                 L: tl.constexpr, LP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 PAGE: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr, KVP: tl.constexpr,
                 M: tl.constexpr, BLOCK_N: tl.constexpr, ONE_SPLIT: tl.constexpr,
                 QK_FP8: tl.constexpr, EARLY_TRIGGER: tl.constexpr, EDGES: tl.constexpr):
    """Kernel A, grid (split, KV-head group, request). A program owns KVP KV heads (decode tiles:
    the pair, packed block-diagonally into one [M, KVP * D] tile; verify tiles: one head and 64
    rows) and the keys of its split. With ONE_SPLIT it stores the gated output, else its split's
    partial for _combine_kernel."""
    if PDL and EARLY_TRIGGER:
        # Let the combine kernel launch now and load its gates while this grid runs; it still
        # waits for this grid to finish before reading the partials (griddepcontrol.wait).
        gdc.gdc_launch_dependents()
    split = tl.program_id(0)
    kv0 = tl.program_id(1) * KVP
    b = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    R: tl.constexpr = L * G  # query rows per KV head
    W: tl.constexpr = KVP * D  # columns of the packed tile
    QKV_STRIDE: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW: tl.constexpr = HKV * D  # cache bytes per token
    d = tl.arange(0, D)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)

    seq = tl.load(seq_ptr + b)
    hi = seq + L  # keys 0 .. hi - 1, the block's own tokens included
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    else:
        lo = 0

    # Split s takes blocks [n_blocks - (s + 1) * per_split, n_blocks - s * per_split), clipped at 0,
    # so the split holding block 0 (the newest keys, the step's own tokens among them) gets the
    # remainder, the fewest blocks.
    n_blocks = tl.cdiv(hi - lo, BLOCK_N)
    per_split = tl.cdiv(n_blocks, num_splits)
    last = n_blocks - split * per_split
    first = tl.maximum(last - per_split, 0)
    # The tile packs KVP KV heads block-diagonally into [M, KVP * D]. Row r is head part = r // R
    # of the pack, its token (r % R) // G and query head (r % R) % G. Rows past KVP * R pad the
    # tile to the 16 rows tl.dot needs (they repeat real rows and are never stored).
    r = tl.arange(0, M)
    rows_ok = r < KVP * R
    part = tl.minimum(r // R, KVP - 1)
    jr = (r % R) // G
    tok = b * L + jr
    p_row = seq + jr
    head0 = (kv0 + part) * G + (r % R) % G
    q_scale = q_mult * k_scale  # sm_scale * qk_scale_factor * log2(e), and the key scale
    if QK_FP8:  # unit-RMS q to E4M3, as the reference's FP8 variant rounds it; the scale goes on S
        q_dtype: tl.constexpr = tl.float8e4nv
        q_mul = 1.0
    else:
        q_dtype: tl.constexpr = tl.float16
        q_mul = q_scale
    q0 = (_load_heads(qkv_ptr, pos_ptr, rope_ptr, tok, head0 * D, part, rows_ok, eps, QKV_STRIDE, D, W, ROPE)
          * q_mul).to(q_dtype)
    m0 = tl.full([M], float("-inf"), tl.float32)
    l0 = tl.zeros([M], tl.float32)
    acc0 = tl.zeros([M, W], tl.float32)

    # The split holding block 0 writes the new keys and values, after the queries are loaded so
    # both sets of loads can be in flight together; the barrier makes the writes visible to its
    # own loop, which reads them back as cache rows.
    if (first == 0) & (last > 0):
        j = tl.arange(0, LP)
        j_ok = j < L
        tok_w = b * L + j
        loc = tl.load(loc_ptr + tok_w, mask=j_ok, other=0)
        for h in tl.static_range(KVP):
            kvh = kv0 + h
            k_new = _load_heads(qkv_ptr, pos_ptr, rope_ptr, tok_w, tl.zeros_like(j) + (HQ + kvh) * D,
                                tl.zeros_like(j), j_ok, eps, QKV_STRIDE, D, D, ROPE)
            dst = loc[:, None] * ROW + kvh * D + d[None, :]
            tl.store(k_cache_ptr + dst, (k_new / k_scale).to(tl.float8e4nv), mask=j_ok[:, None])
            v_new = tl.load(qkv_ptr + tok_w[:, None].to(tl.int64) * QKV_STRIDE + (HQ + HKV + kvh) * D + d[None, :],
                            mask=j_ok[:, None], other=0.0).to(tl.float32)
            tl.store(v_cache_ptr + dst, (v_new / v_scale).to(tl.float8e4nv), mask=j_ok[:, None])
        tl.debug_barrier()

    if EDGES:
        # Only the edge blocks need masks: block 0 (causal inside the step's own tokens) and the
        # oldest EDGE blocks (clipped at lo; on windows, each block row's window starts up to L - 1
        # keys later, which can spill into the second-oldest block). The rest run unmasked.
        EDGE: tl.constexpr = 2 if WINDOW > 0 else 1
        inner_lo = tl.maximum(first, 1)
        inner_hi = tl.minimum(last, n_blocks - EDGE)
        for i in range(inner_lo, inner_hi):
            m0, l0, acc0 = _block(i, hi, lo, b, kv0, p_row, q0, m0, l0, acc0, q_scale, k_cache_ptr, v_cache_ptr,
                                  page_table_ptr, pt_stride, PAGE, WINDOW, ROW, D, W, BLOCK_N, QK_FP8, False)
        for e in tl.static_range(EDGE + 1):
            # e = 0: block 0; e >= 1: block n_blocks - e, unless block 0 or inside the unmasked range.
            if e == 0:
                i = 0
                take = (first == 0) & (last > 0)  # empty splits (last <= 0) have first == 0 too
            else:
                i = n_blocks - e
                take = (i >= tl.maximum(first, 1)) & (i < last) & (i >= inner_hi)
            if take:
                m0, l0, acc0 = _block(i, hi, lo, b, kv0, p_row, q0, m0, l0, acc0, q_scale, k_cache_ptr,
                                      v_cache_ptr, page_table_ptr, pt_stride, PAGE, WINDOW, ROW, D, W, BLOCK_N,
                                      QK_FP8, True)
    else:
        # One masked loop over every block: less code (one copy of the tile), for short loops.
        for i in range(first, last):
            m0, l0, acc0 = _block(i, hi, lo, b, kv0, p_row, q0, m0, l0, acc0, q_scale, k_cache_ptr, v_cache_ptr,
                                  page_table_ptr, pt_stride, PAGE, WINDOW, ROW, D, W, BLOCK_N, QK_FP8, True)

    _finish(acc0, l0, m0, v_scale, split, num_splits, tok, head0, part, rows_ok, qkv_ptr, out_ptr, ws_o_ptr,
            ws_lse_ptr, eps, QKV_STRIDE, HQ, HKV, D, KVP, ONE_SPLIT)
    if PDL and not EARLY_TRIGGER:
        gdc.gdc_launch_dependents()


@triton.jit
def _combine_kernel(ws_o_ptr, ws_lse_ptr, qkv_ptr, out_ptr, num_splits, eps,
                    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HB: tl.constexpr, SP: tl.constexpr):
    """Merges the splits of HB query heads of one token, SP splits at a time, then the
    norm-gate."""
    tok = tl.program_id(0)
    head = tl.program_id(1) * HB + tl.arange(0, HB)
    d = tl.arange(0, D)
    toks = tl.zeros([HB], tl.int32) + tok
    # The gate doesn't depend on kernel A: load it before waiting on kernel A's partials.
    gate = _gate(toks, head, head < HQ, qkv_ptr, (2 * HQ + 2 * HKV) * D, HQ, HKV, D)
    if PDL:
        gdc.gdc_wait()  # kernel A's partial outputs
    base = (tok.to(tl.int64) * HQ + head) * num_splits  # [HB]
    m = tl.full([HB], float("-inf"), tl.float32)
    den = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, D], tl.float32)
    for s0 in range(0, num_splits, SP):
        sp = s0 + tl.arange(0, SP)
        sp_ok = (sp < num_splits)[None, :]
        slot = base[:, None] + sp[None, :]  # [HB, SP]
        lse = tl.load(ws_lse_ptr + slot, mask=sp_ok, other=float("-inf"))
        o = tl.load(ws_o_ptr + slot[:, :, None] * D + d[None, None, :], mask=sp_ok[:, :, None], other=0.0)
        m_new = tl.maximum(m, tl.max(lse, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)  # a chunk of empty splits only
        scale = tl.exp2(m - m_safe)
        w = tl.exp2(lse - m_safe[:, None])
        den = den * scale + tl.sum(w, 1)
        acc = acc * scale[:, None] + tl.sum(o * w[:, :, None], 1)
        m = m_new
    _store_gated(acc / den[:, None], gate, toks, head, head < HQ, out_ptr, eps, HQ, D)


# ---- CUDA kernel A: sources ----
# Nsight Compute on the Triton kernel A showed each key block paying the page-table load and then
# the key/value copies in series: Triton's pipeliner cannot overlap a gather whose addresses come
# from another load. The CUDA kernels issue the page-table entries blocks ahead into a ring and
# fill a ring of raw E4M3 K/V tiles with cp.async. They are compiled in-process with NVRTC
# (cuda.bindings, in the judge's image; no subprocess) for sm_90a and launched with cuLaunchKernelEx.
# Readable copies: cuda/full_fa2.cu and cuda/verify_ws.cu (generated from this file).

_CUDA_FUNCS = {}
_CUDA_FAILED = set()
_CUDA_CTAS = {}


def _cuda_include_dirs():
    """CUDA headers (cuda_fp8.h, mma.h, cooperative_groups.h, CCCL): nvidia/cu13 in the judge's
    image (a namespace package, so every path of it is tried), else the toolkit."""
    candidates = [os.environ["OKC_CUDA_INCLUDE"]] if "OKC_CUDA_INCLUDE" in os.environ else []  # local override
    try:
        import nvidia.cu13 as cu13
        candidates += [os.path.join(p, "include") for p in cu13.__path__]
    except ImportError:
        pass
    candidates += ["/usr/local/cuda/include", "/usr/local/cuda/targets/x86_64-linux/include"]
    for base in candidates:
        if os.path.exists(os.path.join(base, "crt", "mma.h")):
            return [base, os.path.join(base, "cccl")]
    raise RuntimeError("no CUDA headers found")


def _cuda_check(result):
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA/NVRTC: {err}")
    return rest[0] if len(rest) == 1 else rest


# full_fa2: decode tiles (full layers; with CL, sliding layers at batch <= 2).
KERNEL_A_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

// full_fa2: split-KV attention for decode tiles (one token, 8 query heads per KV head), sm_90a.
// -D from _cufa_consts: HQ HKV L KVP WARP_M WARP_K STAGES PAGE WINDOW ROPE MINB (CTAs per SM the
// registers must fit), KVEARLY=1 EPIW=1 (always), and on sliding decode CL NEWSMEM=1 EARLYQ=1.
// Each warp takes BN / WARP_K keys of every block; QK^T with mma.sync m16n8k32 on E4M3, P V with
// m16n8k8 on FP16 (V converted from E4M3 in registers). Without CL the CTA writes its split's
// normalized partial and log-sum-exp to ws_o / ws_lse for the combine kernel.
#define D 64
#define BN 64                          // keys per block
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define NWARPS (WARP_M * WARP_K)
#define NT (NWARPS * 32)
#define DIMS (KVP * D)
#define QP (DIMS + 16)                 // Q rows (E4M3 bytes), padded
#define RP (DIMS + 16)                 // raw E4M3 rows (bytes), padded
#define ROWS (WARP_M * 16)
#define PRING (2 * STAGES)             // page-table ring slots
#define NEG_INF __int_as_float(0xff800000)
// CL > 0: the whole step in one kernel. A cluster of CL CTAs per (request, KV group) takes the CL
// splits; each keeps its normalized partial in shared memory, and after a cluster barrier every CTA
// merges a few output rows through DSMEM, then the norm-gate and the bfloat16 store (no combine).
#ifndef CL
#define CL 0
#endif
// NEWSMEM 1: the split holding block 0 also writes the step's new keys and values (E4M3) straight
// into its stage-0 tile and skips their copies, so it neither fences nor reads them back.
#ifndef NEWSMEM
#define NEWSMEM 0
#endif
// EARLYQ 1: the queries' loads go first (they don't depend on seq_lens), then the page entries,
// and the block-0 split writes the step's keys and values last, overlapped with the rest.
#ifndef EARLYQ
#define EARLYQ 0
#endif
// _cufa_consts sets NEWSMEM and EARLYQ together with CL, never alone.
static_assert(NEWSMEM == EARLYQ && EARLYQ == (CL > 0), "NEWSMEM and EARLYQ come with CL");
static_assert(KVEARLY == 1 && EPIW == 1, "the only compiled variant");

#define OFF_Q 0
#define OFF_K (OFF_Q + ROWS * QP)
#define OFF_V (OFF_K + STAGES * BN * RP)
#define OFF_P (OFF_V + STAGES * BN * RP)
// CL: the partials every rank sends to the owner of each merge row (rho on rank rho % CL, slot
// rho / CL): [MROWS][CL][D + 1] floats (o, then the lse), apart from everything the main loop uses.
#define MROWS (CL ? (KVP * L * G + CL - 1) / (CL ? CL : 1) : 0)
#define OFF_MG (OFF_P + PRING * BN * 4)
// (the host computes the same total in _cufa_smem)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mma_k8(float* d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
// D += A B over k = 32, A and B in E4M3.
__device__ __forceinline__ void mma_k32_e4m3(float* d, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// cp.async: 16 and 4 bytes, global to shared.
// cp.async goes through the generic proxy, so the __syncthreads after cp.async.wait_group is
// enough; a TMA (async proxy) version would also need fence.proxy.async before reusing a stage.
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ float2 norm_rope(float2 x, float2 cs, float eps) {
    float inv = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + eps);
    x.x *= inv; x.y *= inv;
    return make_float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

extern "C" __global__ void __launch_bounds__(NT, MINB) full_fa2(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps,
    __nv_bfloat16* __restrict__ out, int* __restrict__ cnt)   // out: CL only; cnt: unused
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [ROWS][QP] E4M3, dims permuted (below)
    unsigned char* Kr = smem + OFF_K;   // [STAGES][BN][RP] E4M3
    unsigned char* Vr = smem + OFF_V;
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = blockIdx.x, kv0 = blockIdx.y * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
#if EARLYQ
    // Queries meanwhile: RMSNorm, RoPE, E4M3; decode packs the pair block-diagonally.
    // All of a warp's row loads go out before any is used (one latency, not one per row).
    constexpr int QR = (ROWS + NWARPS - 1) / NWARPS;
    float2 qx[QR], qcs[QR];
    long qpos[QR];
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
    // The step's own keys and values (written by the block-0 split only, but loaded by every CTA:
    // seq_lens isn't known yet), so their chain runs beside the queries' instead of after it.
    constexpr int KR = (KVP * L + NWARPS - 1) / NWARPS;
    float2 kxs[KR], vxs[KR], kcs[KR];
    long kpos[KR], kloc[KR];
#pragma unroll
    for (int i = 0; i < KR; ++i) {
        const int w = warp + i * NWARPS, h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
        kxs[i] = vxs[i] = make_float2(0.f, 0.f);
        kpos[i] = kloc[i] = 0;
        if (w < KVP * L) {
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            kpos[i] = positions[tok];
            kloc[i] = cache_loc[tok];
        }
    }
#endif
    const float k_scale = *k_scale_ptr, v_scale = *v_scale_ptr;
    const int seq = seq_lens[b], hi = seq + L;
    const int lo = WINDOW > 0 ? max(seq - WINDOW + 1, 0) : 0;
    // Blocks counted back from hi: block i holds keys [hi - (i + 1) BN, hi - i BN), clipped at lo.
    // Split s takes blocks [n - (s + 1) per, n - s per), clipped at 0 (as the Triton kernel).
    const int n_blocks = (hi - lo + BN - 1) / BN;
    const int per = (n_blocks + num_splits - 1) / num_splits;
    const int last = n_blocks - split * per, first = max(last - per, 0);
    const int nb = max(last - first, 0);

#if CL
    // Every rank has started before any writes into its shared memory: arrive now, wait before the
    // first remote store (by then long satisfied).
    cooperative_groups::this_cluster().barrier_arrive();
#endif
#if CL
    // Gates of this warp's merge rows (see the merge below; the rank is the split), loaded now: the
    // merge needs them last, after a DRAM latency nobody waits on.
    constexpr int GR = (KVP * L * G + CL * NWARPS - 1) / (CL * NWARPS);
    float2 gate_r[GR];
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = split + CL * (warp + NWARPS * i);
        gate_r[i] = make_float2(0.f, 0.f);
        if (rho < KVP * L * G) {
            const int h = rho / (L * G), rr = rho % (L * G), j = rr / G, head = (kv0 + h) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + j) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if !EARLYQ
    // The split holding block 0 writes the step's new keys and values to the cache first; its own
    // copies of block 0 then read them back (after the fence and the barrier below).
    if (first == 0 && last > 0) {
        for (int w = warp; w < KVP * L; w += NWARPS) {
            int h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
            long pos = positions[tok], loc = cache_loc[tok];
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            float2 kx = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                             : make_float2(1.f, 0.f);
            kx = norm_rope(kx, cs, eps);
            __nv_fp8x2_storage_t k8 = __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale),
                                                                __NV_SATFINITE, __NV_E4M3);
            __nv_fp8x2_storage_t v8 = __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale),
                                                                __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kvh * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kvh * D + 2 * lane) = v8;
        }
        __threadfence();
    }
#endif

    // Pipeline helpers: block number j (0 .. nb) of this split -> block index first + j.
    auto issue_pages = [&](int j) {   // page entries of block j into ring slot j % PRING
        if (j < nb) {
            for (int key = tid; key < BN; key += NT) {
                int t = hi - (first + j + 1) * BN + key;
                if (t >= lo) cp4(Pg + (j % PRING) * BN + key, page_table + (long)b * pt_stride + t / PAGE);
            }
        }
    };
    auto issue_kv = [&](int j) {      // raw K and V rows of block j into stage j % STAGES
        constexpr int CPK = DIMS / 16;
        if (j < nb) {
            for (int i = tid; i < BN * CPK; i += NT) {
                int key = i / CPK, c16 = (i % CPK) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = Kr + ((j % STAGES) * BN + key) * RP + c16;
                unsigned char* dv = Vr + ((j % STAGES) * BN + key) * RP + c16;
                if (t < lo) {  // clipped rows: zeros (stale bytes could encode NaN, and 0 * NaN = NaN)
                    *(uint4*)dk = make_uint4(0, 0, 0, 0);
                    *(uint4*)dv = make_uint4(0, 0, 0, 0);
                    continue;
                }
                if (NEWSMEM && first + j == 0 && t >= seq) continue;   // the step's own rows: written above
                long row = (long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE;
                long at = row * ROW + kv0 * D + c16;
                cp16(dk, k_cache + at);
                cp16(dv, v_cache + at);
            }
        }
    };

    // Prologue: page entries of blocks 0 .. 2 STAGES - 2 (one wait), then the copies of blocks
    // 0 .. STAGES - 2, one group each.
    for (int j = 0; j < 2 * STAGES - 1; ++j) issue_pages(j);
    cp_commit();
#if !EARLYQ
    // Queries meanwhile: RMSNorm, RoPE, E4M3; decode packs the pair block-diagonally.
    // All of a warp's row loads go out before any is used (one latency, not one per row).
    constexpr int QR = (ROWS + NWARPS - 1) / NWARPS;
    float2 qx[QR], qcs[QR];
    long qpos[QR];
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#endif
#pragma unroll
    for (int i = 0; i < QR; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane])
                      : make_float2(1.f, 0.f);
    // The first blocks' copies as soon as their page entries land, while the RoPE loads above are in
    // flight: the K/V chain (seq_lens -> pages -> rows) no longer waits for the queries' chain.
    cp_wait<0>();
    __syncthreads();
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }
#if EARLYQ
    // The block-0 split writes the step's keys and values to the cache and into its stage-0 tile
    // (rows BN - L + j of block 0), whose copies issue_kv skips.
#pragma unroll
    for (int i = 0; i < KR; ++i)
        kcs[i] = ROPE ? make_float2(rope[kpos[i] * D + lane], rope[kpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
    if (first == 0 && last > 0) {
#pragma unroll
        for (int i = 0; i < KR; ++i) {
            const int w = warp + i * NWARPS, h = w / L, j = w % L, kvh = kv0 + h;
            if (w >= KVP * L) continue;
            const float2 kx = norm_rope(kxs[i], kcs[i], eps), vx = vxs[i];
            const __nv_fp8x2_storage_t k8 = __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale),
                                                                      __NV_SATFINITE, __NV_E4M3);
            const __nv_fp8x2_storage_t v8 = __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale),
                                                                      __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(k_cache + kloc[i] * ROW + kvh * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(v_cache + kloc[i] * ROW + kvh * D + 2 * lane) = v8;
            *(__nv_fp8x2_storage_t*)(smem + OFF_K + (BN - L + j) * RP + h * D + 2 * lane) = k8;
            *(__nv_fp8x2_storage_t*)(smem + OFF_V + (BN - L + j) * RP + h * D + 2 * lane) = v8;
        }
    }
#endif
    const float qsc = q_mult * k_scale;
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (r >= ROWS) continue;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (rr >= L * G) x = make_float2(0.f, 0.f);
        // Unit-RMS q to E4M3 (as the Triton kernel's QK_FP8); the scale goes on the scores. Dims are
        // permuted to match K's reads below: thread tig reads key bytes tig CH .. + CH (CH = DIMS / 4),
        // and byte t 8 + r 4 + i of them is element i of B register r of k-tile t, so dim
        // d = tig CH + t 8 + r 4 + i goes where k-tile t's A fragment puts thread tig: t 32 + r 16 + tig 4 + i.
#pragma unroll
        for (int hh = 0; hh < KVP; ++hh) {
            const int d = hh * D + 2 * lane, rem = d % (DIMS / 4);
            const int at = (rem / 8) * 32 + ((rem / 4) & 1) * 16 + (d / (DIMS / 4)) * 4 + rem % 4;
            *(__nv_fp8x2_storage_t*)(Qs + r * QP + at) =
                hh == h ? __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3) : (__nv_fp8x2_storage_t)0;
        }
    }
    __syncthreads();   // queries in shared memory (the first blocks' copies already in flight)

    const int mt = warp % WARP_M, ks = warp / WARP_M;
    constexpr int KT = DIMS / 32, NO = DIMS / 8;   // k-tiles of QK^T (E4M3, k = 32), n-tiles of O
    constexpr int WKEYS = BN / WARP_K;      // keys of each block per warp
    unsigned qa[KT][4];                     // A fragments: rows grp, grp + 8; bytes tig 4, 16 + tig 4
#pragma unroll
    for (int kt = 0; kt < KT; ++kt)
#pragma unroll
        for (int r = 0; r < 4; ++r)
            qa[kt][r] = *(const unsigned*)(Qs + (mt * 16 + grp + (r & 1) * 8) * QP + kt * 32 + (r >> 1) * 16 + tig * 4);
    int ptok[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int rr = (mt * 16 + grp + e * 8) % (ROWS / KVP);
        ptok[e] = seq + min(rr / G, L - 1);
    }
    float o[NO][4];
#pragma unroll
    for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    for (int j = 0; j < nb; ++j) {
        cp_wait<STAGES - 2>();   // block j's copies have landed (and the pages STAGES - 1 ahead)
        __syncthreads();          // ... for every thread; and nobody still reads the stage refilled next
        issue_pages(j + 2 * STAGES - 1);
        issue_kv(j + STAGES - 1);
        cp_commit();

        const unsigned char* K = Kr + (j % STAGES) * BN * RP;
        const unsigned char* V = Vr + (j % STAGES) * BN * RP;
        const int kb0 = hi - (first + j + 1) * BN;     // position of key 0 of this block
        const int k_lo = ks * WKEYS;                   // this warp's keys of the block
        float s[WKEYS / 8][4];
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            // B fragments of key grp: bytes tig CH .. + CH (CH = DIMS / 4 = 8 KT), 16-byte loads;
            // k-tile kt takes words 2 kt, 2 kt + 1, raw E4M3 (no conversion).
            const unsigned char* krow = K + (k_lo + nt * 8 + grp) * RP + tig * 8 * KT;
            unsigned kw[2 * KT];
#pragma unroll
            for (int c = 0; c < KT / 2; ++c) {
                const uint4 u = *(const uint4*)(krow + 16 * c);
                kw[4 * c] = u.x; kw[4 * c + 1] = u.y; kw[4 * c + 2] = u.z; kw[4 * c + 3] = u.w;
            }
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) mma_k32_e4m3(s[nt], qa[kt], kw + 2 * kt);
        }
        float mx[2] = {NEG_INF, NEG_INF};
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                int e = c >> 1, t = kb0 + k_lo + nt * 8 + tig * 2 + (c & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                s[nt][c] = ok ? s[nt][c] * qsc : NEG_INF;
                mx[e] = fmaxf(mx[e], s[nt][c]);
            }
        float alpha[2], safe[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
            float m_new = fmaxf(m_run[e], mx[e]);
            safe[e] = m_new == NEG_INF ? 0.f : m_new;
            alpha[e] = exp2f(m_run[e] - safe[e]);
            m_run[e] = m_new;
        }
        float rs[2] = {0.f, 0.f};
        unsigned pa[WKEYS / 8][2];
#pragma unroll
        for (int nt = 0; nt < WKEYS / 8; ++nt) {
            float p0 = exp2f(s[nt][0] - safe[0]), p1 = exp2f(s[nt][1] - safe[0]);
            float p2 = exp2f(s[nt][2] - safe[1]), p3 = exp2f(s[nt][3] - safe[1]);
            rs[0] += p0 + p1; rs[1] += p2 + p3;
            pa[nt][0] = pack_h2(p0, p1); pa[nt][1] = pack_h2(p2, p3);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
#pragma unroll
        for (int n = 0; n < NO; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        // O += P V. Output dims are permuted inside each head: column c of n-tile h * 8 + n is dim
        // h * 64 + c * 8 + n. The B fragment (keys 2 tig, 2 tig + 1; column grp) of n-tiles
        // h * 8 .. + 8 is then 8 consecutive bytes of each of two rows: two 8-byte loads, byte
        // permutes pair the rows, one cvt per n-tile.
#pragma unroll
        for (int kt = 0; kt < WKEYS / 8; ++kt) {
            const unsigned char* v0 = V + (k_lo + kt * 8 + 2 * tig) * RP + grp * 8;
#pragma unroll
            for (int h = 0; h < KVP; ++h) {
                const uint2 a = *(const uint2*)(v0 + h * D), c = *(const uint2*)(v0 + RP + h * D);
                const unsigned pr[4] = {__byte_perm(a.x, c.x, 0x5140), __byte_perm(a.x, c.x, 0x7362),
                                        __byte_perm(a.y, c.y, 0x5140), __byte_perm(a.y, c.y, 0x7362)};
#pragma unroll
                for (int n = 0; n < 8; ++n)
                    mma_k8(o[h * 8 + n], pa[kt][0], pa[kt][1],
                           e4m3x2_to_h2((unsigned short)(pr[n >> 1] >> (16 * (n & 1)))));
            }
        }
    }
    cp_wait<0>();

    // Partials: rows keep their own head's 64 dims (n-tiles h * 8 .. + 8). Key slices (decode)
    // merge through shared memory (the stages are free now). Then normalized o and lse to ws.
    __syncthreads();
    float* Wm = (float*)(smem + OFF_K);     // [NWARPS][16][2]
    float* Wo = Wm + NWARPS * 16 * 2;       // [NWARPS][16][D]
    // Row grp + 8 e of the m-tile belongs to head e on decode (rows 0-7 head 0, 8-15 head 1),
    // to the program's only head on verify: a compile-time index into o. Decode with L G < 8
    // fails here at compile time, so _cuda_kernel_consts keeps the Triton kernel for it (with the
    // "CUDA kernel A unavailable" line; no test or scored shape has L G < 8).
    static_assert(KVP == 1 || (WARP_M == 1 && L * G == 8), "decode packs 8 rows per head");
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int rl = grp + e * 8, h = KVP == 1 ? 0 : e;
        if (tig == 0) { Wm[(warp * 16 + rl) * 2] = m_run[e]; Wm[(warp * 16 + rl) * 2 + 1] = l_run[e]; }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            Wo[(warp * 16 + rl) * D + tig * 16 + n] = o[h * 8 + n][e * 2];       // dim (2 tig) 8 + n
            Wo[(warp * 16 + rl) * D + tig * 16 + 8 + n] = o[h * 8 + n][e * 2 + 1];
        }
    }
    __syncthreads();
    // Each row's slice weights (exp2 of the max gap, over the row's denominator, times v_scale) and
    // its lse, once per row instead of once per (row, dim).
    float* Ww = Wo + NWARPS * 16 * D;       // [ROWS][WARP_K + 1]
    if (tid < ROWS) {
        const int mr = tid / 16, rl = tid % 16;
        float mm = NEG_INF, wgt[WARP_K], den = 0.f;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        const float safe = mm == NEG_INF ? 0.f : mm;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) {
            const int w = k * WARP_M + mr;
            wgt[k] = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt[k] * Wm[(w * 16 + rl) * 2 + 1];
        }
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) Ww[tid * (WARP_K + 1) + k] = den > 0.f ? wgt[k] / den * v_scale : 0.f;
        Ww[tid * (WARP_K + 1) + WARP_K] = den > 0.f ? mm + __log2f(den) : NEG_INF;
    }
    __syncthreads();
#if CL
    cooperative_groups::this_cluster().barrier_wait();   // every rank has started (arrived at the top)
#endif
    for (int idx = tid; idx < ROWS * D; idx += NT) {
        int r = idx / D, d = idx % D, mr = r / 16, rl = r % 16;
        int h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (rr >= L * G) continue;
        float val = 0.f;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) val += Ww[r * (WARP_K + 1) + k] * Wo[((k * WARP_M + mr) * 16 + rl) * D + d];
        const float lse = Ww[r * (WARP_K + 1) + WARP_K];
#if CL
        // Row r's partial straight into its owner's buffer (rank rho % CL, slot rho / CL).
        {
            const int rho = h * (L * G) + rr;
            float* mg = cooperative_groups::this_cluster().map_shared_rank((float*)(smem + OFF_MG), rho % CL);
            float* slot = mg + ((rho / CL) * CL + split) * (D + 1);
            slot[d] = val;
            if (d == 0) slot[D] = lse;
        }
#else
        int j = rr / G, g = rr % G, head = (kv0 + h) * G + g;
        long slot = ((long)(b * L + j) * HQ + head) * num_splits + split;
        ws_o[slot * D + d] = val;
        if (d == 0) ws_lse[slot] = lse;
#endif
    }
#if CL
    // Merge: real row rho on rank rho % CL, warp (rho / CL) % NWARPS; lane holds dims lane, lane + 32.
    const int rank = split;   // the cluster spans the grid's x
    cooperative_groups::this_cluster().sync();   // every rank's partials are in their owners' buffers
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = rank + CL * (warp + NWARPS * i);
        if (rho >= KVP * L * G) break;
        const int rr = rho % (L * G), j = rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const float* mg = (const float*)(smem + OFF_MG) + (rho / CL) * CL * (D + 1);
        float mm = NEG_INF;
#pragma unroll
        for (int c = 0; c < CL; ++c) mm = fmaxf(mm, mg[c * (D + 1) + D]);
        const float safe = mm == NEG_INF ? 0.f : mm;
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int c = 0; c < CL; ++c) {   // empty splits: lse -inf, weight 0
            const float w = exp2f(mg[c * (D + 1) + D] - safe);
            den += w;
            o0 += w * mg[c * (D + 1) + lane];
            o1 += w * mg[c * (D + 1) + lane + 32];
        }
        o0 /= den; o1 /= den;
        const float inv = rsqrtf(warp_sum(o0 * o0 + o1 * o1) / D + eps);
        const long tok = b * L + j;
        const float g0 = 1.f / (1.f + __expf(-gate_r[i].x)), g1 = 1.f / (1.f + __expf(-gate_r[i].y));
        out[tok * (HQ * D) + head * D + lane] = __float2bfloat16(o0 * inv * g0);
        out[tok * (HQ * D) + head * D + lane + 32] = __float2bfloat16(o1 * inv * g1);
    }
#endif
}
'''

# verify_ws: verify tiles, sliding and full layers.
KERNEL_A_WS_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// verify_ws: split-KV attention for verify tiles (8 tokens x 8 query heads = 64 rows per KV head),
// warp-specialized, sm_90a. -D from _cufa_consts: HQ HKV L STAGES PAGE WINDOW ROPE WS=1 KVP=1
// PNOWAIT=1 PSMEM=1. One producer warp gathers each block's scattered K and V rows with cp.async
// into a STAGES-deep ring and signals full[s] with cp.async.mbarrier.arrive.noinc (it never waits on
// its own copies); one consumer warpgroup per KV head runs S = Q K^T as wgmma m64n64k32 on E4M3
// from shared memory, converts V to FP16 while the wgmma runs, and does P V with mma.sync m16n8k16.
// Each CTA writes its split's normalized partial and log-sum-exp for the combine kernel.
#define D 64
#define BN 64                          // keys per block
#define NS (BN / 2)                    // scores per thread (the m64nBN accumulator)
#define ROWS 64                        // query rows per KV head: one warpgroup m64
#define NWG KVP                        // consumer warpgroups, one per KV head
#define NCONS (128 * NWG)              // consumer threads
#define NT (NCONS + 128)               // + the producer warpgroup (warp NCONS / 32 copies)
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define KROW (KVP * D)
#define TILE (BN * D)                  // bytes of an E4M3 key/value tile (BN x 64)
#define QTILE (ROWS * D)               // bytes of an E4M3 query tile (64 x 64)
#define VP (D + 8)                     // FP16 V rows (halves), padded: ldmatrix without bank conflicts
#define NEG_INF __int_as_float(0xff800000)
// Registers: 64 at launch x 256 threads (4 CTAs/SM), then setmaxnreg moves them to 24 for the
// producer warpgroup and 104 for the consumer warpgroup (24 x 128 + 104 x 128 = 64 x 256).
#define MINB 4                         // CTAs per SM (__launch_bounds__)
#define CREGS 104                      // consumer registers (setmaxnreg.inc)
#define PREGS 24                       // producer registers (setmaxnreg.dec)
#define VHBUF 2                        // FP16 V buffers per consumer warpgroup
// The page entries go to a shared-memory ring by cp.async, PQS blocks ahead, and each block waits
// for its own with wait_group. A register queue made the compiler move every loaded entry into its
// slot right away, so each load was waited for where it was issued.
#define PQS 4                          // blocks of page entries in flight
#define PSL (PQS + 1)                  // ring slots

#define OFF_Q 0
#define OFF_K (OFF_Q + NWG * QTILE)    // a Q tile per consumer warpgroup
#define OFF_V (OFF_K + STAGES * KVP * TILE)
#define OFF_VH (OFF_V + STAGES * KVP * TILE)
#define OFF_B (OFF_VH + NWG * VHBUF * BN * VP * 2)   // mbarriers: full[STAGES], empty[STAGES]
#define OFF_PG (OFF_B + (2 * STAGES + 4) * 16)      // 16 bytes per barrier (+ 4 unused slots)
// [PSL][BN] page entries from OFF_PG on; the host's _cufa_smem has the same total.

static_assert(L * G <= ROWS, "one warpgroup holds a head's rows");
static_assert(KVP == 1 && PNOWAIT == 1 && PSMEM == 1, "the only compiled variant (MINB, CREGS, PREGS assume it)");
static_assert(STAGES >= 2, "one block in flight at least");
#define CPR (KROW / 16)                // the producer's lanes per key row (16 bytes each)

__device__ __forceinline__ int cm(int row, int byte) {   // no-swizzle core matrices of a 64 x 64-byte tile
    return (row >> 3) * 512 + (byte >> 4) * 128 + (row & 7) * 16 + (byte & 15);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma_k16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
// 16-byte copy, zero-filled when !ok (src-size 0).
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// Named barrier for one warpgroup (ids 1 .. KVP; 0 is __syncthreads).
__device__ __forceinline__ void wg_bar(int id) { asm volatile("bar.sync %0, 128;\n" :: "r"(id) : "memory"); }

// mbarriers: arrive count fixed at init, parity waits.
__device__ __forceinline__ void mb_init(unsigned long long* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, int parity) {
    asm volatile("{\n.reg .pred done;\nWAIT_%=:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
                 "@!done bra WAIT_%=;\n}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
// Arrive on b once all of this thread's earlier cp.async copies have landed (counted in init).
__device__ __forceinline__ void mb_arrive_cp(unsigned long long* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ unsigned long long wg_desc(const void* p) {   // no swizzle, LBO 128, SBO 512
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | (8ull << 16) | (32ull << 32);
}
__device__ __forceinline__ void wg_qk(float* d, unsigned long long da, unsigned long long db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15,"
                 "%16, %17, %18, %19, %20, %21, %22, %23,"
                 "%24, %25, %26, %27, %28, %29, %30, %31"
                 "}, %32, %33, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                   "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                   "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ float2 norm_rope(float2 x, float2 cs, float eps) {
    float inv = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + eps);
    x.x *= inv; x.y *= inv;
    return make_float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

extern "C" __global__ void __launch_bounds__(NT, MINB) verify_ws(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [NWG][64 rows x 64 B] E4M3, core-matrix layout
    unsigned char* Ks = smem + OFF_K;   // [STAGES][KVP][64 keys x 64 B] E4M3, core-matrix layout
    unsigned char* Vr = smem + OFF_V;   // [STAGES][64 keys][KROW] E4M3, row-major
    __half* Vh = (__half*)(smem + OFF_VH);   // [NWG][VHBUF][64 keys][VP] FP16
    // Barrier k at 16-byte steps.
    unsigned long long* full = (unsigned long long*)(smem + OFF_B);       // full[2 k]
    unsigned long long* empty = full + 2 * STAGES;                        // empty[2 k]

    const int split = blockIdx.x, kv0 = blockIdx.y * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    const float k_scale = *k_scale_ptr, v_scale = *v_scale_ptr;
    const int seq = seq_lens[b], hi = seq + L;
    const int lo = WINDOW > 0 ? max(seq - WINDOW + 1, 0) : 0;
    const int n_blocks = (hi - lo + BN - 1) / BN;
    const int per = (n_blocks + num_splits - 1) / num_splits;
    const int last = n_blocks - split * per, first = max(last - per, 0);
    const int nb = max(last - first, 0);

    // The split holding block 0 writes the step's new keys and values to the cache before anyone
    // copies a block (the barrier below).
    if (first == 0 && last > 0) {
        for (int w = warp; w < KVP * L; w += NT / 32) {
            int j = w % L, kvh = kv0 + w / L, tok = b * L + j;
            long pos = positions[tok], loc = cache_loc[tok];
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            float2 kx = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                             : make_float2(1.f, 0.f);
            kx = norm_rope(kx, cs, eps);
            *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kvh * D + 2 * lane) =
                __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale), __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kvh * D + 2 * lane) =
                __nv_cvt_float2_to_fp8x2(make_float2(vx.x / v_scale, vx.y / v_scale), __NV_SATFINITE, __NV_E4M3);
        }
        __threadfence();
    }
    if (tid < STAGES) {
        mb_init(full + 2 * tid, 32);         // the producer warp's lanes
        mb_init(empty + 2 * tid, NCONS / 32);   // a lane per consumer warp reading the stage
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();   // new keys and values in the cache, barriers initialized: the last CTA-wide barrier

    if (warp >= NCONS / 32) {
        // ---------------- producer warpgroup: its first warp copies, the others leave ----------------
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(PREGS));
        if (warp != NCONS / 32) return;
        // Lane l loads the page entries of keys l and l + 32 of each block, PQS blocks ahead into the
        // shared-memory ring (loading them one block ahead left the producer waiting a DRAM latency
        // per block); the copies are spread so that each warp instruction takes whole rows (below).
        int* Pg = (int*)(smem + OFF_PG);
        auto issue_pages = [&](int j) {   // block j's entries into slot j % PSL, one cp.async group
            if (j < nb)
#pragma unroll
                for (int e = 0; e < BN / 32; ++e) {
                    const int key = lane + 32 * e, t = hi - (first + j + 1) * BN + key;
                    if (t >= lo) cp4(Pg + (j % PSL) * BN + key, page_table + (long)b * pt_stride + t / PAGE);
                }
            cp_commit();
        };
#pragma unroll
        for (int u = 0; u < PQS; ++u) issue_pages(u);
        for (int j = 0; j < nb; ++j) {
            const int s = j % STAGES;
            if (j >= STAGES)   // the consumers left j - STAGES
                mb_wait(empty + 2 * s, ((j / STAGES) - 1) & 1);
            cp_wait<PQS - 1>();   // block j's entries (and the copies of blocks before j - PQS + 1)
            __syncwarp();         // ... landed for every lane
#pragma unroll
            for (int i = 0; i < BN * CPR / 32; ++i) {
                const int key = (32 / CPR) * i + lane / CPR, c = lane % CPR, t = hi - (first + j + 1) * BN + key;
                const bool ok = t >= lo;
                const long at = (long)(ok ? Pg[(j % PSL) * BN + key] * PAGE + t % PAGE : 0) * ROW + kv0 * D + c * 16;
                cp16z(Ks + (s * KVP + c / 4) * TILE + cm(key, (c % 4) * 16), k_cache + at, ok);
                cp16z(Vr + s * KVP * TILE + key * KROW + c * 16, v_cache + at, ok);
            }
            mb_arrive_cp(full + 2 * s);   // full[s] completes when every lane's copies so far have landed
            __syncwarp();                 // every lane has read slot j % PSL before it is refilled
            issue_pages(j + PQS);
        }
        cp_wait<0>();   // no thread leaves with cp.async in flight
        return;
    }

    // ---------------- consumer warpgroup wg: KV head kv0 + wg ----------------
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CREGS));
    const int wg = warp >> 2, h = wg, wl = warp & 3, ct = tid & 127;
    unsigned char* Q = Qs + wg * QTILE;
    // Queries: RMSNorm, RoPE, E4M3 at unit RMS (16 rows per warp; rows past L G are zero).
    float2 qx[ROWS / 4], qcs[ROWS / 4];
    long qpos[ROWS / 4];
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = wl + i * 4;
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < L * G) {
            int tok = b * L + r / G;
            const __nv_bfloat162* src =
                (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + r % G) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = wl + i * 4;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (r >= L * G) x = make_float2(0.f, 0.f);
        *(__nv_fp8x2_storage_t*)(Q + cm(r, 2 * lane)) = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    }
    fence_async_smem();
    wg_bar(1 + wg);

    const float qsc = q_mult * k_scale;
    int ptok[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) ptok[e] = seq + min((wl * 16 + grp + e * 8) / G, L - 1);
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};
    unsigned pa[BN / 16][4];   // A fragments of P for k-tiles of 16 keys

    for (int j = 0; j < nb; ++j) {
        const int s = j % STAGES;
        const unsigned char* K = Ks + (s * KVP + h) * TILE;
        const unsigned char* V = Vr + s * KVP * TILE;
        __half* vh = Vh + (wg * VHBUF + j % VHBUF) * BN * VP;
        const int kb0 = hi - (first + j + 1) * BN;
        mb_wait(full + 2 * s, (j / STAGES) & 1);   // block j landed
        fence_async_smem();   // the producer's generic-proxy copies, before wgmma reads K
        // S = Q K^T: accumulator element nt 4 + c is row wl 16 + grp + 8 (c >> 1), key nt 8 + 2 tig + (c & 1).
        float sc[NS];
        // Pinned zeros; the first QK^T wgmma has scale-d 0 anyway.
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            sc[i] = 0.f;
            asm volatile("" : "+f"(sc[i]) :: "memory");
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wg_qk(sc, wg_desc(Q), wg_desc(K), 0);
        wg_qk(sc, wg_desc(Q + 256), wg_desc(K + 256), 1);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
#pragma unroll
        for (int vk = 0; vk < BN / 64; ++vk) {   // 32 bytes of key vk 64 + ct / 2 per pass
            const int key = vk * 64 + (ct >> 1), c0 = (ct & 1) * 32;
            const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
            const uint4 c = *(const uint4*)(V + key * KROW + h * D + c0 + 16);
            const unsigned w[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
            unsigned hv[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
            }
            uint4* dst = (uint4*)(vh + key * VP + c0);
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = make_uint4(hv[4 * i], hv[4 * i + 1], hv[4 * i + 2], hv[4 * i + 3]);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // QK^T of j
#pragma unroll
        for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(sc[i]) :: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive(empty + 2 * s);   // this warp is done with stage s (K by wgmma, V converted)
        wg_bar(1 + wg);                        // vh complete for the warpgroup

        const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
        float mx[2] = {NEG_INF, NEG_INF};
        if (edge) {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                sc[i] = ok ? sc[i] * qsc : NEG_INF;
                mx[e] = fmaxf(mx[e], sc[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                sc[i] *= qsc;
                mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], sc[i]);
            }
        }
        float alpha[2], safe[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
            float m_new = fmaxf(m_run[e], mx[e]);
            safe[e] = m_new == NEG_INF ? 0.f : m_new;
            alpha[e] = exp2f(m_run[e] - safe[e]);
            m_run[e] = m_new;
        }
        float rs[2] = {0.f, 0.f};
        // sc * 1 - safe as an FMA: the scored build has this form, and sc - safe compiles to other PTX.
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            float p0 = exp2f(fmaf(sc[nt * 4], 1.f, -safe[0])), p1 = exp2f(fmaf(sc[nt * 4 + 1], 1.f, -safe[0]));
            float p2 = exp2f(fmaf(sc[nt * 4 + 2], 1.f, -safe[1])), p3 = exp2f(fmaf(sc[nt * 4 + 3], 1.f, -safe[1]));
            rs[0] += p0 + p1; rs[1] += p2 + p3;
            pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
            rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk)
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int m = lane >> 3;
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, vh + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                mma_k16(o[2 * np], pa[kk], b0, b1);
                mma_k16(o[2 * np + 1], pa[kk], b2, b3);
            }
    }

    // Partials: normalized o and base-2 lse per row, straight to ws (one warp per 16 rows).
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = wl * 16 + grp + e * 8;
        if (r >= L * G) continue;
        const long slot = ((long)(b * L + r / G) * HQ + (kv0 + h) * G + r % G) * num_splits + split;
        const float l = l_run[e], inv = l > 0.f ? v_scale / l : 0.f;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *(float2*)(ws_o + slot * D + n * 8 + 2 * tig) = make_float2(o[n][2 * e] * inv, o[n][2 * e + 1] * inv);
        if (tig == 0) ws_lse[slot] = l > 0.f ? m_run[e] + __log2f(l) : NEG_INF;
    }
}
'''


# ---- CUDA kernel A: host side (compile, occupancy, launch) ----

def _cufa_consts(cfg, kind, cl=0, kvp=None):
    """-D constants of the CUDA kernel A of this kind: "fa2" (full_fa2), "cl" (full_fa2 with
    clusters of cl CTAs) or "ws" (verify_ws). kvp: the tuning's cuda_kvp (decode tiles; 1: one KV
    head per CTA, 8 of 16 rows real, half the O registers)."""
    hq, hkv, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    if kind == "ws":
        return dict(HQ=hq, HKV=hkv, L=length, STAGES=CUDA_WS_STAGES, PAGE=cfg["page_size"], WINDOW=cfg["window"],
                    ROPE=int(bool(cfg["rope_theta"])), WS=1, KVP=WS_KVP, PNOWAIT=1, PSMEM=1)
    g = hq // hkv
    decode = hkv % 2 == 0 and 2 * length * g <= 16
    warp_m = 1 if decode else -(-length * g // 16)
    return dict(HQ=hq, HKV=hkv, L=length, KVP=(kvp or 2) if decode else 1, WARP_M=warp_m,
                WARP_K=4 if decode else 1, STAGES=CUDA_STAGES, PAGE=cfg["page_size"], WINDOW=cfg["window"],
                ROPE=int(bool(cfg["rope_theta"])), MINB=CUDA_DECODE_MINB if decode else max(1, 16 // warp_m),
                **({"CL": cl, "NEWSMEM": 1, "EARLYQ": 1} if kind == "cl" else {}),
                KVEARLY=1, EPIW=1)


def _cufa_smem(k):
    """Dynamic shared memory of the CUDA kernel A with constants k (the OFF_* layout of its source)."""
    if k.get("WS"):  # per consumer warpgroup a Q tile and two FP16 V buffers; per head the K and V
        # stages; the barriers; the page-entry ring
        return (k["KVP"] * (64 * 64 + 2 * 64 * 72 * 2) + k["KVP"] * 2 * k["STAGES"] * 64 * 64
                + (2 * k["STAGES"] + 4) * 16 + 5 * 64 * 4)
    rows, dims = k["WARP_M"] * 16, k["KVP"] * 64
    total = rows * (dims + 16) + 2 * k["STAGES"] * 64 * (dims + 16) + 2 * k["STAGES"] * 64 * 4
    nw = k["WARP_M"] * k["WARP_K"]
    merge = nw * 16 * 2 * 4 + nw * 16 * 64 * 4 + rows * (k["WARP_K"] + 1) * 4   # Wm, Wo and the EPIW weights
    assert merge <= 2 * k["STAGES"] * 64 * (dims + 16), k  # merge buffers fit
    if k.get("CL"):   # the merge rows' buffers: [MROWS][CL][D + 1] floats
        nreal = k["KVP"] * k["L"] * (k["HQ"] // k["HKV"])
        total += -(-nreal // k["CL"]) * k["CL"] * 65 * 4
    return total


def _cufa_compile(k, arch):
    from cuda.bindings import driver, nvrtc
    key = (tuple(sorted(k.items())), arch)
    if key in _CUDA_FUNCS:
        return _CUDA_FUNCS[key]
    source, name = (KERNEL_A_WS_CUDA_SOURCE, b"verify_ws") if k.get("WS") else (KERNEL_A_CUDA_SOURCE, b"full_fa2")
    prog = _cuda_check(nvrtc.nvrtcCreateProgram(source.encode(), name + b".cu", 0, [], []))
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", b"-default-device", b"--use_fast_math"]
    opts += [f"-D{n}={v}".encode() for n, v in k.items()]
    opts += [b"-I" + d.encode() for d in _cuda_include_dirs()]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = _cuda_check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError(f"NVRTC: {log.decode(errors='replace')}")
    n = _cuda_check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    _cuda_check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    smem = _cufa_smem(k)
    if not torch.cuda.is_available():
        _CUDA_FUNCS[key] = (cubin, smem)
        return _CUDA_FUNCS[key]
    torch.cuda.init()
    module = _cuda_check(driver.cuModuleLoadData(cubin))
    func = _cuda_check(driver.cuModuleGetFunction(module, name))
    A = driver.CUfunction_attribute
    _cuda_check(driver.cuFuncSetAttribute(func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
    if k.get("CL", 0) > 8:
        _cuda_check(driver.cuFuncSetAttribute(func, A.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
    _CUDA_FUNCS[key] = (func, smem)
    return _CUDA_FUNCS[key]


def _cufa_threads(k):
    if k.get("WS"):   # a consumer warpgroup per KV head, plus the producer warpgroup
        return 128 * k["KVP"] + 128
    return k["WARP_M"] * k["WARP_K"] * 32


def _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, eps, k, out):
    """Kernel A of the split-KV path, grid (splits, KV-head groups, requests): per-split partials into
    ws_o [tokens, Hq, splits, 64] and ws_lse [tokens, Hq, splits], as the Triton kernel A writes
    them; with k["CL"], the whole step into out (clusters of CL = splits CTAs; ws unused)."""
    from cuda.bindings import driver
    cfg = data["config"]
    func, smem = _cufa_compile(k, _cuda_arch(ws_o.device))
    lc = driver.CUlaunchConfig()
    lc.gridDimX, lc.gridDimY, lc.gridDimZ = splits, cfg["num_kv_heads"] // k["KVP"], cfg["batch"]
    lc.blockDimX, lc.blockDimY, lc.blockDimZ = _cufa_threads(k), 1, 1
    lc.sharedMemBytes = smem
    lc.hStream = torch.cuda.current_stream(ws_o.device).cuda_stream
    attrs = []
    if k.get("CL", 0) > 1:   # the cluster spans the splits, along x
        a = driver.CUlaunchAttribute()
        a.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = k["CL"], 1, 1
        attrs.append(a)
    lc.attrs = attrs
    lc.numAttrs = len(attrs)
    vp = ctypes.c_void_p
    tensors = (data["qkv"], data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
               data["k_scale"], data["v_scale"], data["page_table"], rope_table, ws_o, ws_lse)
    values = tuple(t.data_ptr() for t in tensors) + (data["page_table"].stride(0), splits, q_mult, eps)
    types = (vp,) * len(tensors) + (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float)
    if not k.get("WS"):   # full_fa2 also takes out (used with CL) and cnt (unused: out again)
        values, types = values + (out.data_ptr(), out.data_ptr()), types + (vp, vp)
    _cuda_check(driver.cuLaunchKernelEx(lc, func, (values, types), 0))


def _cuda_arch(device):
    major, minor = torch.cuda.get_device_capability(device)
    return "sm_90a" if major == 9 else f"sm_{major}{minor}"


def _cufa_ctas(k, device):
    """Resident CTAs per SM of the compiled CUDA kernel A (the driver's occupancy query)."""
    key = tuple(sorted(k.items()))
    if key not in _CUDA_CTAS:
        from cuda.bindings import driver
        func, smem = _cufa_compile(k, _cuda_arch(device))
        nt = _cufa_threads(k)
        _CUDA_CTAS[key] = max(1, _cuda_check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, nt, smem)))
    return _CUDA_CTAS[key]


def _cuda_kernel_consts(data, tune):
    """Whether this step's kernel A runs in CUDA: on a GPU (not the interpreter), within the
    tuning's cuda_max_batch, Hopper, 64-wide heads, at most 64 rows per KV head, and a kernel that
    compiled (tried on the first call of each config, outside graph capture; a failure keeps the
    Triton path for good). Returns its constants, or None for the Triton kernel."""
    cfg = data["config"]
    if not (ON_GPU and data["qkv"].is_cuda):
        return None
    if tune.get("cuda_max_batch") is not None and cfg["batch"] > tune["cuda_max_batch"]:
        return None
    if torch.cuda.get_device_capability(data["qkv"].device)[0] != 9 or cfg["head_dim"] != 64:
        return None
    if cfg["num_q_heads"] % cfg["num_kv_heads"] or cfg["q_len"] * cfg["num_q_heads"] // cfg["num_kv_heads"] > 64:
        return None
    k = _cufa_consts(cfg, tune["cuda"], cl=tune.get("cl", 0), kvp=tune.get("cuda_kvp"))
    key = tuple(sorted(k.items()))
    if key in _CUDA_FAILED:
        return None
    try:
        _cufa_ctas(k, data["qkv"].device)
    except Exception as e:  # noqa: BLE001  (any compile, load or query failure: keep the Triton path)
        print(f"CUDA kernel A unavailable for {k}: {e!r}", flush=True)
        _CUDA_FAILED.add(key)
        return None
    return k


# ---- Host helpers and kernel_fn ----

_ROPE = {}
_SMS = {}
_PDL_LAUNCH = {"launch_pdl": True} if ON_GPU and gdc is not None else {}


ROPE_POSITIONS = 1 << 21  # as SGLang's table (baseline.py): built once, never regrown inside a graph


def _rope_table(theta, d, positions, device):
    """[ROPE_POSITIONS, d] float32: cos | sin of pos * theta^(-2i/d), as the reference computes
    them (SGLang's RotaryEmbedding keeps the same table). Built from config only, once per
    theta, on the first call (a warmup, outside graph capture)."""
    assert positions <= ROPE_POSITIONS, f"RoPE table holds {ROPE_POSITIONS} positions, cache has {positions}"
    key = (theta, d, device)
    table = _ROPE.get(key)
    if table is None:
        n = ROPE_POSITIONS
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        ang = torch.arange(n, dtype=torch.float, device=device)[:, None] * inv_freq[None, :]
        table = _ROPE[key] = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
    return table


def _regs(cap):
    return {} if cap is None else {"maxnreg": cap}


def _sm_count(device):
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def num_splits(span: int, want: int, min_chunk: int) -> int:
    """Splits per (request, KV-head group): `want`, but no split under min_chunk keys nor over
    MAX_SPLITS. span bounds the keys a request attends over."""
    return max(1, min(want, triton.cdiv(span, min_chunk), MAX_SPLITS))


_OCCUPANCY = {}


def _ctas_per_sm(kernel, make_args, meta, device):
    """Resident CTAs per SM of kernel compiled with meta: the tightest of registers, shared
    memory and warps. Compiled once per meta (warmup, outside graph capture); num_splits isn't
    specialized, so this is the binary every split count launches. If Triton's compiled-kernel
    API differs from 3.8's, falls back to 4 (what 128 registers and 4 warps give)."""
    key = tuple(sorted(meta.items()))
    if key not in _OCCUPANCY:
        ctas = 4
        if ON_GPU:
            try:
                compiled = kernel.warmup(*make_args(), grid=(1,), **meta)
                compiled._init_handles()
                props = torch.cuda.get_device_properties(device)
                warps = meta["num_warps"]
                regs_per_warp = -(-compiled.n_regs * 32 // 256) * 256  # allocated per warp in 256s
                by_regs = getattr(props, "regs_per_multiprocessor", 65536) // regs_per_warp // warps
                smem = compiled.metadata.shared
                by_smem = getattr(props, "shared_memory_per_multiprocessor", 233472) // (smem + 1024) if smem else 32
                by_warps = getattr(props, "max_threads_per_multi_processor", 2048) // 32 // warps
                ctas = max(1, min(by_regs, by_smem, by_warps, 32))
            except (AttributeError, TypeError, RuntimeError):
                pass
        _OCCUPANCY[key] = ctas
    return _OCCUPANCY[key]


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window = cfg["window"]
    qkv = data["qkv"]
    device = qkv.device
    tokens = batch * length
    page_table = data["page_table"]

    # Decode packs both KV heads of a pair into one 16-row block-diagonal tile. Verify blocks
    # (64 rows per KV head) take one KV head per program and one tl.dot per head, as packing
    # would double their MMA work.
    group_rows = length * hq // hkv
    tune = TUNING[("sliding" if window else "full", "decode" if group_rows <= 16 else "verify")]
    kvp = 2 if hkv % 2 == 0 and 2 * group_rows <= 16 else 1
    # The step's own tokens must fit in the newest key block (block 0), the only one with the
    # causal mask.
    assert length <= tune["block_n"], f"q_len {length} > key block"
    pairs = hkv // kvp
    span = window + length if window else page_table.shape[1] * page
    rows = max(16, triton.next_power_of_2(kvp * group_rows))
    rope = bool(cfg["rope_theta"])
    rope_table = _rope_table(cfg["rope_theta"], d, page_table.shape[1] * page, device) if rope else data["k_scale"]
    q_mult = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    meta = dict(
        L=length, LP=triton.next_power_of_2(max(length, 2)), HQ=hq, HKV=hkv, D=d, PAGE=page, WINDOW=window,
        ROPE=rope, KVP=kvp, M=rows, BLOCK_N=tune["block_n"], ONE_SPLIT=False,
        # q in E4M3 and Q K^T on the FP8 keys where wgmma runs (64-row tiles); Triton's mma.sync
        # path for 16-row tiles would upcast to FP16 anyway.
        QK_FP8=rows >= 64, EDGES=tune["edges"],
        # Sliding layers (few, short programs): kernel A lets the combine launch at its start.
        EARLY_TRIGGER=window > 0,
        num_warps=tune["warps"], num_stages=tune["stages"], **_regs(tune["maxnreg"]))

    def args(ws_o, ws_lse, splits):
        return (qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
                data["k_scale"], data["v_scale"], page_table, rope_table, out, ws_o, ws_lse,
                page_table.stride(0), splits, q_mult, RMS_EPS)

    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    sms = _sm_count(device)
    k = _cuda_kernel_consts(data, tune)
    cuda = k is not None
    if cuda:  # kernel A in CUDA: its own KV grouping and occupancy
        pairs = hkv // k["KVP"]
    if cuda and k.get("CL"):
        # One kernel: a cluster of CL CTAs (the splits) merges in place and stores the output.
        _cufa_launch(data, out, out, k["CL"], rope_table, q_mult, RMS_EPS, k, out)
        return out
    if window:
        splits = num_splits(span, triton.cdiv(tune["per_sm"] * sms, batch * pairs), tune["min_chunk"])
    elif cuda:
        splits = num_splits(span, tune["waves"] * sms * _cufa_ctas(k, device) // (batch * pairs), tune["min_chunk"])
    else:
        def probe_args():
            probe = torch.empty(1, dtype=torch.float32, device=device)
            return args(probe, probe, 2)

        ctas = _ctas_per_sm(_attn_kernel, probe_args, meta, device)
        splits = num_splits(span, tune["waves"] * sms * ctas // (batch * pairs), tune["min_chunk"])
    if splits > 1 or cuda:  # the CUDA kernel always writes partials, even for one split
        ws_o = torch.empty((tokens, hq, splits, d), dtype=torch.float32, device=device)
        ws_lse = torch.empty((tokens, hq, splits), dtype=torch.float32, device=device)
    else:
        ws_o = ws_lse = out
    if cuda:
        _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, RMS_EPS, k, out)
    else:
        meta.update(ONE_SPLIT=splits == 1)
        _attn_kernel[(splits, pairs, batch)](*args(ws_o, ws_lse, splits), **meta)
    if splits > 1 or cuda:
        # One program per (token, query head). (HB > 1 heads per program is untested on GPU.)
        hb = 1
        _combine_kernel[(tokens, triton.cdiv(hq, hb))](
            ws_o, ws_lse, qkv, out, splits, RMS_EPS, HQ=hq, HKV=hkv, D=d, HB=hb,
            SP=min(COMBINE_SP, triton.next_power_of_2(splits)), num_warps=COMBINE_WARPS if splits > 32 else 2,
            **_PDL_LAUNCH)
    return out
