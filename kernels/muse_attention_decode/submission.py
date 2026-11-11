# okc: problem=muse_attention_decode
"""Gated attention decode in Triton: QK-norm, RoPE, the FP8 cache write, paged attention and
the norm-gate in one kernel, plus a combine kernel when the keys are split across programs.

Kernel A, grid (split, KV-head group, request). Decode programs own a pair of KV heads and
read whole cache rows (both tp8 heads sit in 128 contiguous bytes); they pack the pair into
one block-diagonal [16, 128] query tile (16 useful rows, where one head alone has 8). Verify
programs (8 tokens x 8 query heads = 64 rows per KV head) own one KV head.

Key blocks are counted back from the newest token: block 0 holds the step's new tokens and
belongs to exactly one split, which writes them to the cache and reads them back after a
barrier, so no other program depends on that write. On verify tiles only the edge blocks
(block 0 and the oldest one or two) apply the causal and window masks; decode tiles run one
masked loop, which keeps the kernel's code small. With one split, kernel A also applies
the norm-gate; with more, it writes each split's normalized output and log-sum-exp, and
kernel B merges them and applies the norm-gate, launched early with PDL.

Precision: in the Triton kernel A, q is rounded to E4M3 (as FA3 does) on verify tiles and Q @ K^T
runs on the cache's E4M3 keys as stored; elsewhere, and everywhere in the CUDA kernel A, q is
FP16 and keys and values go from E4M3 to FP16 (exact); P stays FP16; everything accumulates in
FP32. Cache scales are read from their tensors, the page size from
config, RoPE's cos and sin from a table built from config. The split count depends only on
shapes and on the compiled kernel's occupancy, and per-shape-range tuning is in TUNING."""

import atexit
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


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
MAX_SPLITS = 128      # most splits per (request, KV-head pair); each one costs the combine a step
CUDA_STAGES = 3       # cp.async pipeline depth of the CUDA kernel A (tiles in flight + 1)
CUDA_EXTRA = {}       # extra -D constants for kernel A in CUDA (memory-path experiments)
CUDA_WG_STAGES = 3    # the same, for the wgmma verify kernel
CUDA_WS_STAGES = 3    # and for the warp-specialized one
CUDA_DECODE_MINB = 1  # decode CTAs per SM the CUDA kernel A's registers must fit (__launch_bounds__)
COMBINE_SP = 32       # combine: splits merged per step (fewer serial L2 round trips with many splits)
COMBINE_WARPS = 2     # ... and its warps when there are more than 32 splits

# Tuning by shape range, measured on the bench (H100) and confirmed with H200 checks; diario.md
# has every number. Layer "sliding" (a window) or "full"; tile "decode" (one 16-row tile per KV
# pair: L x Hq/Hkv x 2 <= 16) or "verify" (64+ rows per KV head).
#   cuda      kernel A in CUDA (end of file: explicit cp.async pipeline), else Triton; "wg" (verify
#             tiles of up to 64 rows): the wgmma version, QK^T in E4M3 from shared memory
#             "ws": the warp-specialized wgmma version (a copying warp, a warpgroup per KV head
#             of a pair, mbarriers instead of CTA barriers)
#             "cl": the whole step in one CUDA kernel, a cluster of `cl` CTAs per (request, KV group)
#             merging its splits through DSMEM, then the norm-gate (no combine kernel)
#   cl        cluster size for cuda="cl" (the split count; 9-16 need non-portable clusters)
#   cl_fit    cuda="cl": shrink the cluster to the largest power of two that keeps one wave
#   lastm     cuda=True on decode tiles: the last CTA of each (request, KV pair) merges (no combine)
#   cuda_small_batch, cuda_small  keys of cuda_small replace the tuning's up to that batch
#   cuda_max_batch  the CUDA kernel A only up to this many requests (None: any), Triton above
#   block_n   keys per tile; min_chunk: fewest keys a split gets
#   pair      verify tiles: a KV-head pair per program (decode tiles always take a pair)
#   swap      16-row tiles computed transposed, S^T = K Q^T, so the key tile is the 64-row
#             operand wgmma needs (else Triton uses mma.sync for M = 16)
#   stages    software-pipeline stages of the key loop
#   maxnreg   register cap (ptxas .maxnreg). With 4 warps, 128 registers fit 4 CTAs per SM and
#             137 only 3, which spills a full layer's programs into one more wave
#   prefetch  page-table entries this many key blocks ahead (see the inner loop; 0 = none)
#   edges     masks only on the edge key blocks (a masked copy of the tile code per edge), else
#             one masked loop over every block
#   per_sm    sliding layers: programs per SM the split count aims for (one short wave)
#   waves     full layers: the split count fills this many whole waves of resident CTAs (SMs x
#             CTAs per SM, from the compiled kernel), so no last wave runs part-empty
TUNING = {
    ("sliding", "decode"): dict(cuda=False, block_n=64, min_chunk=128, pair=False, edges=False,
                                swap=False, stages=3, maxnreg=None, prefetch=0, warps=4, per_sm=2),
    ("full", "decode"): dict(cuda=False, block_n=64, min_chunk=128, pair=False, edges=False,
                             swap=False, stages=3, maxnreg=128, prefetch=0, warps=4, waves=1),
    ("sliding", "verify"): dict(cuda=False, block_n=64, min_chunk=128, pair=False, edges=True,
                                swap=False, stages=2, maxnreg=None, prefetch=0, warps=4, per_sm=2),
    ("full", "verify"): dict(cuda=False, block_n=64, min_chunk=128, pair=False, edges=True,
                             swap=False, stages=2, maxnreg=None, prefetch=0, warps=4, waves=1),
}
PACK_DECODE = True    # both KV heads of a pair in one tl.dot when the rows fit in 16
# q rounded to E4M3 (as FA3 does) and Q @ K^T on the FP8 keys as stored, wherever wgmma runs
# (64+-row tiles and swap-AB decode tiles). Triton's mma.sync path would upcast to FP16 anyway.
QK_FP8 = True
# Sliding layers (few, short programs): kernel A lets the combine launch at its start, not its end.
EARLY_TRIGGER_SLIDING = True
ON_GPU = os.environ.get("TRITON_INTERPRET") != "1"  # off in the CPU interpreter (local logic tests)
# The combine kernel launches while kernel A drains and waits on griddepcontrol.wait.
PDL = tl.constexpr(ON_GPU and gdc is not None)
# Inside a layer whose qkv producer launches kernel A with PDL (decode_layer): kernel A waits
# (griddepcontrol.wait) only before its first read of qkv, so its prologue overlaps the producer.
# Here (no producer) the wait returns at once. CUDA kernels: CUDA_EXTRA["GDCQ"].
TRITON_GDCQ = tl.constexpr(False)


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
def _attend_t(q, k, v, allowed, m, l, acc, s_scale, QK_FP8: tl.constexpr, MASKED: tl.constexpr):
    """_attend transposed: scores S^T = K Q^T [N, M] and accumulator acc^T [W, M], so the
    [N, W] key and value tiles are wgmma's 64-row operands. allowed is [N, M]."""
    s = tl.dot(k, tl.trans(q))
    if QK_FP8:
        s = s * s_scale
    if MASKED:
        s = tl.where(allowed, s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, 0))
    m_safe = m_new
    if MASKED:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m - m_safe)
    p = tl.exp2(s - m_safe[None, :])
    l = l * alpha + tl.sum(p, 0)
    acc = acc * alpha[None, :] + tl.dot(tl.trans(v), p.to(v.dtype))
    return m_new, l, acc


@triton.jit
def _block(i, hi, lo, b, kv0, p_row, q0, q1, m0, l0, acc0, m1, l1, acc1, q_scale,
           k_cache_ptr, v_cache_ptr, page_table_ptr, pt_stride, page_in,
           PAGE: tl.constexpr, WINDOW: tl.constexpr, ROW: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
           BLOCK_N: tl.constexpr, GROUPS: tl.constexpr, QK_FP8: tl.constexpr, SWAP: tl.constexpr,
           MASKED: tl.constexpr, HAS_PAGE: tl.constexpr):
    """Key block i (the BLOCK_N keys ending i * BLOCK_N before hi) for both tl.dot groups.
    MASKED blocks apply the causal and window masks and the block's clip at lo. With HAS_PAGE,
    page_in holds the block's page-table entries, loaded by the caller an iteration earlier."""
    q_dtype: tl.constexpr = q0.dtype
    n = tl.arange(0, BLOCK_N)
    c = tl.arange(0, W)
    t = hi - (i + 1) * BLOCK_N + n
    allowed = 0  # unused on unmasked blocks
    if MASKED:
        t_ok = t >= lo
        if HAS_PAGE:
            page = page_in
        else:
            page = tl.load(page_table_ptr + b * pt_stride + t // PAGE, mask=t_ok, other=0)
        if SWAP:
            allowed = t_ok[:, None] & (t[:, None] <= p_row[None, :])
            if WINDOW > 0:
                allowed = allowed & (t[:, None] > p_row[None, :] - WINDOW)
        else:
            allowed = t_ok[None, :] & (t[None, :] <= p_row[:, None])
            if WINDOW > 0:
                allowed = allowed & (t[None, :] > p_row[:, None] - WINDOW)
    else:
        t_ok = n < BLOCK_N
        if HAS_PAGE:
            page = page_in
        else:
            page = tl.load(page_table_ptr + b * pt_stride + t // PAGE)
    row = (page.to(tl.int64) * PAGE + t % PAGE) * ROW + kv0 * D
    k = tl.load(k_cache_ptr + row[:, None] + c[None, :], mask=t_ok[:, None], other=0.0).to(q_dtype)
    v = tl.load(v_cache_ptr + row[:, None] + c[None, :], mask=t_ok[:, None], other=0.0).to(tl.float16)
    if SWAP:
        m0, l0, acc0 = _attend_t(q0, k, v, allowed, m0, l0, acc0, q_scale, QK_FP8, MASKED)
    else:
        m0, l0, acc0 = _attend(q0, k, v, allowed, m0, l0, acc0, q_scale, QK_FP8, MASKED)
    if GROUPS == 2:
        k = tl.load(k_cache_ptr + row[:, None] + W + c[None, :], mask=t_ok[:, None], other=0.0).to(q_dtype)
        v = tl.load(v_cache_ptr + row[:, None] + W + c[None, :], mask=t_ok[:, None], other=0.0).to(tl.float16)
        if SWAP:
            m1, l1, acc1 = _attend_t(q1, k, v, allowed, m1, l1, acc1, q_scale, QK_FP8, MASKED)
        else:
            m1, l1, acc1 = _attend(q1, k, v, allowed, m1, l1, acc1, q_scale, QK_FP8, MASKED)
    return m0, l0, acc0, m1, l1, acc1


@triton.jit
def _gate(tok, head, rows_ok, qkv_ptr, QKV_STRIDE: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
          D: tl.constexpr):
    """sigmoid(gate logits) for [R, D] rows (token tok[i], query head head[i]), float32."""
    d = tl.arange(0, D)
    gate = tl.load(qkv_ptr + tok[:, None].to(tl.int64) * QKV_STRIDE + ((HQ + 2 * HKV) * D + head * D)[:, None]
                   + d[None, :], mask=rows_ok[:, None], other=0.0).to(tl.float32)
    return tl.sigmoid(gate)


@triton.jit
def _block_pages(pt_row, hi, j, lo, end, PAGE: tl.constexpr, BLOCK_N: tl.constexpr):
    """Page-table entries of key block j (zeros past lo, or if j >= end)."""
    t = hi - (j + 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    return tl.load(pt_row + t // PAGE, mask=(t >= lo) & (j < end), other=0)


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
                 PACK: tl.constexpr, M: tl.constexpr, BLOCK_N: tl.constexpr, ONE_SPLIT: tl.constexpr,
                 QK_FP8: tl.constexpr, SWAP: tl.constexpr, PREFETCH: tl.constexpr, EARLY_TRIGGER: tl.constexpr,
                 EDGES: tl.constexpr):
    if PDL and EARLY_TRIGGER:
        # Let the combine kernel launch now and load its gates while this grid runs; it still
        # waits for this grid to finish before reading the partials (griddepcontrol.wait).
        gdc.gdc_launch_dependents()
    split = tl.program_id(0)
    kv0 = tl.program_id(1) * KVP
    b = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    R: tl.constexpr = L * G  # query rows per KV head
    W: tl.constexpr = PACK * D  # columns of one tl.dot group
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
    # A program runs KVP // PACK tl.dot groups; each packs PACK KV heads into one block-diagonal
    # [M, PACK * D] tile. Row r is head part = r // R of the pack, its token (r % R) // G and query
    # head (r % R) % G. Rows past PACK * R pad the tile to the 16 rows tl.dot needs (they repeat
    # real rows and are never stored).
    r = tl.arange(0, M)
    rows_ok = r < PACK * R
    part = tl.minimum(r // R, PACK - 1)
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
    if PDL and TRITON_GDCQ:
        gdc.gdc_wait()  # qkv comes from the layer's producer
    q0 = (_load_heads(qkv_ptr, pos_ptr, rope_ptr, tok, head0 * D, part, rows_ok, eps, QKV_STRIDE, D, W, ROPE)
          * q_mul).to(q_dtype)
    m0 = tl.full([M], float("-inf"), tl.float32)
    l0 = tl.zeros([M], tl.float32)
    if SWAP:
        acc0 = tl.zeros([W, M], tl.float32)
    else:
        acc0 = tl.zeros([M, W], tl.float32)
    GROUPS: tl.constexpr = KVP // PACK
    head1 = head0 + PACK * G
    if GROUPS == 2:
        q1 = (_load_heads(qkv_ptr, pos_ptr, rope_ptr, tok, head1 * D, part, rows_ok, eps, QKV_STRIDE, D, W,
                          ROPE) * q_mul).to(q_dtype)
        m1 = tl.full([M], float("-inf"), tl.float32)
        l1 = tl.zeros([M], tl.float32)
        if SWAP:
            acc1 = tl.zeros([W, M], tl.float32)
        else:
            acc1 = tl.zeros([M, W], tl.float32)
    else:  # unused stand-ins, so _block has one signature
        q1, m1, l1, acc1 = q0, m0, l0, acc0

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
        # With PREFETCH = d, the page-table entries of block i + d are loaded during block i: the
        # cache rows' addresses then come from a loop-carried value, not from a load in the same
        # iteration, so the software pipeliner can keep key and value copies in flight. It issues
        # them stages - 1 blocks ahead, so d should be at least that.
        pt_row = page_table_ptr + b * pt_stride
        page = _block_pages(pt_row, hi, inner_lo, lo, inner_hi, PAGE, BLOCK_N)
        page_b = _block_pages(pt_row, hi, inner_lo + 1, lo, inner_hi, PAGE, BLOCK_N)
        for i in range(inner_lo, inner_hi):
            page_cur = page
            if PREFETCH == 1:
                page = _block_pages(pt_row, hi, i + 1, lo, inner_hi, PAGE, BLOCK_N)
            if PREFETCH == 2:
                page = page_b
                page_b = _block_pages(pt_row, hi, i + 2, lo, inner_hi, PAGE, BLOCK_N)
            m0, l0, acc0, m1, l1, acc1 = _block(
                i, hi, lo, b, kv0, p_row, q0, q1, m0, l0, acc0, m1, l1, acc1, q_scale, k_cache_ptr, v_cache_ptr,
                page_table_ptr, pt_stride, page_cur, PAGE, WINDOW, ROW, D, W, BLOCK_N,
                GROUPS, QK_FP8, SWAP,
                False, PREFETCH > 0)
        for e in tl.static_range(EDGE + 1):
            # e = 0: block 0; e >= 1: block n_blocks - e, unless block 0 or inside the unmasked range.
            if e == 0:
                i = 0
                take = (first == 0) & (last > 0)  # empty splits (last <= 0) have first == 0 too
            else:
                i = n_blocks - e
                take = (i >= tl.maximum(first, 1)) & (i < last) & (i >= inner_hi)
            if take:
                m0, l0, acc0, m1, l1, acc1 = _block(
                    i, hi, lo, b, kv0, p_row, q0, q1, m0, l0, acc0, m1, l1, acc1, q_scale, k_cache_ptr, v_cache_ptr,
                    page_table_ptr, pt_stride, page, PAGE, WINDOW, ROW, D, W, BLOCK_N,
                    GROUPS, QK_FP8, SWAP,
                    True, False)
    else:
        # One masked loop over every block: less code (one copy of the tile), for short loops.
        # With PREFETCH, the same page-table prefetch as the unmasked loop.
        pt_row = page_table_ptr + b * pt_stride
        page = _block_pages(pt_row, hi, first, lo, last, PAGE, BLOCK_N)
        page_b = _block_pages(pt_row, hi, first + 1, lo, last, PAGE, BLOCK_N)
        for i in range(first, last):
            page_cur = page
            if PREFETCH == 1:
                page = _block_pages(pt_row, hi, i + 1, lo, last, PAGE, BLOCK_N)
            if PREFETCH == 2:
                page = page_b
                page_b = _block_pages(pt_row, hi, i + 2, lo, last, PAGE, BLOCK_N)
            m0, l0, acc0, m1, l1, acc1 = _block(
                i, hi, lo, b, kv0, p_row, q0, q1, m0, l0, acc0, m1, l1, acc1, q_scale, k_cache_ptr, v_cache_ptr,
                page_table_ptr, pt_stride, page_cur, PAGE, WINDOW, ROW, D, W, BLOCK_N,
                GROUPS, QK_FP8, SWAP,
                True, PREFETCH > 0)

    if SWAP:
        acc0 = tl.trans(acc0)
    _finish(acc0, l0, m0, v_scale, split, num_splits, tok, head0, part, rows_ok, qkv_ptr, out_ptr, ws_o_ptr,
            ws_lse_ptr, eps, QKV_STRIDE, HQ, HKV, D, PACK, ONE_SPLIT)
    if GROUPS == 2:
        if SWAP:
            acc1 = tl.trans(acc1)
        _finish(acc1, l1, m1, v_scale, split, num_splits, tok, head1, part, rows_ok, qkv_ptr, out_ptr, ws_o_ptr,
                ws_lse_ptr, eps, QKV_STRIDE, HQ, HKV, D, PACK, ONE_SPLIT)
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


# ---------------------------------------------------------------------------------------------
# Kernel A in CUDA (TUNING "cuda"): the same split-KV partials as the Triton kernel A, with an
# explicit cp.async pipeline. Nsight Compute on the Triton kernel shows each key block paying the
# page-table load and then the key/value copies in series: Triton's pipeliner cannot overlap a
# gather whose addresses come from another load. Here a ring of STAGES raw E4M3 tiles is filled
# with cp.async, the page-table entries go 2 * STAGES - 1 blocks ahead (also cp.async), and each
# iteration waits for one group and syncs once. The math is FlashAttention-2 style in registers:
# mma.sync m16n8k8 (FP16, FP32 accumulate), Q fragments by ldmatrix, K and V fragments read from
# the E4M3 tiles and converted in registers. Compiled in-process with NVRTC (cuda.bindings, in
# the judge's image; no subprocess) and launched with cuLaunchKernelEx.
# ---------------------------------------------------------------------------------------------

_CUDA_FUNCS = {}
_CUDA_FAILED = set()
_CUDA_CTAS = {}


def _cuda_include_dirs():
    """CUDA headers (cuda_fp8.h, mma.h, cooperative_groups.h, CCCL): nvidia/cu13 in the judge's
    image (a namespace package, so every path of it is tried), else the toolkit."""
    candidates = [os.environ["OKC_CUDA_INCLUDE"]] if "OKC_CUDA_INCLUDE" in os.environ else []  # local tests
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


KERNEL_A_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

// -D: HQ HKV L KVP WARP_M WARP_K STAGES PAGE WINDOW ROPE MINB (CTAs per SM the registers must fit)
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
#define NREAL (KVP * L * G)
#define PRING (2 * STAGES)             // page-table ring slots
#define NEG_INF __int_as_float(0xff800000)
// CL > 0: the whole step in one kernel. A cluster of CL CTAs per (request, KV group) takes the CL
// splits; each keeps its normalized partial in shared memory, and after a cluster barrier every CTA
// merges a few output rows through DSMEM, then the norm-gate and the bfloat16 store (no combine).
#ifndef LASTM
#define LASTM 0                        // 1 (decode, no cluster): the last CTA of a (request, KV pair) merges
#endif
#ifndef GDCQ
#define GDCQ 0                         // 1 (inside a layer, launched with PDL): griddepcontrol.wait after the
                                       // K/V chain is issued and before any read of qkv or write of the cache
#endif
#ifndef EPIW
#define EPIW 0                         // 1: the partials' weights once per row (epilogue)
#endif
#ifndef KVEARLY
#define KVEARLY 0                      // 1: the first blocks' K/V copies before the queries are done
#endif
#ifndef CL
#define CL 0
#endif
// NEWSMEM 1: the split holding block 0 also writes the step's new keys and values (E4M3) straight
// into its stage-0 tile and skips their copies, so it neither fences nor reads them back.
#ifndef NEWSMEM
#define NEWSMEM 0
#endif
// TSTAMP (diagnostic): thread 0 of each CTA stores %globaltimer at each phase into ws_o (as u64,
// [CTA][8]).
// EARLYQ 1: the queries' loads go first (they don't depend on seq_lens), then the page entries,
// and the block-0 split writes the step's keys and values last, overlapped with the rest.
#ifndef EARLYQ
#define EARLYQ 0
#endif
#ifndef TSTAMP
#define TSTAMP 0
#endif
#define TS(k) do { if (TSTAMP && tid == 0) { unsigned long long t_; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    const unsigned cta_ = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; \
    ((unsigned long long*)ws_o)[cta_ * 8 + (k)] = t_; } } while (0)
// Memory-path experiments (defaults = the reviewed kernel): KVORD 1 issues a block's K rows before
// its V rows; CPCA 1 copies through L1 (cp.async.ca); GSWAP 1 puts the KV group on blockIdx.x.
#ifndef KVORD
#define KVORD 0
#endif
#ifndef CPCA
#define CPCA 0
#endif
#ifndef GSWAP
#define GSWAP 0
#endif

#define OFF_Q 0
#define OFF_K (OFF_Q + ROWS * QP)
#define OFF_V (OFF_K + STAGES * BN * RP)
#define OFF_P (OFF_V + STAGES * BN * RP)
// CL: the partials every rank sends to the owner of each merge row (rho on rank rho % CL, slot
// rho / CL): [MROWS][CL][D + 1] floats (o, then the lse), apart from everything the main loop uses.
#define MROWS (CL ? (KVP * L * G + CL - 1) / (CL ? CL : 1) : 0)
#define OFF_MG (OFF_P + PRING * BN * 4)
#define SMEM_BYTES (OFF_MG + MROWS * (CL ? CL : 1) * (D + 1) * 4)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}
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
// D += A B over k = 32, A and B in E4M3 (sm89+). Below sm89 (local logic tests), four k = 8 FP16
// MMAs over the same registers: slice s takes bytes 2 (s & 1), + 1 of registers a[(s >> 1) 2],
// a[(s >> 1) 2 + 1] and b[s >> 1], the same k values in A and B, so the sum is the same.
__device__ __forceinline__ void mma_k32_e4m3(float* d, const unsigned* a, const unsigned* b) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
#pragma unroll
    for (int sl = 0; sl < 4; ++sl) {
        const int r = (sl >> 1) * 2, sh = 16 * (sl & 1);
        mma_k8(d, e4m3x2_to_h2((unsigned short)(a[r] >> sh)), e4m3x2_to_h2((unsigned short)(a[r + 1] >> sh)),
               e4m3x2_to_h2((unsigned short)(b[sl >> 1] >> sh)));
    }
#endif
}
// cp.async: 16 and 4 bytes, global to shared (sm80+; plain copies below, for local logic tests).
// cp.async goes through the generic proxy, so the __syncthreads after cp.async.wait_group is
// enough; a TMA (async proxy) version would also need fence.proxy.async before reusing a stage.
__device__ __forceinline__ void cp16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
#if CPCA
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
#endif
#else
    *(uint4*)dst = *(const uint4*)src;
#endif
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    *(int*)dst = *(const int*)src;
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
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
    __nv_bfloat16* __restrict__ out, int* __restrict__ cnt)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [ROWS][QP] E4M3, dims permuted (below)
    unsigned char* Kr = smem + OFF_K;   // [STAGES][BN][RP] E4M3
    unsigned char* Vr = smem + OFF_V;
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = GSWAP ? blockIdx.y : blockIdx.x, kv0 = (GSWAP ? blockIdx.x : blockIdx.y) * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    TS(0);
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
            if (!GDCQ) qx[i] = __bfloat1622float2(src[lane]);   // GDCQ: after the wait, below
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
            if (!GDCQ) {
                kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
                vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
            }
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

#if CL && __CUDA_ARCH__ >= 900
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
            if (!GDCQ) gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if LASTM
    // Gates of the rows this CTA may merge (any CTA can be the last), loaded now: a warp per row.
    constexpr int LGR = (KVP * L * G + NWARPS - 1) / NWARPS;
    float2 gate_l[LGR];
#pragma unroll
    for (int i = 0; i < LGR; ++i) {
        const int rho = warp + NWARPS * i;
        gate_l[i] = make_float2(0.f, 0.f);
        if (rho < KVP * L * G) {
            const int rr = rho % (L * G), head = (kv0 + rho / (L * G)) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + rr / G) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            gate_l[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#if !EARLYQ
    // The split holding block 0 writes the step's new keys and values to the cache first; its own
    // copies of block 0 then read them back (after the fence and barrier below). NEWSMEM: it puts
    // them in its stage-0 tile too (rows BN - L + j of block 0) and copies none of them.
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
            if (NEWSMEM) {
                *(__nv_fp8x2_storage_t*)(smem + OFF_K + (BN - L + j) * RP + h * D + 2 * lane) = k8;
                *(__nv_fp8x2_storage_t*)(smem + OFF_V + (BN - L + j) * RP + h * D + 2 * lane) = v8;
            }
        }
        if (!NEWSMEM) __threadfence();
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
            for (int idx = tid; idx < (KVORD ? 2 : 1) * BN * CPK; idx += NT) {
                const int kv = KVORD ? idx / (BN * CPK) : 0, i = KVORD ? idx % (BN * CPK) : idx;
                int key = i / CPK, c16 = (i % CPK) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = Kr + ((j % STAGES) * BN + key) * RP + c16;
                unsigned char* dv = Vr + ((j % STAGES) * BN + key) * RP + c16;
                if (t < lo) {  // clipped rows: zeros (stale bytes could encode NaN, and 0 * NaN = NaN)
                    if (!KVORD || kv == 0) *(uint4*)dk = make_uint4(0, 0, 0, 0);
                    if (!KVORD || kv == 1) *(uint4*)dv = make_uint4(0, 0, 0, 0);
                    continue;
                }
                if (NEWSMEM && first + j == 0 && t >= seq) continue;   // the step's own rows: written above
                long row = (long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE;
                long at = row * ROW + kv0 * D + c16;
                if (!KVORD || kv == 0) cp16(dk, k_cache + at);
                if (!KVORD || kv == 1) cp16(dv, v_cache + at);
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
#if KVEARLY
    // The first blocks' copies as soon as their page entries land, while the RoPE loads above are in
    // flight: the K/V chain (seq_lens -> pages -> rows) no longer waits for the queries' chain.
    cp_wait<0>();
    __syncthreads();
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }
#endif
#if GDCQ
    static_assert(!GDCQ || (EARLYQ && NEWSMEM && KVEARLY), "GDCQ: the step's own rows from shared memory, K/V first");
    // Everything before this line reads only what the layer's earlier kernels don't write (seq_lens,
    // the page table, positions, cache_loc, the RoPE table, the cache's older rows; NEWSMEM skips the
    // step's own rows). Now the qkv producer's results: q, the new k/v (and the gate).
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        int r = warp + i * NWARPS, h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (r < ROWS && rr < L * G) {
            int j = rr / G, g = rr % G, tok = b * L + j;
            const __nv_bfloat162* src = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + ((kv0 + h) * G + g) * D);
            qx[i] = __bfloat1622float2(src[lane]);
        }
    }
#pragma unroll
    for (int i = 0; i < KR; ++i) {
        const int w = warp + i * NWARPS, h = w / L, j = w % L, tok = b * L + j, kvh = kv0 + h;
        if (w < KVP * L) {
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            kxs[i] = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
            vxs[i] = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
        }
    }
#if CL
#pragma unroll
    for (int i = 0; i < GR; ++i) {
        const int rho = split + CL * (warp + NWARPS * i);
        if (rho < KVP * L * G) {
            const int h = rho / (L * G), rr = rho % (L * G), j = rr / G, head = (kv0 + h) * G + rr % G;
            const __nv_bfloat16* gp = qkv + (long)(b * L + j) * QKV_STRIDE + (HQ + 2 * HKV) * D + head * D;
            gate_r[i] = make_float2(__bfloat162float(gp[lane]), __bfloat162float(gp[lane + 32]));
        }
    }
#endif
#endif
#if EARLYQ
    static_assert(!EARLYQ || NEWSMEM, "EARLYQ: the block-0 split reads its own rows from shared memory");
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
#if KVEARLY
    __syncthreads();
    TS(1);   // queries in shared memory (the first blocks' copies already in flight)
#else
    cp_wait<0>();
    __syncthreads();
    TS(1);   // queries in shared memory, the first page entries landed
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }
#endif

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
        if (j == 0) TS(2);        // block 0's keys and values landed
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
    TS(3);   // main loop done

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
#if EPIW
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
#endif
#if CL && __CUDA_ARCH__ >= 900
    cooperative_groups::this_cluster().barrier_wait();   // every rank has started (arrived at the top)
#endif
    for (int idx = tid; idx < ROWS * D; idx += NT) {
        int r = idx / D, d = idx % D, mr = r / 16, rl = r % 16;
        int h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (rr >= L * G) continue;
#if EPIW
        float val = 0.f;
#pragma unroll
        for (int k = 0; k < WARP_K; ++k) val += Ww[r * (WARP_K + 1) + k] * Wo[((k * WARP_M + mr) * 16 + rl) * D + d];
        const float lse = Ww[r * (WARP_K + 1) + WARP_K];
#else
        float mm = NEG_INF;
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        float safe = mm == NEG_INF ? 0.f : mm, den = 0.f, acc = 0.f;
        for (int k = 0; k < WARP_K; ++k) {
            int w = k * WARP_M + mr;
            float wgt = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt * Wm[(w * 16 + rl) * 2 + 1];
            acc += wgt * Wo[(w * 16 + rl) * D + d];
        }
        const float val = den > 0.f ? acc / den * v_scale : 0.f, lse = den > 0.f ? mm + __log2f(den) : NEG_INF;
#endif
#if CL
        // Row r's partial straight into its owner's buffer (rank rho % CL, slot rho / CL).
        {
            const int rho = h * (L * G) + rr;
#if __CUDA_ARCH__ >= 900
            float* mg = cooperative_groups::this_cluster().map_shared_rank((float*)(smem + OFF_MG), rho % CL);
#else
            float* mg = (float*)(smem + OFF_MG);
#endif
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
#if LASTM
    static_assert(!LASTM || (!CL && WARP_M == 1), "LASTM: decode tiles, no cluster");
    // The last CTA of this (request, KV pair) to finish merges every split's partials (from L2), then
    // the norm-gate and the bfloat16 store, as the combine kernel; it resets the counter for the next
    // call (every split has counted by then).
    __threadfence();   // this CTA's partials, before its count
    __syncthreads();
    int* last_flag = (int*)(smem + OFF_K);   // Wm, unused from here on
    if (tid == 0) {
        int* c = cnt + b * gridDim.y + blockIdx.y;
        const int lastc = atomicAdd(c, 1) == num_splits - 1;
        if (lastc) *c = 0;
        *last_flag = lastc;
    }
    __syncthreads();
    if (!*last_flag) return;
    __threadfence();   // the other CTAs' partials, after their counts
    // Every load independent, all issued before use: the lse table, then each row's split weights
    // (one thread per row), then 4 dims per thread with the splits unrolled (16 L2 loads in flight).
    constexpr int R = KVP * L * G;   // 16 rows: 8 per head
    float* Lw = (float*)(smem + OFF_K) + 4;   // [R][num_splits] lse, then weights
    float* Ov = Lw + R * 64;                  // [R][D] merged rows (num_splits <= 64)
    for (int i = tid; i < R * num_splits; i += NT) {
        const int rho = i / num_splits, c = i % num_splits, rr = rho % (L * G);
        const long tok = b * L + rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        Lw[i] = __ldcg(ws_lse + (tok * HQ + head) * num_splits + c);
    }
    __syncthreads();
    if (tid < R) {
        float mm = NEG_INF, den = 0.f;
        for (int c = 0; c < num_splits; ++c) mm = fmaxf(mm, Lw[tid * num_splits + c]);
        const float safe = mm == NEG_INF ? 0.f : mm;
        for (int c = 0; c < num_splits; ++c) {   // empty splits: lse -inf, weight 0
            const float w = exp2f(Lw[tid * num_splits + c] - safe);
            Lw[tid * num_splits + c] = w;
            den += w;
        }
        for (int c = 0; c < num_splits; ++c) Lw[tid * num_splits + c] /= den;
    }
    __syncthreads();
    for (int e = tid; e < R * (D / 4); e += NT) {
        const int rho = e / (D / 4), q = e % (D / 4), rr = rho % (L * G);
        const long tok = b * L + rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const float4* src = (const float4*)(ws_o + (tok * HQ + head) * num_splits * D) + q;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int c0 = 0; c0 < num_splits; c0 += 16) {
            float4 v[16];
#pragma unroll
            for (int c = 0; c < 16; ++c)
                v[c] = c0 + c < num_splits ? __ldcg(src + (c0 + c) * (D / 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float w = c0 + c < num_splits ? Lw[rho * num_splits + c0 + c] : 0.f;
                acc.x += w * v[c].x; acc.y += w * v[c].y; acc.z += w * v[c].z; acc.w += w * v[c].w;
            }
        }
        *(float4*)(Ov + rho * D + 4 * q) = acc;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < LGR; ++i) {   // norm-gate and the bfloat16 store, a warp per row
        const int rho = warp + NWARPS * i;
        if (rho >= R) break;
        const int rr = rho % (L * G), j = rr / G, head = (kv0 + rho / (L * G)) * G + rr % G;
        const long tok = b * L + j;
        const float o0 = Ov[rho * D + lane], o1 = Ov[rho * D + lane + 32];
        const float inv = rsqrtf(warp_sum(o0 * o0 + o1 * o1) / D + eps);
        const float g0 = 1.f / (1.f + __expf(-gate_l[i].x)), g1 = 1.f / (1.f + __expf(-gate_l[i].y));
        out[tok * (HQ * D) + head * D + lane] = __float2bfloat16(o0 * inv * g0);
        out[tok * (HQ * D) + head * D + lane + 32] = __float2bfloat16(o1 * inv * g1);
    }
#endif
#if CL
    // Merge: real row rho on rank rho % CL, warp (rho / CL) % NWARPS; lane holds dims lane, lane + 32.
    const int rank = split;   // the cluster spans the grid's x
#if __CUDA_ARCH__ >= 900
    TS(4);   // partials sent
    cooperative_groups::this_cluster().sync();   // every rank's partials are in their owners' buffers
    TS(5);
#else
    __syncthreads();
#endif
    static_assert(__CUDA_ARCH__ >= 900 || CL == 1, "clusters need sm90");
    static_assert(!CL || !GSWAP, "cl: the cluster rank is the split, blockIdx.x");
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
    TS(6);
#endif
}
'''

# Verify tiles (one KV head, up to 64 rows) with wgmma: experimentos/verify_wg.py, copied by
# sincroniza_cuda.py.
KERNEL_A_WG_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// -D: HQ HKV L STAGES PAGE WINDOW ROPE [KVP REGLD DSWAP]
#define D 64
#define BN 64                          // keys per block
#define ROWS 64                        // query rows per KV head: one warpgroup m64
#ifndef KVP
#define KVP 1                          // KV heads per CTA, a warpgroup each; 2: the copies take whole cache rows
#endif
#define NT (128 * KVP)
#define NWARPS (NT / 32)
#define KROW (KVP * D)                 // bytes of each key's row the CTA copies
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define TILE (BN * D)                  // bytes of an E4M3 tile (64 x 64)
#define VP (D + 8)                     // FP16 V rows (halves), padded: ldmatrix without bank conflicts
#define PRING (2 * STAGES)
#define NEG_INF __int_as_float(0xff800000)

#define OFF_Q 0
#define OFF_K (OFF_Q + KVP * TILE)
#define OFF_V (OFF_K + STAGES * KVP * TILE)
#define OFF_VH (OFF_V + STAGES * KVP * TILE)
#define OFF_P (OFF_VH + KVP * BN * VP * 2)
#define SMEM_BYTES (OFF_P + PRING * BN * 4)

static_assert(L * G <= ROWS, "one warpgroup holds the rows");
#ifndef REGLD
#define REGLD 0   // 1 (experiment): K/V through registers (plain loads after the barrier, stores at the end)
#endif
#define KVI (BN * (KROW / 16) / NT)    // 16-byte K (and V) chunks per thread and block
#ifndef PIPE
#define PIPE 0    // 1 (experiment): QK^T of the next block overlaps this block's softmax and P V
#endif
#ifndef PVWG
#define PVWG 0    // 1 (experiment): P V with wgmma too (P from registers, V^T in FP16 core matrices)
#endif

// No-swizzle core-matrix layout of a 64 x 64-byte tile: 8 x 16-byte core matrices, contiguous.
__device__ __forceinline__ int cm(int row, int byte) {
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
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    *(uint4*)dst = *(const uint4*)src;
#endif
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    *(int*)dst = *(const int*)src;
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
// Generic-proxy writes (stores, cp.async) to shared memory, made visible to wgmma (async proxy).
__device__ __forceinline__ void fence_async_smem() {
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
#if __CUDA_ARCH__ >= 900
#ifndef DSWAP
#define DSWAP 0   // 1: LBO and SBO swapped (bench check of the descriptor convention)
#endif
__device__ __forceinline__ unsigned long long wg_desc(const void* p) {   // no swizzle, LBO 128, SBO 512
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | ((DSWAP ? 32ull : 8ull) << 16)
         | ((DSWAP ? 8ull : 32ull) << 32);
}
__device__ __forceinline__ unsigned long long wg_desc_v(const void* p) {   // V^T tile: LBO 128, SBO 1024
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | (8ull << 16) | (64ull << 32);
}
// O (64 x 64, FP32) += P (64 x 16 keys, FP16, A from registers as mma.sync's m16n8k16 A) V (16 keys x 64).
__device__ __forceinline__ void wg_pv(float* d, const unsigned* a, unsigned long long db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15,"
                 "%16, %17, %18, %19, %20, %21, %22, %23,"
                 "%24, %25, %26, %27, %28, %29, %30, %31"
                 "}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                   "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                   "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(1));
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
#endif
__device__ __forceinline__ float2 norm_rope(float2 x, float2 cs, float eps) {
    float inv = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + eps);
    x.x *= inv; x.y *= inv;
    return make_float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

extern "C" __global__ void __launch_bounds__(NT, PIPE ? 3 : 4 / KVP) verify_wg(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [KVP][64 rows x 64 B] E4M3, core-matrix layout
    unsigned char* Ks = smem + OFF_K;   // [STAGES][KVP][64 keys x 64 B] E4M3, core-matrix layout
    unsigned char* Vr = smem + OFF_V;   // [STAGES][64 keys][KROW] E4M3, row-major
    __half* Vh = (__half*)(smem + OFF_VH);   // [KVP][64 keys][VP] FP16, the block being used
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = blockIdx.x, kv0 = blockIdx.y * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    const int wgi = warp >> 2, wl = warp & 3;   // warpgroup (its KV head kv0 + wgi), warp in it
    const float k_scale = *k_scale_ptr, v_scale = *v_scale_ptr;
    const int seq = seq_lens[b], hi = seq + L;
    const int lo = WINDOW > 0 ? max(seq - WINDOW + 1, 0) : 0;
    // Blocks counted back from hi: block i holds keys [hi - (i + 1) BN, hi - i BN), clipped at lo.
    // Split s takes blocks [n - (s + 1) per, n - s per), clipped at 0 (as the Triton kernel).
    const int n_blocks = (hi - lo + BN - 1) / BN;
    const int per = (n_blocks + num_splits - 1) / num_splits;
    const int last = n_blocks - split * per, first = max(last - per, 0);
    const int nb = max(last - first, 0);

    // The split holding block 0 writes the step's new keys and values to the cache first; its own
    // copies of block 0 then read them back (after the fence and barrier below).
    if (first == 0 && last > 0) {
        for (int w = warp; w < KVP * L; w += NWARPS) {
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

    auto issue_pages = [&](int j) {   // page entries of block j into ring slot j % PRING
        if (j < nb) {
            for (int key = tid; key < BN; key += NT) {
                int t = hi - (first + j + 1) * BN + key;
                if (t >= lo) cp4(Pg + (j % PRING) * BN + key, page_table + (long)b * pt_stride + t / PAGE);
            }
        }
    };
    // 16-byte chunk idx of a block: key idx / (KROW / 16), byte c of its row (head c / D).
    auto k_dst = [&](int j, int key, int c) { return Ks + ((j % STAGES) * KVP + c / D) * TILE + cm(key, c % D); };
    auto v_dst = [&](int j, int key, int c) { return Vr + (j % STAGES) * KVP * TILE + key * KROW + c; };
    auto issue_kv = [&](int j) {      // K (core-matrix layout) and V (row-major) of block j, stage j % STAGES
        if (j < nb) {
            for (int idx = tid; idx < BN * (KROW / 16); idx += NT) {
                int key = idx / (KROW / 16), c16 = (idx % (KROW / 16)) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = k_dst(j, key, c16);
                unsigned char* dv = v_dst(j, key, c16);
                if (t < lo) {  // clipped rows: zeros (stale bytes could encode NaN, and 0 * NaN = NaN)
                    *(uint4*)dk = make_uint4(0, 0, 0, 0);
                    *(uint4*)dv = make_uint4(0, 0, 0, 0);
                    continue;
                }
                long at = ((long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE) * ROW + kv0 * D + c16;
                cp16(dk, k_cache + at);
                cp16(dv, v_cache + at);
            }
        }
    };

    // REGLD: block j's K and V chunks into registers (plain loads), then into stage j % STAGES.
    auto load_kv = [&](int j, uint4* r) {
        if (j < nb) {
#pragma unroll
            for (int i = 0; i < KVI; ++i) {
                int idx = tid + i * NT, key = idx / (KROW / 16), c16 = (idx % (KROW / 16)) * 16;
                int t = hi - (first + j + 1) * BN + key;
                r[2 * i] = r[2 * i + 1] = make_uint4(0, 0, 0, 0);
                if (t >= lo) {
                    long at = ((long)Pg[(j % PRING) * BN + key] * PAGE + t % PAGE) * ROW + kv0 * D + c16;
                    r[2 * i] = *(const uint4*)(k_cache + at);
                    r[2 * i + 1] = *(const uint4*)(v_cache + at);
                }
            }
        }
    };
    auto store_kv = [&](int j, const uint4* r) {
        if (j < nb) {
#pragma unroll
            for (int i = 0; i < KVI; ++i) {
                int idx = tid + i * NT, key = idx / (KROW / 16), c16 = (idx % (KROW / 16)) * 16;
                *(uint4*)k_dst(j, key, c16) = r[2 * i];
                *(uint4*)v_dst(j, key, c16) = r[2 * i + 1];
            }
        }
    };

    for (int j = 0; j < 2 * STAGES - 1; ++j) issue_pages(j);
    cp_commit();
    // Queries meanwhile: RMSNorm, RoPE, E4M3 at unit RMS (16 rows per warp; rows past L G are zero).
    // Row r: head kv0 + r / ROWS, row r % ROWS of its tile.
    float2 qx[ROWS / 4], qcs[ROWS / 4];
    long qpos[ROWS / 4];
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = warp + i * NWARPS, rr = r % ROWS;
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (rr < L * G) {
            int tok = b * L + rr / G, head = (kv0 + r / ROWS) * G + rr % G;
            const __nv_bfloat162* src = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + head * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = warp + i * NWARPS, rr = r % ROWS;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (rr >= L * G) x = make_float2(0.f, 0.f);
        *(__nv_fp8x2_storage_t*)(Qs + (r / ROWS) * TILE + cm(rr, 2 * lane)) =
            __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    }
    cp_wait<0>();
    fence_async_smem();
    __syncthreads();
    for (int j = 0; j < STAGES - 1; ++j) {
        if (REGLD) { uint4 r[2 * KVI]; load_kv(j, r); store_kv(j, r); } else issue_kv(j);
        cp_commit();
    }

    const float qsc = q_mult * k_scale;
    int ptok[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) ptok[e] = seq + min((wl * 16 + grp + e * 8) / G, L - 1);
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    // One key block's steps: QK^T issued (async on sm90), V to FP16, the wait for QK^T, and the
    // softmax and P V of a block whose scores are in s.
    auto qk_issue = [&](float* s, int j) {
            const unsigned char* K = Ks + ((j % STAGES) * KVP + wgi) * TILE;   // this warpgroup's head
            const unsigned char* Q = Qs + wgi * TILE;
            // S = Q K^T: accumulator element nt 4 + c is row wl 16 + grp + 8 (c >> 1), key nt 8 + 2 tig + (c & 1).
#if __CUDA_ARCH__ >= 900
#pragma unroll
            for (int i = 0; i < 32; ++i) {   // defined inputs for the "+f" operands (scale-d 0 ignores them)
                s[i] = 0.f;
                asm volatile("" : "+f"(s[i]) :: "memory");
            }
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
            wg_qk(s, wg_desc(Q), wg_desc(K), 0);
            wg_qk(s, wg_desc(Q + 256), wg_desc(K + 256), 1);   // dims 32 .. 63: two core matrices on
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                float acc = 0.f;
                for (int c = 0; c < D; ++c) {
                    __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                    __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                    acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                }
                s[i] = acc;
            }
#endif
    };
    auto qk_wait = [&](float* s) {
#if __CUDA_ARCH__ >= 900
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(s[i]) :: "memory");
#endif
    };
    auto convert_v = [&](int j) {
        const unsigned char* V = Vr + (j % STAGES) * KVP * TILE;
            // Meanwhile V to FP16, once per block.
#if PVWG
            // V^T [dims][keys] in FP16 core matrices (K-major for the P V wgmma): thread tid takes keys
            // 2 kp, 2 kp + 1 of its head and dims q + 4 i, so a warp's 4-byte stores spread over the banks.
            {
                const int h = tid / 128, kp = (tid % 128) >> 2, q = tid & 3;
                const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
                const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
                unsigned char* vt = (unsigned char*)Vh + h * (BN * VP * 2);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 a = ra[c], bq = rb[c];
                    const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int d = (c * 4 + i) * 4 + q;   // byte q of word c 4 + i
                        *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                            e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                    }
                }
            }
#else
            {
                const int key = tid / (KROW / 32), c0 = (tid % (KROW / 32)) * 32;
                const uint4 a = *(const uint4*)(V + key * KROW + c0), c = *(const uint4*)(V + key * KROW + c0 + 16);
                const unsigned w[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
                unsigned h[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    h[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                    h[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                }
                uint4* dst = (uint4*)(Vh + ((c0 / D) * BN + key) * VP + c0 % D);
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
            }
#endif
    };
    auto consume = [&](float* s, int j) {
        const int kb0 = hi - (first + j + 1) * BN;     // position of key 0 of this block
            // Scores: the scale, and the mask only where the block reaches past lo, the causal edge or
            // the window (warp-uniform).
            const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
            float mx[2] = {NEG_INF, NEG_INF};
            if (edge) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                    bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                    s[i] = ok ? s[i] * qsc : NEG_INF;
                    mx[e] = fmaxf(mx[e], s[i]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    s[i] *= qsc;
                    mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], s[i]);
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
            unsigned pa[4][4];   // A fragments of P for k-tiles of 16 keys
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                float p0 = exp2f(s[nt * 4] - safe[0]), p1 = exp2f(s[nt * 4 + 1] - safe[0]);
                float p2 = exp2f(s[nt * 4 + 2] - safe[1]), p3 = exp2f(s[nt * 4 + 3] - safe[1]);
                rs[0] += p0 + p1; rs[1] += p2 + p3;
                pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);       // row grp, keys 2 tig (+ 8)
                pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);   // row grp + 8
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
#if PVWG
            {
                const unsigned char* vt = (const unsigned char*)Vh + wgi * (BN * VP * 2);
#if __CUDA_ARCH__ >= 900
                float* od = &o[0][0];
#pragma unroll
                for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
                asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)   // 16 keys per wgmma: two core matrices on along K
                    wg_pv(od, pa[kk], wg_desc_v(vt + kk * 256));
                asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
                for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
#else
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                    for (int n = 0; n < 8; ++n) {   // B fragments from the same V^T layout
                        const int d = n * 8 + grp, k0 = kk * 16 + 2 * tig, k1 = k0 + 8;
                        const unsigned char* row = vt + (d >> 3) * 1024 + (d & 7) * 16;
                        mma_k16(o[n], pa[kk], *(const unsigned*)(row + (k0 >> 3) * 128 + (k0 & 7) * 2),
                                *(const unsigned*)(row + (k1 >> 3) * 128 + (k1 & 7) * 2));
                    }
#endif
            }
#else
            // O += P V: B fragments of 16 keys x 16 dims per ldmatrix.x4.trans.
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    const int m = lane >> 3;
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_t(b0, b1, b2, b3,
                              Vh + (wgi * BN + kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                    mma_k16(o[2 * np], pa[kk], b0, b1);
                    mma_k16(o[2 * np + 1], pa[kk], b2, b3);
                }
#endif
    };

#if PIPE
    // QK^T of block j + 1 runs on the tensor cores while this warpgroup does block j's softmax and
    // P V: two score buffers, block j + 1 must have landed one iteration earlier (STAGES >= 4).
    static_assert(KVP == 1 && STAGES >= 4 && !REGLD && !PVWG, "PIPE: one head, 4+ stages, cp.async, mma.sync P V");
    float sb0[32], sb1[32];
    auto step = [&](float* sc, float* sn, int j) {
        cp_wait<STAGES - 3>();   // block j + 1's copies have landed
        fence_async_smem();
        __syncthreads();          // ... for every thread; block j - 1's stage and Vh are free
        issue_pages(j + 2 * STAGES - 1);
        issue_kv(j + STAGES - 1);
        cp_commit();
        qk_wait(sc);                      // block j's scores (issued last step)
        if (j + 1 < nb) qk_issue(sn, j + 1);
        convert_v(j);
        __syncthreads();   // Vh complete
        consume(sc, j);
    };
    if (nb > 0) {
        cp_wait<STAGES - 2>();   // block 0 landed
        fence_async_smem();
        __syncthreads();
        qk_issue(sb0, 0);
    }
#pragma unroll 1
    for (int j = 0; j < nb; j += 2) {
        step(sb0, sb1, j);
        if (j + 1 < nb) step(sb1, sb0, j + 1);
    }
#else
    for (int j = 0; j < nb; ++j) {
        cp_wait<STAGES - 2>();   // block j's copies have landed (and the pages STAGES - 1 ahead)
        fence_async_smem();      // ... visible to wgmma
        __syncthreads();          // ... for every thread; nobody still reads the stage refilled next, or Vh
        issue_pages(j + 2 * STAGES - 1);
#if REGLD
        uint4 rkv[2 * KVI];
        load_kv(j + STAGES - 1, rkv);
#else
        issue_kv(j + STAGES - 1);
#endif
        cp_commit();
        float s[32];   // S = Q K^T
        qk_issue(s, j);
        convert_v(j);   // meanwhile
        qk_wait(s);
        if (PVWG) fence_async_smem();   // V^T is read by wgmma (async proxy)
        __syncthreads();   // Vh complete
        consume(s, j);
#if REGLD
        store_kv(j + STAGES - 1, rkv);
#endif
    }
#endif
    cp_wait<0>();

    // Partials: normalized o and base-2 lse per row, straight to ws (one warp per row).
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = wl * 16 + grp + e * 8;
        if (r >= L * G) continue;
        const long slot = ((long)(b * L + r / G) * HQ + (kv0 + wgi) * G + r % G) * num_splits + split;
        const float l = l_run[e], inv = l > 0.f ? v_scale / l : 0.f;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *(float2*)(ws_o + slot * D + n * 8 + 2 * tig) = make_float2(o[n][2 * e] * inv, o[n][2 * e + 1] * inv);
        if (tig == 0) ws_lse[slot] = l > 0.f ? m_run[e] + __log2f(l) : NEG_INF;
    }
}
'''

# Verify tiles of KV-head pairs, warp-specialized: experimentos/verify_ws.py, copied by
# sincroniza_cuda.py.
KERNEL_A_WS_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// -D: HQ HKV L STAGES PAGE WINDOW ROPE [CREGS]
#define D 64
#ifndef BNK
#define BNK 64                         // keys per block: 64, or 32 (KVP 1, PSMEM: smaller S and P, 5 CTAs/SM)
#endif
#define BN BNK
#define NS (BN / 2)                    // scores per thread (the m64nBN accumulator)
#define ROWS 64                        // query rows per KV head: one warpgroup m64
#ifndef KVP
#define KVP 2                          // KV heads per CTA: a consumer warpgroup each
#endif
#ifndef PP
#define PP 0                           // 1 (KVP 1): two consumer warpgroups take alternate blocks, merged at the end
#endif
#define NWG (KVP * (PP ? 2 : 1))       // consumer warpgroups
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
// Registers: 2 consumer warpgroups (KVP 2 or PP), 80 at launch x 384 threads (2 CTAs/SM) = 32 x 128 +
// 2 x 128 x 104; one, 64 x 256 (4 CTAs/SM) = 24 x 128 + 128 x 104 (VCONV: 32 x 128 + 128 x 96). Either way 4
// consumer warpgroups per SM.
#ifndef MINB
#define MINB (NWG == 2 ? 2 : BN == 32 && !PIPE ? 5 : BN == 128 ? 3 : 4)
#endif
#ifndef VCONV
#define VCONV 0                        // 1 (KVP 1): the producer warpgroup's other 3 warps convert V to FP16
#endif
#ifndef CREGS
#define CREGS (VCONV ? 96 : BN == 32 && !PIPE ? 72 : BN == 128 ? 136 : 104)   // consumer registers (setmaxnreg)
// (BN 32: 24 x 128 + 72 x 128 = 48 x 256, 5 CTAs/SM; BN 128: 24 x 128 + 136 x 128 = 80 x 256, 3 CTAs/SM)
#endif
#ifndef PREGS
#define PREGS (NWG == 2 || VCONV ? 32 : 24)   // producer registers
#endif
#ifndef PQ
#define PQ 4                           // blocks of page entries the producer loads ahead
#endif
// PSMEM 1 (PNOWAIT): the page entries go to a shared-memory ring by cp.async, PQS blocks ahead, and
// each block waits for its own with wait_group. The register queue above made the compiler move
// every loaded entry into its slot right away, so each load was waited for where it was issued.
#ifndef PSMEM
#define PSMEM 0
#endif
#define PQS 4                          // PSMEM: blocks of page entries in flight
#define PSL (PQS + 1)                  // PSMEM: ring slots
#ifndef PNOWAIT
#define PNOWAIT 0                      // 1: the producer never waits on its copies (cp.async.mbarrier.arrive.noinc)
#endif
#ifndef VBULK
#define VBULK 0                        // 1 (KVP 1, PNOWAIT): V rows by 64-byte bulk copy, tx on full[]
#endif
// TPH (diagnostic, full layers only: ROPE 0, so rope is free): thread 0 of each consumer
// warpgroup adds up clock64 per phase of the block loop and stores the sums into rope (as u64,
// [CTA][8]: wait full, QK^T issue + V, wgmma wait, empty + barrier, softmax, P V, blocks).
#ifndef TPH
#define TPH 0
#endif
// EXPH 1: P = exp2 in packed half precision (ex2.approx.f16x2: two per MUFU op, straight into the
// FP16 A fragments of P V); the block's row sums in half2 too. DEFERL 1: each thread keeps its own
// part of the row sums and the quad adds them once, after the loop (no shuffles per block).
#ifndef EXPH
#define EXPH 0
#endif
#ifndef DEFERL
#define DEFERL 0
#endif
#ifndef OVL
#define OVL 0                          // 1 (BN 64, VHBUF 3): FA3-style QK^T(j + 1) || P V(j) overlap
#endif
#ifndef PIPE
#define PIPE 0                         // 1 (BN 32): the next block's QK^T overlaps this block's softmax and P V
#endif
#ifndef LEAN
#define LEAN 0                         // 1: the score scale inside exp2 (FFMA), O rescaled only when a row max moves
#endif
#ifndef GDCQ
#define GDCQ 0                         // 1 (inside a layer, launched with PDL): griddepcontrol.wait before any
                                       // read of qkv or write of the cache (K/V copies of older blocks before it)
#endif
#ifndef CLW
#define CLW 0                          // 2: the two KV heads' CTAs in a cluster share the gather of 128-byte rows
#endif
#ifndef VTROT
#define VTROT 0                        // PVA 2: V^T stores without bank conflicts (lane-rotated dims)
#endif
#ifndef PVA
#define PVA 0                          // 1: P V with wgmma, waited one block later (overlaps the next QK^T and V)
                                       // 2: the same in E4M3 (P rounded to E4M3, V^T kept in FP8), as FA3 FP8
#endif
#ifndef VHBUF
#define VHBUF 2                        // FP16 V buffers per head (1: one more barrier; for 64 KB of shared memory)
#endif

#define OFF_Q 0
#define OFF_K (OFF_Q + NWG * QTILE)    // a Q tile per consumer warpgroup
#define OFF_V (OFF_K + STAGES * KVP * TILE)
#define OFF_VH (OFF_V + STAGES * KVP * TILE)
#define OFF_B (OFF_VH + NWG * VHBUF * BN * VP * 2)   // mbarriers: full[STAGES], empty[STAGES]
#define OFF_PG (OFF_B + (2 * STAGES + 4) * 16)      // 16 bytes per barrier (+ vready[2], vfree[2])
#define OFF_ST (OFF_PG + (PSMEM ? PSL * BN * 4 : 0))   // PSMEM: [PSL][BN] page entries
#define SMEM_BYTES (OFF_ST + (CLW ? 2 * TILE : 0))   // CLW: 2 staging buffers, half a K tile + half a V tile each

static_assert(L * G <= ROWS, "one warpgroup holds a head's rows");
static_assert(!PP || (KVP == 1 && STAGES % 2 == 0 && VHBUF == 2), "PP: one head, stages in pairs (one consumer each)");
static_assert(HKV % KVP == 0, "KV heads in groups of KVP");
static_assert(STAGES >= 2, "one block in flight at least");
#define CPR (KROW / 16)                // the producer's lanes per key row (16 bytes each)
static_assert(!TPH || !ROPE, "TPH stores into rope");
static_assert(BN == 64 || ((BN == 32 || BN == 128) && KVP == 1 && PSMEM && !PP && !PVA && !VCONV),
              "BNK 32 / 128: one head, PSMEM");
static_assert(!PSMEM || (PNOWAIT && !VBULK), "PSMEM: copies tracked by noinc arrivals, groups only for pages");
static_assert(!VBULK || (KVP == 1 && PNOWAIT), "VBULK: one head per CTA, arrivals by cp.async.mbarrier.arrive.noinc");
static_assert(!VCONV || (KVP == 1 && !PP && !PVA && VHBUF == 2), "VCONV: one head, one consumer, two FP16 V buffers");
static_assert(!PVA || VHBUF == 2, "PVA reads a V buffer until the next block");
static_assert(PVA != 2 || !EXPH, "PVA 2 rounds P from FP32");
static_assert(!CLW || (CLW == 2 && HKV % 2 == 0 && KVP == 1 && PSMEM && PNOWAIT && !PP && !VBULK && !VCONV && !OVL
                      && !PIPE && BN == 64),
              "CLW: a cluster per KV pair (heads 2 i, 2 i + 1), one head per CTA");
static_assert(!GDCQ || (PSMEM && PNOWAIT && !CLW && !VCONV && !VBULK), "GDCQ: the PSMEM producer");

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
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
// exp2 of two halves in one MUFU op (cuda_fp16's h2exp2 goes through float on sm90).
__device__ __forceinline__ __half2 ex2_h2(__half2 x) {
    unsigned r, v = *(unsigned*)&x;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(v));
    return *(__half2*)&r;
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned e4m3x2_to_h2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return *(unsigned*)&h;
}
__device__ __forceinline__ unsigned pack_e4m3x2(float lo, float hi) {   // 16 bits: lo in the low byte
    return (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
// 16-byte copy, zero-filled when !ok (src-size 0).
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr(dst)), "l"(src), "r"(ok ? 16 : 0));
#else
    *(uint4*)dst = ok ? *(const uint4*)src : make_uint4(0, 0, 0, 0);
#endif
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr(dst)), "l"(src));
#else
    *(int*)dst = *(const int*)src;
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
__device__ __forceinline__ void cp_wait_dyn(int n) {   // n in [0, 7]
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (k == n) {
            if (k == 0) cp_wait<0>(); else if (k == 1) cp_wait<1>(); else if (k == 2) cp_wait<2>();
            else if (k == 3) cp_wait<3>(); else if (k == 4) cp_wait<4>(); else if (k == 5) cp_wait<5>();
            else if (k == 6) cp_wait<6>(); else cp_wait<7>();
        }
}
__device__ __forceinline__ void fence_async_smem() {
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
// Named barrier for one warpgroup (ids 1 .. KVP; 0 is __syncthreads).
__device__ __forceinline__ void wg_bar(int id) { asm volatile("bar.sync %0, 128;\n" :: "r"(id) : "memory"); }

// mbarriers. sm90: the hardware ones (arrive count fixed at init, parity waits). Below sm90: a
// counter and a completed-phase count in shared memory (same semantics for these waits).
#if __CUDA_ARCH__ >= 900
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
// The phase also waits for n more bytes (completed by bulk copies).
__device__ __forceinline__ void mb_expect_tx(unsigned long long* b, unsigned n) {
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(n) : "memory");
}
// 64-byte bulk copy global -> shared, completing its bytes on mbarrier b.
__device__ __forceinline__ void bulk64(void* dst, const void* src, unsigned long long* b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 64, [%2];\n"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(smem_addr(b)) : "memory");
}
// CLW: the CTA's own buffer or barrier at p, as seen in CTA rank of the cluster.
__device__ __forceinline__ unsigned cl_map(const void* p, unsigned rank) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(smem_addr(p)), "r"(rank));
    return r;
}
// n-byte bulk copy of this CTA's buffer src into CTA rank's buffer dst (at dst's offset), completing its
// bytes on that CTA's barrier b; one bulk group per call.
__device__ __forceinline__ void fwd_to(void* dst, const void* src, unsigned n, unsigned long long* b, unsigned rank) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 "cp.async.bulk.commit_group;\n"
                 :: "r"(cl_map(dst, rank)), "r"(smem_addr(src)), "r"(n), "r"(cl_map(b, rank)) : "memory");
}
__device__ __forceinline__ void fwd_wait_read() {   // the sources of every forward so far were read
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void fwd_wait() {   // every forward so far landed
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void mb_arrive_to(unsigned long long* b, unsigned rank) {   // on CTA rank's barrier b
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(cl_map(b, rank)) : "memory");
}
__device__ __forceinline__ void mb_wait_cl(unsigned long long* b, int parity) {   // arrivals from the cluster
    asm volatile("{\n.reg .pred done;\nWAIT_%=:\n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], %1;\n"
                 "@!done bra WAIT_%=;\n}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cl_sync() {   // every thread of the cluster
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
#else
__device__ __forceinline__ void mb_init(unsigned long long* b, int count) {
    unsigned* w = (unsigned*)b;
    w[0] = 0; w[1] = 0; w[2] = count;
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    unsigned* w = (unsigned*)b;
    __threadfence_block();
    if (atomicAdd(&w[0], 1u) + 1 == w[2]) {
        w[0] = 0;
        __threadfence_block();
        atomicAdd(&w[1], 1u);
    }
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, int parity) {
    volatile unsigned* w = (volatile unsigned*)b;
    while ((w[1] & 1u) == (unsigned)parity) { }
    __threadfence_block();
}
__device__ __forceinline__ void mb_arrive_cp(unsigned long long* b) { mb_arrive(b); }   // copies are synchronous here
__device__ __forceinline__ void mb_expect_tx(unsigned long long*, unsigned) {}
__device__ __forceinline__ void bulk64(void* dst, const void* src, unsigned long long*) {
    for (int i = 0; i < 4; ++i) ((uint4*)dst)[i] = ((const uint4*)src)[i];
}
// CLW below sm90: a one-CTA "cluster" (the copies for the other parity come back to this CTA).
__device__ __forceinline__ void fwd_to(void* dst, const void* src, unsigned n, unsigned long long*, unsigned) {
    for (unsigned i = 0; i < n / 16; ++i) ((uint4*)dst)[i] = ((const uint4*)src)[i];
}
__device__ __forceinline__ void fwd_wait_read() {}
__device__ __forceinline__ void fwd_wait() {}
__device__ __forceinline__ void mb_arrive_to(unsigned long long* b, unsigned) { mb_arrive(b); }
__device__ __forceinline__ void mb_wait_cl(unsigned long long* b, int parity) { mb_wait(b, parity); }
__device__ __forceinline__ void cl_sync() { __syncthreads(); }
#endif
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned long long wg_desc(const void* p) {   // no swizzle, LBO 128, SBO 512
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | (8ull << 16) | (32ull << 32);
}
__device__ __forceinline__ unsigned long long wg_desc_v(const void* p) {   // V^T tile: LBO 128, SBO 1024
    return (unsigned long long)((smem_addr(p) & 0x3FFFF) >> 4) | (8ull << 16) | (64ull << 32);
}
// O (64 x 64, FP32) += P (64 x 16 keys, FP16, A from registers as mma.sync's m16n8k16 A) V (16 keys x 64).
__device__ __forceinline__ void wg_pv(float* d, const unsigned* a, unsigned long long db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15,"
                 "%16, %17, %18, %19, %20, %21, %22, %23,"
                 "%24, %25, %26, %27, %28, %29, %30, %31"
                 "}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                   "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                   "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(1));
}
// O (64 x 64, FP32) += P (64 x 32 positions, E4M3, A from registers) V^T (64 dims x 32 positions, E4M3).
__device__ __forceinline__ void wg_pv8(float* d, const unsigned* a, unsigned long long db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15,"
                 "%16, %17, %18, %19, %20, %21, %22, %23,"
                 "%24, %25, %26, %27, %28, %29, %30, %31"
                 "}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                   "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                   "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(1));
}
#if BN == 64 || BN == 128   // BN 128: two n64 halves (n128 needs 90+ registers)
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
#else
__device__ __forceinline__ void wg_qk(float* d, unsigned long long da, unsigned long long db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {"
                 "%0, %1, %2, %3, %4, %5, %6, %7,"
                 "%8, %9, %10, %11, %12, %13, %14, %15"
                 "}, %16, %17, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                   "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                   "+f"(d[14]), "+f"(d[15])
                 : "l"(da), "l"(db), "r"(acc));
}
#endif
#endif
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
    // Barrier k at 16-byte steps (the emulation below sm90 keeps three words per barrier).
    unsigned long long* full = (unsigned long long*)(smem + OFF_B);       // full[2 k]
    unsigned long long* empty = full + 2 * STAGES;                        // empty[2 k]
    unsigned long long* vready = empty + 2 * STAGES;                      // VCONV: V buffer b converted
    unsigned long long* vfree = vready + 4;                               // VCONV: the consumer is done with b

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
    if (!GDCQ && first == 0 && last > 0) {   // GDCQ: by the producer, after the wait (below)
        for (int w = warp; w < (CLW ? 2 : KVP) * L; w += NT / 32) {   // CLW: both heads of the pair (same bytes)
            int j = w % L, kvh = (CLW ? kv0 & ~1 : kv0) + w / L, tok = b * L + j;
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
        mb_init(empty + 2 * tid, PP ? 4 : (NCONS / 32) * (CLW ? 2 : 1) + (VCONV ? 3 : 0));  // a lane per warp reading
                                                                                          // the stage (CLW: both CTAs')
    }
    if (CLW && tid < 2) mb_init(vready + 2 * tid, 32);   // CLW: staging buffer tid landed (the producer's lanes)
    if (VCONV && tid < 2) {
        mb_init(vready + 2 * tid, 3);   // the converting warps
        mb_init(vfree + 2 * tid, 4);    // the consumer's warps
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    __syncthreads();   // new keys and values in the cache, barriers initialized: the last CTA-wide barrier
    if (CLW) cl_sync();   // ... and the other CTA's barriers before any copy or arrival reaches them
    const int crank = blockIdx.y & 1;   // CLW: rank in the cluster (clusters of 2 along y, the KV pairs)

    if (warp >= NCONS / 32) {
        // ---------------- producer warpgroup: its first warp copies, the others leave ----------------
#if __CUDA_ARCH__ >= 900
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(PREGS));
#endif
#if VCONV
        if (warp != NCONS / 32) {   // warps 1-3 of the producer warpgroup: V of block j to FP16 buffer j % 2
            const int t3 = tid - NCONS - 32;
            for (int j = 0; j < nb; ++j) {
                const int s = j % STAGES, bf = j & 1;
                mb_wait(full + 2 * s, (j / STAGES) & 1);              // block j landed
                if (j >= 2) mb_wait(vfree + 2 * bf, ((j >> 1) - 1) & 1);   // P V of block j - 2 done with bf
                const unsigned char* V = Vr + s * TILE;
                __half* vh = Vh + bf * BN * VP;
                for (int idx = t3; idx < BN * (D / 16); idx += 96) {   // 16 bytes of key idx / 4
                    const int key = idx >> 2, c0 = (idx & 3) * 16;
                    const uint4 a = *(const uint4*)(V + key * KROW + c0);
                    const unsigned w[4] = {a.x, a.y, a.z, a.w};
                    unsigned hv[8];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                        hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                    }
                    uint4* dst = (uint4*)(vh + key * VP + c0);
                    dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                    dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
                }
                __syncwarp();
                if (lane == 0) { mb_arrive(vready + 2 * bf); mb_arrive(empty + 2 * s); }
            }
            return;
        }
#else
        if (warp != NCONS / 32) {
            if (CLW) cl_sync();
            return;
        }
#endif
        // Lane l loads the page entries of keys l and l + 32 of each block; the copies are spread so
        // that each warp instruction takes whole rows (below). The page entries go PQ blocks ahead in a
        // register queue (the loop is unrolled by PQ so the queue is indexed statically): loading
        // them one block ahead left the producer waiting a DRAM latency per block.
#if !PSMEM
        auto page_of = [&](int j, int key) {   // physical row of key key of block j, -1 if clipped
            int t = hi - (first + j + 1) * BN + key;
            return j < nb && t >= lo ? page_table[(long)b * pt_stride + t / PAGE] * PAGE + t % PAGE : -1;
        };
#endif
#if PSMEM
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
            if (j >= STAGES) {   // the consumers left j - STAGES (CLW: both CTAs' consumers)
                if (CLW) mb_wait_cl(empty + 2 * s, ((j / STAGES) - 1) & 1);
                else mb_wait(empty + 2 * s, ((j / STAGES) - 1) & 1);
            }
            cp_wait<PQS - 1>();   // block j's entries (and the copies of blocks before j - PQS + 1)
            __syncwarp();         // ... landed for every lane
#if GDCQ
            if (j == 0 && first == 0) {   // block 0 holds the step's own keys: wait for qkv, write them first
#if __CUDA_ARCH__ >= 900
                asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
                for (int w = 0; w < KVP * L; ++w) {
                    const int jt = w % L, kvh = kv0 + w / L, tok = b * L + jt;
                    const long pos = positions[tok], loc = cache_loc[tok];
                    const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
                    float2 kx = __bfloat1622float2(row[(HQ * D + kvh * D) / 2 + lane]);
                    const float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kvh * D) / 2 + lane]);
                    const float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                                           : make_float2(1.f, 0.f);
                    kx = norm_rope(kx, cs, eps);
                    *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kvh * D + 2 * lane) = __nv_cvt_float2_to_fp8x2(
                        make_float2(kx.x / k_scale, kx.y / k_scale), __NV_SATFINITE, __NV_E4M3);
                    *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kvh * D + 2 * lane) = __nv_cvt_float2_to_fp8x2(
                        make_float2(vx.x / v_scale, vx.y / v_scale), __NV_SATFINITE, __NV_E4M3);
                }
                __threadfence();
                __syncwarp();   // every lane's rows written before any lane copies them back
            }
#endif
#if CLW
            // Keys 32 crank .. + 31, whole 128-byte rows by cp.async (the two halves of a row from the same
            // instruction pair): this head's halves into stage s, the other head's into staging buffer j % 2,
            // laid out as the other CTA's stage. Block j - 1's staging is then forwarded to the other CTA
            // (one bulk copy each for K and V, completing on its full[]). Below sm90 the "other CTA" is this
            // one: the staging takes this head's halves of keys 32 (crank ^ 1) .. + 31.
#if __CUDA_ARCH__ >= 900
            const int kh = crank, sh = kv0 ^ 1;
            if (lane == 0) mb_expect_tx(full + 2 * s, 2 * (BN / 2) * D);   // the other CTA's half tiles of K and V
#else
            const int kh = crank ^ 1, sh = kv0;
#endif
            unsigned char* st = smem + OFF_ST + (j & 1) * TILE;   // [K: 32 keys x 64 B][V: 32 keys x 64 B]
            if (j >= 2) {
                if (lane == 0) fwd_wait_read();   // block j - 2's forward read this buffer
                __syncwarp();
            }
#pragma unroll
            for (int i = 0; i < BN / 2 * CPR / 32; ++i) {
                const int k = (32 / CPR) * i + lane / CPR, c = lane % CPR;
                {
                    const int key = 32 * crank + k, t = hi - (first + j + 1) * BN + key;
                    const bool ok = t >= lo;
                    const long at = (long)(ok ? Pg[(j % PSL) * BN + key] * PAGE + t % PAGE : 0) * ROW + c * 16;
                    cp16z(Ks + s * TILE + cm(key, c * 16), k_cache + at + kv0 * D, ok);
                    cp16z(Vr + s * TILE + key * KROW + c * 16, v_cache + at + kv0 * D, ok);
                }
                {
                    const int key = 32 * kh + k, t = hi - (first + j + 1) * BN + key;
                    const bool ok = t >= lo;
                    const long at = (long)(ok ? Pg[(j % PSL) * BN + key] * PAGE + t % PAGE : 0) * ROW + c * 16;
                    cp16z(st + cm(k, c * 16), k_cache + at + sh * D, ok);
                    cp16z(st + TILE / 2 + k * KROW + c * 16, v_cache + at + sh * D, ok);
                }
            }
            mb_arrive_cp(vready + 2 * (j & 1));   // staging buffer j % 2 landed
            if (j >= 1) {   // forward block j - 1
                const int jp = j - 1, sp = jp % STAGES;
                mb_wait(vready + 2 * (jp & 1), (jp >> 1) & 1);
                if (lane == 0) {
                    fence_async_smem();   // cp.async (generic proxy) writes, read by the bulk copy
                    const unsigned char* sb = smem + OFF_ST + (jp & 1) * TILE;
                    fwd_to(Ks + sp * TILE + 32 * kh * 64, sb, TILE / 2, full + 2 * sp, crank ^ 1);
                    fwd_to(Vr + sp * TILE + 32 * kh * KROW, sb + TILE / 2, TILE / 2, full + 2 * sp, crank ^ 1);
                }
                __syncwarp();
#if __CUDA_ARCH__ < 900
                mb_arrive(full + 2 * sp);   // below sm90 full[] has no byte count: arrive once forwarded
#endif
            }
#else
#pragma unroll
            for (int i = 0; i < BN * CPR / 32; ++i) {
                const int key = (32 / CPR) * i + lane / CPR, c = lane % CPR, t = hi - (first + j + 1) * BN + key;
                const bool ok = t >= lo;
                const long at = (long)(ok ? Pg[(j % PSL) * BN + key] * PAGE + t % PAGE : 0) * ROW + kv0 * D + c * 16;
                cp16z(Ks + (s * KVP + c / 4) * TILE + cm(key, (c % 4) * 16), k_cache + at, ok);
                cp16z(Vr + s * KVP * TILE + key * KROW + c * 16, v_cache + at, ok);
            }
#endif
#if !CLW || __CUDA_ARCH__ >= 900
            mb_arrive_cp(full + 2 * s);   // full[s] completes when every lane's copies so far have landed
#endif
            __syncwarp();                 // every lane has read slot j % PSL before it is refilled
            issue_pages(j + PQS);
        }
        cp_wait<0>();   // no thread leaves with cp.async in flight
#if CLW
        if (nb > 0) {   // forward the last block
            const int jp = nb - 1, sp = jp % STAGES;
#if __CUDA_ARCH__ >= 900
            const int kh = crank;
#else
            const int kh = crank ^ 1;
#endif
            mb_wait(vready + 2 * (jp & 1), (jp >> 1) & 1);
            if (lane == 0) {
                fence_async_smem();
                const unsigned char* sb = smem + OFF_ST + (jp & 1) * TILE;
                fwd_to(Ks + sp * TILE + 32 * kh * 64, sb, TILE / 2, full + 2 * sp, crank ^ 1);
                fwd_to(Vr + sp * TILE + 32 * kh * KROW, sb + TILE / 2, TILE / 2, full + 2 * sp, crank ^ 1);
                fwd_wait();
            }
            __syncwarp();
#if __CUDA_ARCH__ < 900
            mb_arrive(full + 2 * sp);
#endif
        }
        cl_sync();   // nor before the other CTA has received its forwards (its last full[] waits)
#endif
        return;
#else
        int pq[PQ][2];
#pragma unroll
        for (int u = 0; u < PQ; ++u) { pq[u][0] = page_of(u, lane); pq[u][1] = page_of(u, lane + 32); }
        int signaled = 0;   // blocks already announced on full[]
        for (int j0 = 0; j0 < nb; j0 += PQ) {
#pragma unroll
            for (int u = 0; u < PQ; ++u) {
                const int j = j0 + u;
                if (j >= nb) break;
                const int s = j % STAGES;
                if (j >= STAGES) mb_wait(empty + 2 * s, ((j / STAGES) - 1) & 1);   // both heads left j - STAGES
                if (VBULK) {   // keys lane, lane + 32: their V rows by bulk copy, after announcing the bytes
                    const int n = (pq[u][0] >= 0) + (pq[u][1] >= 0);
                    if (n) mb_expect_tx(full + 2 * s, 64u * n);
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                        if (pq[u][e] >= 0)
                            bulk64(Vr + s * KVP * TILE + (lane + 32 * e) * KROW,
                                   v_cache + (long)pq[u][e] * ROW + kv0 * D, full + 2 * s);
                }
                // Coalesced: copy i covers keys (32 / CPR) i .., whole rows of KROW bytes (CPR chunks);
                // key k's page entry comes from lane k % 32, slot k / 32.
#pragma unroll
                for (int i = 0; i < BN * CPR / 32; ++i) {
                    const int key = (32 / CPR) * i + lane / CPR, c = lane % CPR;
                    const int row = __shfl_sync(0xffffffffu, pq[u][key >> 5], key & 31);
                    const bool ok = row >= 0;
                    const long at = (long)(ok ? row : 0) * ROW + kv0 * D + c * 16;
                    cp16z(Ks + (s * KVP + c / 4) * TILE + cm(key, (c % 4) * 16), k_cache + at, ok);
                    if (!VBULK || !ok)   // VBULK: clipped rows still need their zeros
                        cp16z(Vr + s * KVP * TILE + key * KROW + c * 16, v_cache + at, ok);
                }
                pq[u][0] = page_of(j + PQ, lane);   // refill this slot for block j + PQ
                pq[u][1] = page_of(j + PQ, lane + 32);
                if (PNOWAIT) {   // full[s] completes when every lane's copies of block j have landed
                    mb_arrive_cp(full + 2 * s);
                    continue;
                }
                cp_commit();
                // Announce the oldest block once STAGES - 1 are in flight.
                if (j - signaled >= STAGES - 1) {
                    cp_wait<STAGES - 2>();
                    fence_async_smem();   // generic-proxy copies, read by wgmma (async proxy)
                    mb_arrive(full + 2 * (signaled % STAGES));
                    ++signaled;
                }
            }
        }
        if (!PNOWAIT) {
            cp_wait<0>();
            fence_async_smem();
            for (; signaled < nb; ++signaled) mb_arrive(full + 2 * (signaled % STAGES));
        } else {
            cp_wait<0>();   // no thread leaves with cp.async in flight (PTX leaves that undefined; Fable round 5)
            if (VBULK)      // nor with bulk copies: wait for the last blocks' phases (their bytes landed)
                for (int j = max(nb - STAGES, 0); j < nb; ++j) mb_wait(full + 2 * (j % STAGES), (j / STAGES) & 1);
        }
        return;
#endif
    }

    // ---------------- consumer warpgroup wg: KV head kv0 + h (PP: blocks of parity wg) ----------------
#if __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CREGS));
    if (GDCQ) asm volatile("griddepcontrol.wait;\n" ::: "memory");   // q comes from the qkv producer
#endif
    const int wg = warp >> 2, h = PP ? 0 : wg, wl = warp & 3, ct = tid & 127;
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
#if __CUDA_ARCH__ >= 900
    if (PVA) {   // pinned here: a zeroing sunk into the nb == 0 path, which joins the final wait, is C7515
        float* od = &o[0][0];
#pragma unroll
        for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
    }
#endif
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};
    unsigned pa[BN / 16][4];   // A fragments of P for k-tiles of 16 keys (PVA: read by the P V in flight)
    long long tph[7] = {0, 0, 0, 0, 0, 0, 0};

#if OVL
    // FA3's overlap with one score buffer: block j's softmax first, then block j + 1's QK^T is issued
    // and runs during V(j + 1)'s conversion and block j's P V (mma.sync on P in registers, not on
    // wgmma accumulators). One wgmma group in flight at a time and every wait a wait_group 0, so
    // ptxas keeps the wgmmas asynchronous. Three FP16 V buffers: a fast warp converts V(j + 1)
    // while a slow one may still run P V(j - 1). j + 1 == nb: a filler QK^T and V (results unused).
    static_assert(!OVL || (KVP == 1 && !PP && !PVA && !VCONV && !PIPE && VHBUF == 3 && !TPH), "OVL: one head, VHBUF 3");
    float sc[NS];
    {   // block 0
        const int s = 0;
        const unsigned char* K = Ks + h * TILE;
        const unsigned char* V = Vr;
        __half* vh = Vh + wg * VHBUF * BN * VP;
        if (nb > 0) {
            mb_wait(full, 0);
            if (PNOWAIT) fence_async_smem();
        }
#if __CUDA_ARCH__ >= 900
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                sc[i] = 0.f;
                asm volatile("" : "+f"(sc[i]) :: "memory");
            }
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
            wg_qk(sc, wg_desc(Q), wg_desc(K), 0);
            wg_qk(sc, wg_desc(Q + 256), wg_desc(K + 256), 1);
            if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
                wg_qk(sc + 32, wg_desc(Q), wg_desc(K + 4096), 0);
                wg_qk(sc + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
            }
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                float acc = 0.f;
                for (int c = 0; c < D; ++c) {
                    __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                    __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                    acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                }
                sc[i] = acc;
            }
#endif
            // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
            if (VHBUF == 1) wg_bar(1 + wg);   // the warpgroup is done reading vh (last block's P V)
#if VCONV
            // (converted by the producer warpgroup)
#elif PVA
            // V^T [dims][keys] in FP16 core matrices (K-major B for the P V wgmma): thread ct takes keys
            // 2 kp, 2 kp + 1 and dims q + 4 i (4-byte stores spread over the banks).
            {
                const int kp = ct >> 2, q = ct & 3;
                const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
                const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
                unsigned char* vt = (unsigned char*)vh;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 a = ra[c], bq = rb[c];
                    const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int d = (c * 4 + i) * 4 + q;
                        *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                            e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                    }
                }
            }
            fence_async_smem();   // V^T is read by wgmma (async proxy)
#else
#if BN == 32
            {   // 16 bytes of key ct / 4 per thread
                const int key = ct >> 2, c0 = (ct & 3) * 16;
                const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
                const unsigned w[4] = {a.x, a.y, a.z, a.w};
                unsigned hv[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                    hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                }
                uint4* dst = (uint4*)(vh + key * VP + c0);
                dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
            }
#else
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
#endif
#endif
    }
    for (int j = 0; j < nb; ++j) {
        const int s = j % STAGES;
        const int kb0 = hi - (first + j + 1) * BN;
        __half* vh = Vh + (wg * VHBUF + j % VHBUF) * BN * VP;
#if __CUDA_ARCH__ >= 900
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // block j's QK^T
#pragma unroll
        for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(sc[i]) :: "memory");
#endif
        __syncwarp();
        if (lane == 0) mb_arrive(empty + 2 * s);   // K(j) read by its wgmma, V(j) converted last block
        long long tq4 = TPH ? clock64() : 0;
        const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
        float mx[2] = {NEG_INF, NEG_INF};
        if (edge) {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                sc[i] = ok ? (LEAN ? sc[i] : sc[i] * qsc) : NEG_INF;
                mx[e] = fmaxf(mx[e], sc[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                if (!LEAN) sc[i] *= qsc;
                mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], sc[i]);
            }
        }
        // LEAN: sc holds raw scores; qsc > 0 keeps the max, and the scale goes into each exp2's FFMA.
        float alpha[2], safe[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
            float m_new = fmaxf(m_run[e], LEAN ? mx[e] * qsc : mx[e]);
            safe[e] = m_new == NEG_INF ? 0.f : m_new;
            alpha[e] = exp2f(m_run[e] - safe[e]);
            m_run[e] = m_new;
        }
        float rs[2] = {0.f, 0.f};
#if EXPH
        __half2 hs0 = __float2half2_rn(0.f), hs1 = __float2half2_rn(0.f);
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            const float sl = LEAN ? qsc : 1.f;
            const __half2 p01 = ex2_h2(__floats2half2_rn(fmaf(sc[nt * 4], sl, -safe[0]),
                                                         fmaf(sc[nt * 4 + 1], sl, -safe[0])));
            const __half2 p23 = ex2_h2(__floats2half2_rn(fmaf(sc[nt * 4 + 2], sl, -safe[1]),
                                                         fmaf(sc[nt * 4 + 3], sl, -safe[1])));
            hs0 = __hadd2(hs0, p01); hs1 = __hadd2(hs1, p23);
            pa[nt >> 1][(nt & 1) * 2] = *(const unsigned*)&p01;
            pa[nt >> 1][(nt & 1) * 2 + 1] = *(const unsigned*)&p23;
        }
        rs[0] = __low2float(hs0) + __high2float(hs0);
        rs[1] = __low2float(hs1) + __high2float(hs1);
#else
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            const float sl = LEAN ? qsc : 1.f;
            float p0 = exp2f(fmaf(sc[nt * 4], sl, -safe[0])), p1 = exp2f(fmaf(sc[nt * 4 + 1], sl, -safe[0]));
            float p2 = exp2f(fmaf(sc[nt * 4 + 2], sl, -safe[1])), p3 = exp2f(fmaf(sc[nt * 4 + 3], sl, -safe[1]));
            rs[0] += p0 + p1; rs[1] += p2 + p3;
            pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
#endif
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            if (!DEFERL) {
                rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
                rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            }
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
        // LEAN: once the row maxima settle (long contexts), no warp needs the rescale (warp-uniform).
        if (!LEAN || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
            }
        }
        {   // block j + 1: QK^T issued now, V converted now (filler past nb)
            const int jn = j + 1, sn = (jn < nb ? jn : j) % STAGES;
            const unsigned char* K = Ks + (sn * KVP + h) * TILE;
            const unsigned char* V = Vr + sn * KVP * TILE;
            __half* vh = Vh + (wg * VHBUF + jn % VHBUF) * BN * VP;
            if (jn < nb) {
                mb_wait(full + 2 * sn, (jn / STAGES) & 1);
                if (PNOWAIT) fence_async_smem();
            }
#if __CUDA_ARCH__ >= 900
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                sc[i] = 0.f;
                asm volatile("" : "+f"(sc[i]) :: "memory");
            }
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
            wg_qk(sc, wg_desc(Q), wg_desc(K), 0);
            wg_qk(sc, wg_desc(Q + 256), wg_desc(K + 256), 1);
            if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
                wg_qk(sc + 32, wg_desc(Q), wg_desc(K + 4096), 0);
                wg_qk(sc + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
            }
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                float acc = 0.f;
                for (int c = 0; c < D; ++c) {
                    __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                    __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                    acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                }
                sc[i] = acc;
            }
#endif
            // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
            if (VHBUF == 1) wg_bar(1 + wg);   // the warpgroup is done reading vh (last block's P V)
#if VCONV
            // (converted by the producer warpgroup)
#elif PVA
            // V^T [dims][keys] in FP16 core matrices (K-major B for the P V wgmma): thread ct takes keys
            // 2 kp, 2 kp + 1 and dims q + 4 i (4-byte stores spread over the banks).
            {
                const int kp = ct >> 2, q = ct & 3;
                const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
                const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
                unsigned char* vt = (unsigned char*)vh;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 a = ra[c], bq = rb[c];
                    const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int d = (c * 4 + i) * 4 + q;
                        *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                            e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                    }
                }
            }
            fence_async_smem();   // V^T is read by wgmma (async proxy)
#else
#if BN == 32
            {   // 16 bytes of key ct / 4 per thread
                const int key = ct >> 2, c0 = (ct & 3) * 16;
                const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
                const unsigned w[4] = {a.x, a.y, a.z, a.w};
                unsigned hv[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                    hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                }
                uint4* dst = (uint4*)(vh + key * VP + c0);
                dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
            }
#else
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
#endif
#endif
        }
        wg_bar(1 + wg);   // V(j) (converted last block) complete for the warpgroup
        long long tq5 = TPH ? clock64() : 0;
        // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
#if PVA
        {
            const unsigned char* vt = (const unsigned char*)vh;
#if __CUDA_ARCH__ >= 900
            float* od = &o[0][0];
#pragma unroll
            for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)   // 16 keys per wgmma: two core matrices on along K
                wg_pv(od, pa[kk], wg_desc_v(vt + kk * 256));
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // waited next block
#else
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int n = 0; n < 8; ++n) {   // B fragments from the same V^T layout
                    const int d = n * 8 + grp, k0 = kk * 16 + 2 * tig, k1 = k0 + 8;
                    const unsigned char* row = vt + (d >> 3) * 1024 + (d & 7) * 16;
                    mma_k16(o[n], pa[kk], *(const unsigned*)(row + (k0 >> 3) * 128 + (k0 & 7) * 2),
                            *(const unsigned*)(row + (k1 >> 3) * 128 + (k1 & 7) * 2));
                }
#endif
        }
#else
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
#endif
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // the filler QK^T
#endif
#elif PIPE
    // QK^T of block j + 1 runs on the tensor cores during block j's V conversion, softmax and P V
    // (FA3's intra-warpgroup overlap): two score buffers, the loop unrolled by two so they stay in
    // registers, and wgmma.wait_group 1 waits for block j's QK^T only.
    static_assert(!PIPE || (BN == 32 && KVP == 1 && !PP && !PVA && !VCONV && VHBUF == 2 && !TPH), "PIPE: BN 32");
    float scA[NS], scB[NS];
    // j == nb: a filler QK^T on the last block's stage (no wait, result unused), so every block
    // waits with wait_group 1 (a wait that depends on a branch makes ptxas serialize the wgmmas).
    if (nb > 0)
    {
        const int jq0 = 0;
        const int s = (jq0 < nb ? jq0 : nb - 1) % STAGES;
        const unsigned char* K = Ks + (s * KVP + h) * TILE;
        if (jq0 < nb) {
            mb_wait(full + 2 * s, (jq0 / STAGES) & 1);   // block jq0 landed
            if (PNOWAIT) fence_async_smem();
        }
#if __CUDA_ARCH__ >= 900
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                scA[i] = 0.f;
                asm volatile("" : "+f"(scA[i]) :: "memory");
            }
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
            wg_qk(scA, wg_desc(Q), wg_desc(K), 0);
            wg_qk(scA, wg_desc(Q + 256), wg_desc(K + 256), 1);
            if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
                wg_qk(scA + 32, wg_desc(Q), wg_desc(K + 4096), 0);
                wg_qk(scA + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
            }
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                float acc = 0.f;
                for (int c = 0; c < D; ++c) {
                    __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                    __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                    acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                }
                scA[i] = acc;
            }
#endif
    }
    for (int j0 = 0; j0 < nb; j0 += 2) {
        {
            const int jq = j0 + 1;
            const int s = (jq < nb ? jq : nb - 1) % STAGES;
            const unsigned char* K = Ks + (s * KVP + h) * TILE;
            if (jq < nb) {
                mb_wait(full + 2 * s, (jq / STAGES) & 1);   // block jq landed
                if (PNOWAIT) fence_async_smem();
            }
    #if __CUDA_ARCH__ >= 900
    #pragma unroll
                for (int i = 0; i < NS; ++i) {
                    scB[i] = 0.f;
                    asm volatile("" : "+f"(scB[i]) :: "memory");
                }
                asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
                wg_qk(scB, wg_desc(Q), wg_desc(K), 0);
                wg_qk(scB, wg_desc(Q + 256), wg_desc(K + 256), 1);
                if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
                    wg_qk(scB + 32, wg_desc(Q), wg_desc(K + 4096), 0);
                    wg_qk(scB + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
                }
                asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    #else
    #pragma unroll
                for (int i = 0; i < NS; ++i) {
                    int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                    float acc = 0.f;
                    for (int c = 0; c < D; ++c) {
                        __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                        __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                        acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                    }
                    scB[i] = acc;
                }
    #endif
        }

        {
            const int jr = j0;
            const int s = jr % STAGES;
            const unsigned char* V = Vr + s * KVP * TILE;
            __half* vh = Vh + (wg * VHBUF + jr % VHBUF) * BN * VP;
            const int kb0 = hi - (first + jr + 1) * BN;
                // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
                if (VHBUF == 1) wg_bar(1 + wg);   // the warpgroup is done reading vh (last block's P V)
    #if VCONV
                // (converted by the producer warpgroup)
    #elif PVA
                // V^T [dims][keys] in FP16 core matrices (K-major B for the P V wgmma): thread ct takes keys
                // 2 kp, 2 kp + 1 and dims q + 4 i (4-byte stores spread over the banks).
                {
                    const int kp = ct >> 2, q = ct & 3;
                    const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
                    const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
                    unsigned char* vt = (unsigned char*)vh;
    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint4 a = ra[c], bq = rb[c];
                        const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
    #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int d = (c * 4 + i) * 4 + q;
                            *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                                e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                        }
                    }
                }
                fence_async_smem();   // V^T is read by wgmma (async proxy)
    #else
    #if BN == 32
                {   // 16 bytes of key ct / 4 per thread
                    const int key = ct >> 2, c0 = (ct & 3) * 16;
                    const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
                    const unsigned w[4] = {a.x, a.y, a.z, a.w};
                    unsigned hv[8];
    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                        hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                    }
                    uint4* dst = (uint4*)(vh + key * VP + c0);
                    dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                    dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
                }
    #else
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
                    for (int i = 0; i < 4; ++i)
                        dst[i] = make_uint4(hv[4 * i], hv[4 * i + 1], hv[4 * i + 2], hv[4 * i + 3]);
                }
    #endif
    #endif
    #if __CUDA_ARCH__ >= 900
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");   // block jr's QK^T (jr + 1's in flight)
    #pragma unroll
            for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(scA[i]) :: "memory");
    #endif
            __syncwarp();
            if (lane == 0) mb_arrive(empty + 2 * s);   // this warp is done with stage s
            wg_bar(1 + wg);                            // vh complete for the warpgroup
                long long tq4 = TPH ? clock64() : 0;
                const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
                float mx[2] = {NEG_INF, NEG_INF};
                if (edge) {
    #pragma unroll
                    for (int i = 0; i < NS; ++i) {
                        int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                        bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                        scA[i] = ok ? (LEAN ? scA[i] : scA[i] * qsc) : NEG_INF;
                        mx[e] = fmaxf(mx[e], scA[i]);
                    }
                } else {
    #pragma unroll
                    for (int i = 0; i < NS; ++i) {
                        if (!LEAN) scA[i] *= qsc;
                        mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], scA[i]);
                    }
                }
                // LEAN: scA holds raw scores; qsc > 0 keeps the max, and the scale goes into each exp2's FFMA.
                float alpha[2], safe[2];
    #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
                    mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
                    float m_new = fmaxf(m_run[e], LEAN ? mx[e] * qsc : mx[e]);
                    safe[e] = m_new == NEG_INF ? 0.f : m_new;
                    alpha[e] = exp2f(m_run[e] - safe[e]);
                    m_run[e] = m_new;
                }
                float rs[2] = {0.f, 0.f};
    #if EXPH
                __half2 hs0 = __float2half2_rn(0.f), hs1 = __float2half2_rn(0.f);
    #pragma unroll
                for (int nt = 0; nt < BN / 8; ++nt) {
                    const float sl = LEAN ? qsc : 1.f;
                    const __half2 p01 = ex2_h2(__floats2half2_rn(fmaf(scA[nt * 4], sl, -safe[0]),
                                                                 fmaf(scA[nt * 4 + 1], sl, -safe[0])));
                    const __half2 p23 = ex2_h2(__floats2half2_rn(fmaf(scA[nt * 4 + 2], sl, -safe[1]),
                                                                 fmaf(scA[nt * 4 + 3], sl, -safe[1])));
                    hs0 = __hadd2(hs0, p01); hs1 = __hadd2(hs1, p23);
                    pa[nt >> 1][(nt & 1) * 2] = *(const unsigned*)&p01;
                    pa[nt >> 1][(nt & 1) * 2 + 1] = *(const unsigned*)&p23;
                }
                rs[0] = __low2float(hs0) + __high2float(hs0);
                rs[1] = __low2float(hs1) + __high2float(hs1);
    #else
    #pragma unroll
                for (int nt = 0; nt < BN / 8; ++nt) {
                    const float sl = LEAN ? qsc : 1.f;
                    float p0 = exp2f(fmaf(scA[nt * 4], sl, -safe[0])), p1 = exp2f(fmaf(scA[nt * 4 + 1], sl, -safe[0]));
                    float p2 = exp2f(fmaf(scA[nt * 4 + 2], sl, -safe[1]));
                    float p3 = exp2f(fmaf(scA[nt * 4 + 3], sl, -safe[1]));
                    rs[0] += p0 + p1; rs[1] += p2 + p3;
                    pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
                    pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
                }
    #endif
    #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (!DEFERL) {
                        rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
                        rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
                    }
                    l_run[e] = l_run[e] * alpha[e] + rs[e];
                }
                // LEAN: once the row maxima settle (long contexts), no warp needs the rescale (warp-uniform).
                if (!LEAN || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
    #pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
                    }
                }
                long long tq5 = TPH ? clock64() : 0;
                // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
    #if PVA
                {
                    const unsigned char* vt = (const unsigned char*)vh;
    #if __CUDA_ARCH__ >= 900
                    float* od = &o[0][0];
    #pragma unroll
                    for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)   // 16 keys per wgmma: two core matrices on along K
                        wg_pv(od, pa[kk], wg_desc_v(vt + kk * 256));
                    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // waited next block
    #else
    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)
    #pragma unroll
                        for (int n = 0; n < 8; ++n) {   // B fragments from the same V^T layout
                            const int d = n * 8 + grp, k0 = kk * 16 + 2 * tig, k1 = k0 + 8;
                            const unsigned char* row = vt + (d >> 3) * 1024 + (d & 7) * 16;
                            mma_k16(o[n], pa[kk], *(const unsigned*)(row + (k0 >> 3) * 128 + (k0 & 7) * 2),
                                    *(const unsigned*)(row + (k1 >> 3) * 128 + (k1 & 7) * 2));
                        }
    #endif
                }
    #else
    #pragma unroll
                for (int kk = 0; kk < BN / 16; ++kk)
    #pragma unroll
                    for (int np = 0; np < 4; ++np) {
                        const int m = lane >> 3;
                        unsigned b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3,
                                  vh + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                        mma_k16(o[2 * np], pa[kk], b0, b1);
                        mma_k16(o[2 * np + 1], pa[kk], b2, b3);
                    }
    #endif
        }

        // Block j0 + 1 even past nb (a filler: all scores -inf, no arrivals): every wgmma is issued
        // unconditionally, so ptxas can bound each pipeline stage.
        {
            const int jq = j0 + 2;
            const int s = (jq < nb ? jq : nb - 1) % STAGES;
            const unsigned char* K = Ks + (s * KVP + h) * TILE;
            if (jq < nb) {
                mb_wait(full + 2 * s, (jq / STAGES) & 1);   // block jq landed
                if (PNOWAIT) fence_async_smem();
            }
    #if __CUDA_ARCH__ >= 900
    #pragma unroll
                for (int i = 0; i < NS; ++i) {
                    scA[i] = 0.f;
                    asm volatile("" : "+f"(scA[i]) :: "memory");
                }
                asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
                wg_qk(scA, wg_desc(Q), wg_desc(K), 0);
                wg_qk(scA, wg_desc(Q + 256), wg_desc(K + 256), 1);
                if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
                    wg_qk(scA + 32, wg_desc(Q), wg_desc(K + 4096), 0);
                    wg_qk(scA + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
                }
                asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    #else
    #pragma unroll
                for (int i = 0; i < NS; ++i) {
                    int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
                    float acc = 0.f;
                    for (int c = 0; c < D; ++c) {
                        __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                        __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                        acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
                    }
                    scA[i] = acc;
                }
    #endif
        }

        {
            const int jr = j0 + 1;
            const int s = jr % STAGES;
            const unsigned char* V = Vr + s * KVP * TILE;
            __half* vh = Vh + (wg * VHBUF + jr % VHBUF) * BN * VP;
            const int kb0 = hi - (first + jr + 1) * BN;
                // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
                if (VHBUF == 1) wg_bar(1 + wg);   // the warpgroup is done reading vh (last block's P V)
    #if VCONV
                // (converted by the producer warpgroup)
    #elif PVA
                // V^T [dims][keys] in FP16 core matrices (K-major B for the P V wgmma): thread ct takes keys
                // 2 kp, 2 kp + 1 and dims q + 4 i (4-byte stores spread over the banks).
                {
                    const int kp = ct >> 2, q = ct & 3;
                    const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
                    const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
                    unsigned char* vt = (unsigned char*)vh;
    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint4 a = ra[c], bq = rb[c];
                        const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
    #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int d = (c * 4 + i) * 4 + q;
                            *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                                e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                        }
                    }
                }
                fence_async_smem();   // V^T is read by wgmma (async proxy)
    #else
    #if BN == 32
                {   // 16 bytes of key ct / 4 per thread
                    const int key = ct >> 2, c0 = (ct & 3) * 16;
                    const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
                    const unsigned w[4] = {a.x, a.y, a.z, a.w};
                    unsigned hv[8];
    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                        hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
                    }
                    uint4* dst = (uint4*)(vh + key * VP + c0);
                    dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                    dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
                }
    #else
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
                    for (int i = 0; i < 4; ++i)
                        dst[i] = make_uint4(hv[4 * i], hv[4 * i + 1], hv[4 * i + 2], hv[4 * i + 3]);
                }
    #endif
    #endif
    #if __CUDA_ARCH__ >= 900
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");   // block jr's QK^T
    #pragma unroll
            for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(scB[i]) :: "memory");
    #endif
            __syncwarp();
            if (lane == 0 && jr < nb) mb_arrive(empty + 2 * s);   // (jr == nb: a filler block)
            wg_bar(1 + wg);                            // vh complete for the warpgroup
                long long tq4 = TPH ? clock64() : 0;
                const bool edge = jr >= nb || kb0 < lo || kb0 + BN - 1 > seq
                    || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
                float mx[2] = {NEG_INF, NEG_INF};
                if (edge) {
    #pragma unroll
                    for (int i = 0; i < NS; ++i) {
                        int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                        bool ok = jr < nb && t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                        scB[i] = ok ? (LEAN ? scB[i] : scB[i] * qsc) : NEG_INF;
                        mx[e] = fmaxf(mx[e], scB[i]);
                    }
                } else {
    #pragma unroll
                    for (int i = 0; i < NS; ++i) {
                        if (!LEAN) scB[i] *= qsc;
                        mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], scB[i]);
                    }
                }
                // LEAN: scB holds raw scores; qsc > 0 keeps the max, and the scale goes into each exp2's FFMA.
                float alpha[2], safe[2];
    #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
                    mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
                    float m_new = fmaxf(m_run[e], LEAN ? mx[e] * qsc : mx[e]);
                    safe[e] = m_new == NEG_INF ? 0.f : m_new;
                    alpha[e] = exp2f(m_run[e] - safe[e]);
                    m_run[e] = m_new;
                }
                float rs[2] = {0.f, 0.f};
    #if EXPH
                __half2 hs0 = __float2half2_rn(0.f), hs1 = __float2half2_rn(0.f);
    #pragma unroll
                for (int nt = 0; nt < BN / 8; ++nt) {
                    const float sl = LEAN ? qsc : 1.f;
                    const __half2 p01 = ex2_h2(__floats2half2_rn(fmaf(scB[nt * 4], sl, -safe[0]),
                                                                 fmaf(scB[nt * 4 + 1], sl, -safe[0])));
                    const __half2 p23 = ex2_h2(__floats2half2_rn(fmaf(scB[nt * 4 + 2], sl, -safe[1]),
                                                                 fmaf(scB[nt * 4 + 3], sl, -safe[1])));
                    hs0 = __hadd2(hs0, p01); hs1 = __hadd2(hs1, p23);
                    pa[nt >> 1][(nt & 1) * 2] = *(const unsigned*)&p01;
                    pa[nt >> 1][(nt & 1) * 2 + 1] = *(const unsigned*)&p23;
                }
                rs[0] = __low2float(hs0) + __high2float(hs0);
                rs[1] = __low2float(hs1) + __high2float(hs1);
    #else
    #pragma unroll
                for (int nt = 0; nt < BN / 8; ++nt) {
                    const float sl = LEAN ? qsc : 1.f;
                    float p0 = exp2f(fmaf(scB[nt * 4], sl, -safe[0]));
                    float p1 = exp2f(fmaf(scB[nt * 4 + 1], sl, -safe[0]));
                    float p2 = exp2f(fmaf(scB[nt * 4 + 2], sl, -safe[1]));
                    float p3 = exp2f(fmaf(scB[nt * 4 + 3], sl, -safe[1]));
                    rs[0] += p0 + p1; rs[1] += p2 + p3;
                    pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
                    pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
                }
    #endif
    #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (!DEFERL) {
                        rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
                        rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
                    }
                    l_run[e] = l_run[e] * alpha[e] + rs[e];
                }
                // LEAN: once the row maxima settle (long contexts), no warp needs the rescale (warp-uniform).
                if (!LEAN || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
    #pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
                    }
                }
                long long tq5 = TPH ? clock64() : 0;
                // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
    #if PVA
                {
                    const unsigned char* vt = (const unsigned char*)vh;
    #if __CUDA_ARCH__ >= 900
                    float* od = &o[0][0];
    #pragma unroll
                    for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)   // 16 keys per wgmma: two core matrices on along K
                        wg_pv(od, pa[kk], wg_desc_v(vt + kk * 256));
                    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // waited next block
    #else
    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)
    #pragma unroll
                        for (int n = 0; n < 8; ++n) {   // B fragments from the same V^T layout
                            const int d = n * 8 + grp, k0 = kk * 16 + 2 * tig, k1 = k0 + 8;
                            const unsigned char* row = vt + (d >> 3) * 1024 + (d & 7) * 16;
                            mma_k16(o[n], pa[kk], *(const unsigned*)(row + (k0 >> 3) * 128 + (k0 & 7) * 2),
                                    *(const unsigned*)(row + (k1 >> 3) * 128 + (k1 & 7) * 2));
                        }
    #endif
                }
    #else
    #pragma unroll
                for (int kk = 0; kk < BN / 16; ++kk)
    #pragma unroll
                    for (int np = 0; np < 4; ++np) {
                        const int m = lane >> 3;
                        unsigned b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3,
                                  vh + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                        mma_k16(o[2 * np], pa[kk], b0, b1);
                        mma_k16(o[2 * np + 1], pa[kk], b2, b3);
                    }
    #endif
        }

    }
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // the filler QK^T (none pending: a no-op)
#endif
#else
    for (int j = PP ? wg : 0; j < nb; j += PP ? 2 : 1) {
        const int s = j % STAGES;
        const unsigned char* K = Ks + (s * KVP + h) * TILE;
        const unsigned char* V = Vr + s * KVP * TILE;
        __half* vh = Vh + (wg * VHBUF + (PP ? j / 2 : j) % VHBUF) * BN * VP;
        const int kb0 = hi - (first + j + 1) * BN;
        long long tq0 = TPH ? clock64() : 0;
        mb_wait(full + 2 * s, (j / STAGES) & 1);   // block j landed
        long long tq1 = TPH ? clock64() : 0;
        if (PNOWAIT) fence_async_smem();   // the producer's generic-proxy copies, before wgmma reads K
        // S = Q K^T: accumulator element nt 4 + c is row wl 16 + grp + 8 (c >> 1), key nt 8 + 2 tig + (c & 1).
        float sc[NS];
#if __CUDA_ARCH__ >= 900
        // PVA: block j - 1's P V is still in flight, and a non-wgmma definition of sc inside its
        // pipeline stage serializes every wgmma (C7515); the first QK^T wgmma has scale-d 0.
#pragma unroll
        for (int i = 0; i < NS; ++i) if (!PVA) {
            sc[i] = 0.f;
            asm volatile("" : "+f"(sc[i]) :: "memory");
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wg_qk(sc, wg_desc(Q), wg_desc(K), 0);
        wg_qk(sc, wg_desc(Q + 256), wg_desc(K + 256), 1);
        if (BN == 128) {   // keys 64 .. 127: core-matrix rows 8 .. 15 of the tile
            wg_qk(sc + 32, wg_desc(Q), wg_desc(K + 4096), 0);
            wg_qk(sc + 32, wg_desc(Q + 256), wg_desc(K + 4096 + 256), 1);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            int r = wl * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
            float acc = 0.f;
            for (int c = 0; c < D; ++c) {
                __half_raw qh = __nv_cvt_fp8_to_halfraw(Q[cm(r, c)], __NV_E4M3);
                __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
            }
            sc[i] = acc;
        }
#endif
        // Meanwhile this head's V to FP16 (thread ct: 32 bytes of key ct / 2).
        if (VHBUF == 1) wg_bar(1 + wg);   // the warpgroup is done reading vh (last block's P V)
#if VCONV
        // (converted by the producer warpgroup)
#elif PVA == 2
        // V^T [dims][positions] in E4M3 core matrices (K-major B of the e4m3 wgmma, wg_desc's layout).
        // Position 32 c + 16 a + 4 t + {0, 1, 2, 3} holds key 32 c + 16 a + 2 t + {0, 1, 8, 9}: the order in
        // which P's accumulator layout fills the A fragment. Thread ct: those 4 keys, dims 8 (ct >> 4) + i.
        {
            const int qd = ct & 15, dg = ct >> 4, k0 = (qd >> 2) * 16 + 2 * (qd & 3);
            const unsigned char* vr = V + k0 * KROW + h * D + dg * 8;
            const uint2 r0 = *(const uint2*)vr, r1 = *(const uint2*)(vr + KROW);
            const uint2 r2 = *(const uint2*)(vr + 8 * KROW), r3 = *(const uint2*)(vr + 9 * KROW);
            unsigned char* vt = (unsigned char*)vh + dg * 512 + (qd >> 2) * 128 + (qd & 3) * 4;
#if VTROT
            // Bank of a store: 4 (d & 7) + (qd & 3). Each lane starts at its own dim (qd >> 2) + 4 (dg & 1),
            // so the 32 lanes of a warp cover the 32 banks at every step (byte selectors at run time).
            const int rot = (qd >> 2) + 4 * (dg & 1);
#pragma unroll
            for (int st = 0; st < 8; ++st) {
                const int i = (st + rot) & 7;
                const bool hi = i & 4;
                const unsigned sel = 0x40 + 0x11 * (i & 3);   // byte i & 3 of a, then of b
                const unsigned p01 = __byte_perm(hi ? r0.y : r0.x, hi ? r1.y : r1.x, sel);
                const unsigned p23 = __byte_perm(hi ? r2.y : r2.x, hi ? r3.y : r3.x, sel);
                *(unsigned*)(vt + i * 16) = __byte_perm(p01, p23, 0x5410);
            }
#else
            const unsigned x0[2] = {r0.x, r0.y}, x1[2] = {r1.x, r1.y}, x2[2] = {r2.x, r2.y}, x3[2] = {r3.x, r3.y};
#pragma unroll
            for (int w = 0; w < 2; ++w) {   // dims 4 w .. 4 w + 3
                const unsigned l01 = __byte_perm(x0[w], x1[w], 0x5140), h01 = __byte_perm(x0[w], x1[w], 0x7362);
                const unsigned l23 = __byte_perm(x2[w], x3[w], 0x5140), h23 = __byte_perm(x2[w], x3[w], 0x7362);
                *(unsigned*)(vt + (4 * w) * 16) = __byte_perm(l01, l23, 0x5410);
                *(unsigned*)(vt + (4 * w + 1) * 16) = __byte_perm(l01, l23, 0x7632);
                *(unsigned*)(vt + (4 * w + 2) * 16) = __byte_perm(h01, h23, 0x5410);
                *(unsigned*)(vt + (4 * w + 3) * 16) = __byte_perm(h01, h23, 0x7632);
            }
#endif
        }
        fence_async_smem();   // V^T is read by wgmma (async proxy)
#elif PVA
        // V^T [dims][keys] in FP16 core matrices (K-major B for the P V wgmma): thread ct takes keys
        // 2 kp, 2 kp + 1 and dims q + 4 i (4-byte stores spread over the banks).
        {
            const int kp = ct >> 2, q = ct & 3;
            const uint4* ra = (const uint4*)(V + (2 * kp) * KROW + h * D);
            const uint4* rb = (const uint4*)(V + (2 * kp + 1) * KROW + h * D);
            unsigned char* vt = (unsigned char*)vh;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 a = ra[c], bq = rb[c];
                const unsigned wa[4] = {a.x, a.y, a.z, a.w}, wb[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = (c * 4 + i) * 4 + q;
                    *(unsigned*)(vt + (d >> 3) * 1024 + (kp >> 2) * 128 + (d & 7) * 16 + (kp & 3) * 4) =
                        e4m3x2_to_h2((unsigned short)__byte_perm(wa[i], wb[i], q | ((q + 4) << 4)));
                }
            }
        }
        fence_async_smem();   // V^T is read by wgmma (async proxy)
#else
#if BN == 32
        {   // 16 bytes of key ct / 4 per thread
            const int key = ct >> 2, c0 = (ct & 3) * 16;
            const uint4 a = *(const uint4*)(V + key * KROW + h * D + c0);
            const unsigned w[4] = {a.x, a.y, a.z, a.w};
            unsigned hv[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                hv[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                hv[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
            }
            uint4* dst = (uint4*)(vh + key * VP + c0);
            dst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
            dst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
        }
#else
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
#endif
#endif
        long long tq2 = TPH ? clock64() : 0;
#if __CUDA_ARCH__ >= 900
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");   // QK^T of j (and PVA: P V of j - 1)
#pragma unroll
        for (int i = 0; i < NS; ++i) asm volatile("" : "+f"(sc[i]) :: "memory");
        if (PVA) {
            float* od = &o[0][0];
#pragma unroll
            for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
        }
#endif
        long long tq3 = TPH ? clock64() : 0;
        __syncwarp();
        if (lane == 0) mb_arrive(empty + 2 * s);   // this warp is done with stage s (K by wgmma, V converted)
        if (CLW && lane == 0) mb_arrive_to(empty + 2 * s, crank ^ 1);   // ... which the other CTA's producer fills too
#if VCONV
        mb_wait(vready + 2 * (j & 1), (j >> 1) & 1);   // vh of block j converted
#else
        wg_bar(1 + wg);                        // vh complete for the warpgroup
#endif

        long long tq4 = TPH ? clock64() : 0;
        const bool edge = kb0 < lo || kb0 + BN - 1 > seq || (WINDOW > 0 && kb0 <= seq + L - 1 - WINDOW);
        float mx[2] = {NEG_INF, NEG_INF};
        if (edge) {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                int e = (i & 3) >> 1, t = kb0 + (i >> 2) * 8 + 2 * tig + (i & 1), p = ptok[e];
                bool ok = t >= lo && t <= p && (WINDOW == 0 || t > p - WINDOW);
                sc[i] = ok ? (LEAN ? sc[i] : sc[i] * qsc) : NEG_INF;
                mx[e] = fmaxf(mx[e], sc[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                if (!LEAN) sc[i] *= qsc;
                mx[(i & 3) >> 1] = fmaxf(mx[(i & 3) >> 1], sc[i]);
            }
        }
        // LEAN: sc holds raw scores; qsc > 0 keeps the max, and the scale goes into each exp2's FFMA.
        float alpha[2], safe[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 1));
            mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], 2));
            float m_new = fmaxf(m_run[e], LEAN ? mx[e] * qsc : mx[e]);
            safe[e] = m_new == NEG_INF ? 0.f : m_new;
            alpha[e] = exp2f(m_run[e] - safe[e]);
            m_run[e] = m_new;
        }
        float rs[2] = {0.f, 0.f};
#if EXPH
        __half2 hs0 = __float2half2_rn(0.f), hs1 = __float2half2_rn(0.f);
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            const float sl = LEAN ? qsc : 1.f;
            const __half2 p01 = ex2_h2(__floats2half2_rn(fmaf(sc[nt * 4], sl, -safe[0]),
                                                         fmaf(sc[nt * 4 + 1], sl, -safe[0])));
            const __half2 p23 = ex2_h2(__floats2half2_rn(fmaf(sc[nt * 4 + 2], sl, -safe[1]),
                                                         fmaf(sc[nt * 4 + 3], sl, -safe[1])));
            hs0 = __hadd2(hs0, p01); hs1 = __hadd2(hs1, p23);
            pa[nt >> 1][(nt & 1) * 2] = *(const unsigned*)&p01;
            pa[nt >> 1][(nt & 1) * 2 + 1] = *(const unsigned*)&p23;
        }
        rs[0] = __low2float(hs0) + __high2float(hs0);
        rs[1] = __low2float(hs1) + __high2float(hs1);
#else
#pragma unroll
        for (int nt = 0; nt < BN / 8; ++nt) {
            const float sl = LEAN ? qsc : 1.f;
            float p0 = exp2f(fmaf(sc[nt * 4], sl, -safe[0])), p1 = exp2f(fmaf(sc[nt * 4 + 1], sl, -safe[0]));
            float p2 = exp2f(fmaf(sc[nt * 4 + 2], sl, -safe[1])), p3 = exp2f(fmaf(sc[nt * 4 + 3], sl, -safe[1]));
            rs[0] += p0 + p1; rs[1] += p2 + p3;
#if PVA == 2
            // A fragment of the e4m3 wgmma k32: reg 0 / 2 row grp, positions 4 tig .. + 3 / 16 + 4 tig .. + 3;
            // reg 1 / 3 the same at row grp + 8. Tile nt fills half nt & 1 of reg (nt & 2) (+ 1).
            if ((nt & 1) == 0) {
                pa[nt >> 2][nt & 2] = pack_e4m3x2(p0, p1);
                pa[nt >> 2][(nt & 2) + 1] = pack_e4m3x2(p2, p3);
            } else {
                pa[nt >> 2][nt & 2] |= pack_e4m3x2(p0, p1) << 16;
                pa[nt >> 2][(nt & 2) + 1] |= pack_e4m3x2(p2, p3) << 16;
            }
#else
            pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
#endif
        }
#endif
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            if (!DEFERL) {
                rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 1);
                rs[e] += __shfl_xor_sync(0xffffffffu, rs[e], 2);
            }
            l_run[e] = l_run[e] * alpha[e] + rs[e];
        }
        // LEAN: once the row maxima settle (long contexts), no warp needs the rescale (warp-uniform).
        if (!LEAN || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
            }
        }
        long long tq5 = TPH ? clock64() : 0;
        // O += P V; vh is rewritten VHBUF blocks on, after a later wg_bar.
#if PVA
        {
            const unsigned char* vt = (const unsigned char*)vh;
#if __CUDA_ARCH__ >= 900
            float* od = &o[0][0];
#pragma unroll
            for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#if PVA == 2
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)   // 32 positions per wgmma: two core matrices on along K
                wg_pv8(od, pa[kk], wg_desc(vt + kk * 256));
#else
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)   // 16 keys per wgmma: two core matrices on along K
                wg_pv(od, pa[kk], wg_desc_v(vt + kk * 256));
#endif
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // waited next block
#elif PVA == 2
            // (local check) P from the quad's lanes: reg j of lane grp 4 + u holds row grp + 8 (j & 1),
            // positions 32 kk + 16 (j >> 1) + 4 u + {0, 1, 2, 3}.
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
#pragma unroll
                for (int jr = 0; jr < 4; ++jr)
                    for (int u = 0; u < 4; ++u) {
                        const unsigned w = __shfl_sync(0xffffffffu, pa[kk][jr], grp * 4 + u);
                        for (int b = 0; b < 4; ++b) {
                            const int pos = 32 * kk + 16 * (jr >> 1) + 4 * u + b;
                            __half_raw ph = __nv_cvt_fp8_to_halfraw((w >> (8 * b)) & 0xff, __NV_E4M3);
                            const float pv = __half2float(*(__half*)&ph);
#pragma unroll
                            for (int n = 0; n < 8; ++n)
#pragma unroll
                                for (int c = 0; c < 2; ++c) {
                                    const int d = n * 8 + 2 * tig + c;
                                    __half_raw vh8 = __nv_cvt_fp8_to_halfraw(
                                        vt[(d >> 3) * 512 + (pos >> 4) * 128 + (d & 7) * 16 + (pos & 15)], __NV_E4M3);
                                    o[n][(jr & 1) * 2 + c] += pv * __half2float(*(__half*)&vh8);
                                }
                        }
                    }
#else
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int n = 0; n < 8; ++n) {   // B fragments from the same V^T layout
                    const int d = n * 8 + grp, k0 = kk * 16 + 2 * tig, k1 = k0 + 8;
                    const unsigned char* row = vt + (d >> 3) * 1024 + (d & 7) * 16;
                    mma_k16(o[n], pa[kk], *(const unsigned*)(row + (k0 >> 3) * 128 + (k0 & 7) * 2),
                            *(const unsigned*)(row + (k1 >> 3) * 128 + (k1 & 7) * 2));
                }
#endif
        }
#else
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
#endif
#if VCONV
        __syncwarp();
        if (lane == 0) mb_arrive(vfree + 2 * (j & 1));   // this warp's P V is done with vh
#endif
        if (TPH) {
            const long long tq6 = clock64();
            tph[0] += tq1 - tq0; tph[1] += tq2 - tq1; tph[2] += tq3 - tq2; tph[3] += tq4 - tq3;
            tph[4] += tq5 - tq4; tph[5] += tq6 - tq5; tph[6] += 1;
        }
    }
#endif
    if (TPH && ct == 0) {
        const unsigned cta = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * NWG + wg;
#pragma unroll
        for (int i = 0; i < 7; ++i) ((long long*)rope)[cta * 8 + i] = tph[i];
    }
#if __CUDA_ARCH__ >= 900
    if (PVA) {   // the last block's P V
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        float* od = &o[0][0];
#pragma unroll
        for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(od[i]) :: "memory");
    }
#endif

#if DEFERL
#pragma unroll
    for (int e = 0; e < 2; ++e) {   // the quad's parts of each row sum
        l_run[e] += __shfl_xor_sync(0xffffffffu, l_run[e], 1);
        l_run[e] += __shfl_xor_sync(0xffffffffu, l_run[e], 2);
    }
#endif
#if PP
    // Merge: warpgroup 1 leaves its fragments (same rows and layout as warpgroup 0's) in its own V
    // buffers, free now; warpgroup 0 combines them in base 2.
    {
        float* mg = (float*)(Vh + VHBUF * BN * VP) + ct * 36;
        if (wg == 1) {
            wg_bar(2);   // every warp of warpgroup 1 is done reading its V buffers (its last P V)
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) mg[n * 4 + c] = o[n][c];
            mg[32] = m_run[0]; mg[33] = m_run[1]; mg[34] = l_run[0]; mg[35] = l_run[1];
        }
        asm volatile("bar.sync 3, 256;\n" ::: "memory");
        if (wg == 1) return;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const float m1 = mg[32 + e], l1 = mg[34 + e], mm = fmaxf(m_run[e], m1);
            const float safe = mm == NEG_INF ? 0.f : mm, a0 = exp2f(m_run[e] - safe), a1 = exp2f(m1 - safe);
            l_run[e] = l_run[e] * a0 + l1 * a1;
            m_run[e] = mm;
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int c = 0; c < 2; ++c) o[n][2 * e + c] = o[n][2 * e + c] * a0 + mg[n * 4 + 2 * e + c] * a1;
        }
    }
#endif
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
    if (CLW) cl_sync();   // nobody leaves while the other CTA may still arrive on its barriers
}
'''


# CUDA_EXTRA switches of the warp-specialized kernel
_WS_SWITCHES = ("PNOWAIT", "PVA", "PP", "VHBUF", "LEAN", "VBULK", "VCONV", "PSMEM", "TPH", "EXPH", "DEFERL",
                "BNK", "PIPE", "OVL", "MINB", "CREGS", "VTROT", "CLW", "GDCQ")


def _cufa_consts(cfg, stages=None, wg=False, cl=0, minb=None, kvp=None, lastm=False):
    """Constants of the CUDA kernel A: full_fa2 (wg False or "cl", then cl = cluster size), verify_wg
    (wg "wg") or verify_ws (wg "ws"). stages / minb / kvp: the tuning's cuda_stages / cuda_minb /
    cuda_kvp (decode; kvp 1: one KV head per CTA, 8 of 16 rows real, half the O registers)."""
    hq, hkv, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    if wg == "ws":  # KVP (CUDA_EXTRA["WS_KVP"], default a pair) heads per CTA; None if they don't divide
        kvp = CUDA_EXTRA.get("WS_KVP", 2)
        return None if hkv % kvp else dict(HQ=hq, HKV=hkv, L=length, STAGES=stages or CUDA_WS_STAGES,
                                           PAGE=cfg["page_size"], WINDOW=cfg["window"],
                                           ROPE=int(bool(cfg["rope_theta"])), WS=1, KVP=kvp,
                                           **{n: v for n, v in CUDA_EXTRA.items() if n in _WS_SWITCHES})
    if wg == "wg":  # a warpgroup per KV head, KVP heads per CTA; its own switches from CUDA_EXTRA
        k = dict(HQ=hq, HKV=hkv, L=length, STAGES=stages or CUDA_WG_STAGES, PAGE=cfg["page_size"],
                 WINDOW=cfg["window"], ROPE=int(bool(cfg["rope_theta"])), WG=1,
                 **{n: v for n, v in CUDA_EXTRA.items() if n in ("DSWAP", "REGLD", "KVP", "PVWG", "PIPE")})
        if hkv % k.get("KVP", 1):
            k["KVP"] = 1
        return k
    g = hq // hkv
    decode = hkv % 2 == 0 and 2 * length * g <= 16
    warp_m = 1 if decode else -(-length * g // 16)
    # Verify: cap the registers at 128 for 128-thread CTAs (4 per SM; the V loads cost 7 more).
    k = dict(HQ=hq, HKV=hkv, L=length, KVP=(kvp or 2) if decode else 1, WARP_M=warp_m,
                WARP_K=4 if decode else 1, STAGES=stages or CUDA_STAGES, PAGE=cfg["page_size"], WINDOW=cfg["window"],
                ROPE=int(bool(cfg["rope_theta"])), MINB=(minb or CUDA_DECODE_MINB) if decode else max(1, 16 // warp_m),
                **({"CL": cl, "NEWSMEM": 1, "EARLYQ": 1} if wg == "cl" else {}),
                **{n: v for n, v in CUDA_EXTRA.items()
                   if n in ("KVORD", "CPCA", "GSWAP", "NEWSMEM", "EARLYQ", "KVEARLY", "EPIW", "GDCQ")},
                **({n: v for n, v in CUDA_EXTRA.items() if n == "TSTAMP"} if wg == "cl" else {}))
    if CUDA_EXTRA.get("GDCQ"):   # the wait needs the step's own rows from shared memory and K/V first
        k.update(EARLYQ=1, NEWSMEM=1, KVEARLY=1)
    if lastm and decode and not wg:   # the last CTA merges: no combine kernel; cl's prologue
        k.update(LASTM=1, EARLYQ=1, NEWSMEM=1)
    return k


def _cufa_smem(k):
    if k.get("WS"):  # per consumer warpgroup a Q tile and two FP16 V buffers; per head the K and V stages
        nwg, bn = k["KVP"] * (2 if k.get("PP") else 1), k.get("BNK", 64)
        return (nwg * (64 * 64 + k.get("VHBUF", 2) * bn * 72 * 2) + k["KVP"] * 2 * k["STAGES"] * bn * 64
                + (2 * k["STAGES"] + 4) * 16 + (5 * bn * 4 if k.get("PSMEM") else 0)
                + (2 * bn * 64 if k.get("CLW") else 0))   # CLW: two staging buffers
    if k.get("WG"):  # per KV head: Q, K and V stages (E4M3), V in FP16; the page ring
        return k.get("KVP", 1) * (64 * 64 + 2 * k["STAGES"] * 64 * 64 + 64 * 72 * 2) + 2 * k["STAGES"] * 64 * 4
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
    source, name = ((KERNEL_A_WS_CUDA_SOURCE, b"verify_ws") if k.get("WS") else
                    (KERNEL_A_WG_CUDA_SOURCE, b"verify_wg") if k.get("WG") else (KERNEL_A_CUDA_SOURCE, b"full_fa2"))
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
    if k.get("WS"):
        return 128 * k["KVP"] * (2 if k.get("PP") else 1) + 128
    return 128 * k.get("KVP", 1) if k.get("WG") else k["WARP_M"] * k["WARP_K"] * 32


def _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, eps, k, out):
    """Kernel A of the split-KV path: per-split partials into ws_o [tokens, Hq, splits, 64] and
    ws_lse [tokens, Hq, splits], as the Triton kernel A writes them; with k["CL"], the whole step
    into out (clusters of CL = splits CTAs; ws unused)."""
    from cuda.bindings import driver
    cfg = data["config"]
    major, minor = torch.cuda.get_device_capability(ws_o.device)
    arch = "sm_90a" if major == 9 else f"sm_{major}{minor}"
    func, smem = _cufa_compile(k, arch)
    lc = driver.CUlaunchConfig()
    lc.gridDimX, lc.gridDimY, lc.gridDimZ = splits, cfg["num_kv_heads"] // k.get("KVP", 1), cfg["batch"]
    if k.get("GSWAP"):
        lc.gridDimX, lc.gridDimY = lc.gridDimY, lc.gridDimX
    lc.blockDimX, lc.blockDimY, lc.blockDimZ = _cufa_threads(k), 1, 1
    lc.sharedMemBytes = smem
    lc.hStream = torch.cuda.current_stream(ws_o.device).cuda_stream
    attrs = []
    if k.get("CL", 0) > 1 or k.get("CLW"):   # CL: the splits along x; CLW (verify_ws): the KV pair along y
        a = driver.CUlaunchAttribute()
        a.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dims = (1, 2, 1) if k.get("CLW") else (k["CL"], 1, 1)
        a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = dims
        attrs.append(a)
    lc.attrs = attrs
    lc.numAttrs = len(attrs)
    vp = ctypes.c_void_p
    tensors = (data["qkv"], data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
               data["k_scale"], data["v_scale"], data["page_table"], rope_table, ws_o, ws_lse)
    values = tuple(t.data_ptr() for t in tensors) + (data["page_table"].stride(0), splits, q_mult, eps)
    types = (vp,) * len(tensors) + (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float)
    if not (k.get("WG") or k.get("WS")):   # full_fa2 also takes out (CL, LASTM) and the LASTM counters
        cnt = _lastm_counters(out.device) if k.get("LASTM") else out   # (unused without LASTM)
        values, types = values + (out.data_ptr(), cnt.data_ptr()), types + (vp, vp)
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
    """Whether this step's kernel A runs in CUDA: the tuning asks for it, Hopper, 64-wide heads, a
    kernel that compiled (tried on the first call of each config, outside graph capture; a failure
    keeps the Triton path for good). Returns its constants, or None for the Triton kernel."""
    cfg = data["config"]
    if not (tune["cuda"] and ON_GPU and data["qkv"].is_cuda):
        return None
    if tune.get("cuda_max_batch") is not None and cfg["batch"] > tune["cuda_max_batch"]:
        return None
    any_arch = os.environ.get("OKC_CUDA_ANY_ARCH") == "1"  # local logic tests on a non-Hopper GPU
    if (torch.cuda.get_device_capability(data["qkv"].device)[0] != 9 and not any_arch) or cfg["head_dim"] != 64:
        return None
    if cfg["num_q_heads"] % cfg["num_kv_heads"] or cfg["q_len"] * cfg["num_q_heads"] // cfg["num_kv_heads"] > 64:
        return None
    wg = tune["cuda"] if tune["cuda"] in ("wg", "ws", "cl") else False
    big = cfg["batch"] >= tune.get("cuda_big_batch", 0)   # cuda_stages / cuda_minb only from this batch on
    k = _cufa_consts(cfg, stages=tune.get("cuda_stages") if big else None, wg=wg, cl=tune.get("cl", 0),
                     minb=tune.get("cuda_minb") if big else None, kvp=tune.get("cuda_kvp") if big else None,
                     lastm=tune.get("lastm", False))
    if k is None:
        return None
    if wg == "cl" and tune.get("cl_fit"):   # the largest power-of-two cluster with every CTA in one wave
        ctas = _sm_count(data["qkv"].device) // (cfg["batch"] * (cfg["num_kv_heads"] // k["KVP"]))
        fit = 1 << max(ctas, 1).bit_length() - 1
        if fit < k["CL"]:
            k = _cufa_consts(cfg, stages=tune.get("cuda_stages") if big else None, wg=wg, cl=fit,
                             minb=tune.get("cuda_minb") if big else None, kvp=tune.get("cuda_kvp") if big else None)
    if big and wg == "ws":   # cuda_big_extra: verify_ws switches only from cuda_big_batch on (e.g. CLW)
        k.update(tune.get("cuda_big_extra", {}))
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


_ROPE = {}
_SMS = {}
_COUNTERS = {}


def _lastm_counters(device):
    """LASTM: a counter per (request, KV pair), zero between calls (the last CTA resets its own).
    Allocated on the first call (a warm-up, outside graph capture)."""
    if device not in _COUNTERS:
        _COUNTERS[device] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return _COUNTERS[device]


_TS_BUF = {}   # TSTAMP: (batch, q_len, splits) -> (stamps of the last call, scored shape)


def _ts_report(compact=False):
    """TSTAMP: each scored shape's phase stamps (ns from the first CTA's start; min/med/max over CTAs,
    or compact: the medians)."""
    torch.cuda.synchronize()
    for shape, (ts, scored) in _TS_BUF.items():
        if scored:
            t = ts.view(-1, 8).cpu()
            t = t - t[:, :1].min()
            if compact:
                print("TS", *shape, [int(t[:, i].median()) for i in range(7)], flush=True)
            else:
                print("TSTAMP", *shape, [(int(t[:, i].min()), int(t[:, i].median()), int(t[:, i].max()))
                                         for i in range(7)], flush=True)
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
    """Splits per (request, KV-head pair): `want`, but no split under min_chunk keys nor over
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
    # (64 rows per KV head) run one tl.dot per head, as packing would double their MMA work, and
    # one KV head per program unless the tuning asks for a pair (whole 128-byte cache rows per
    # program, at twice the registers).
    group_rows = length * hq // hkv
    tune = TUNING[("sliding" if window else "full", "decode" if group_rows <= 16 else "verify")]
    if batch <= tune.get("cuda_small_batch", 0):   # cuda_small: the tuning's overrides up to that batch
        tune = dict(tune, **tune["cuda_small"])
    kvp = 2 if hkv % 2 == 0 and (2 * group_rows <= 16 or tune["pair"]) else 1
    # The step's own tokens must fit in the newest key block (block 0), the only one with the
    # causal mask.
    assert length <= tune["block_n"], f"q_len {length} > key block"
    pairs = hkv // kvp
    span = window + length if window else page_table.shape[1] * page
    pack = kvp if PACK_DECODE and kvp * group_rows <= 16 else 1
    rows = max(16, triton.next_power_of_2(pack * group_rows))
    rope = bool(cfg["rope_theta"])
    rope_table = _rope_table(cfg["rope_theta"], d, page_table.shape[1] * page, device) if rope else data["k_scale"]
    q_mult = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    swap = tune["swap"] and rows == 16
    meta = dict(
        L=length, LP=triton.next_power_of_2(max(length, 2)), HQ=hq, HKV=hkv, D=d, PAGE=page, WINDOW=window,
        ROPE=rope, KVP=kvp, PACK=pack, M=rows, BLOCK_N=tune["block_n"], ONE_SPLIT=False,
        QK_FP8=QK_FP8 and (rows >= 64 or swap), SWAP=swap, PREFETCH=tune["prefetch"], EDGES=tune["edges"],
        EARLY_TRIGGER=EARLY_TRIGGER_SLIDING and window > 0,
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
        pairs = hkv // k.get("KVP", 1)
    cl = k.get("CL", 0) if cuda else 0
    if cl:   # one kernel: a cluster of cl CTAs (the splits) merges in place
        splits = cl
    elif window:
        splits = num_splits(span, triton.cdiv(tune["per_sm"] * sms, batch * pairs), tune["min_chunk"])
    elif cuda:
        splits = num_splits(span, tune["waves"] * sms * _cufa_ctas(k, device) // (batch * pairs), tune["min_chunk"])
    else:
        def probe_args():
            probe = torch.empty(1, dtype=torch.float32, device=device)
            return args(probe, probe, 2)

        ctas = _ctas_per_sm(_attn_kernel, probe_args, meta, device)
        splits = num_splits(span, tune["waves"] * sms * ctas // (batch * pairs), tune["min_chunk"])
    if cl:
        if k.get("TSTAMP"):   # diagnostic: phase stamps of the last call per scored shape, printed at exit
            shape = (cfg["batch"], cfg["q_len"], splits)
            if shape not in _TS_BUF:
                if torch.cuda.is_current_stream_capturing():   # no warm-up call outside the graph: no stamps
                    _cufa_launch(data, out, out, splits, rope_table, q_mult, RMS_EPS, k, out)
                    return out
                scored = int(data["seq_lens"].max()) > 10000
                _TS_BUF[shape] = (torch.zeros(splits * pairs * batch * 8, dtype=torch.int64, device=device), scored)
                if len(_TS_BUF) == 1:
                    atexit.register(_ts_report)
            ts = _TS_BUF[shape][0]
            _cufa_launch(data, ts, ts, splits, rope_table, q_mult, RMS_EPS, k, out)
            if not torch.cuda.is_current_stream_capturing():   # every call outside a graph (the profiler's last)
                _ts_report(compact=True)
            return out
        _cufa_launch(data, out, out, splits, rope_table, q_mult, RMS_EPS, k, out)
        return out
    if cuda and k.get("LASTM"):
        splits = min(splits, 64)   # the last CTA's weight table holds 64 splits
    if splits > 1 or cuda:  # the CUDA kernel always writes partials, even for one split
        ws_o = torch.empty((tokens, hq, splits, d), dtype=torch.float32, device=device)
        ws_lse = torch.empty((tokens, hq, splits), dtype=torch.float32, device=device)
    else:
        ws_o = ws_lse = out
    if cuda:
        if k.get("TPH") and not torch.cuda.is_current_stream_capturing():   # diagnostic only (see verify_ws.py)
            ts = torch.zeros(splits * pairs * batch * 2 * 8, dtype=torch.int64, device=device)
            _cufa_launch(data, ws_o, ws_lse, splits, ts, q_mult, RMS_EPS, k, out)
            torch.cuda.synchronize()
            t = ts.view(-1, 8)[:, :7].double().cpu()
            t = t[t[:, 6] > 0]
            per = (t[:, :6].sum(0) / t[:, 6].sum()).tolist()
            print("TPH", cfg["batch"], cfg["q_len"], "cycles per block (wait full, QK+V, wgmma wait, empty+bar, "
                  "softmax, PV):", [round(x) for x in per], "blocks", int(t[:, 6].sum()), flush=True)
        else:
            _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, RMS_EPS, k, out)
            if k.get("LASTM"):   # merged, normed and gated in place
                return out
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

TUNING[("full", "decode")]["cuda"] = True
TUNING[("full", "verify")]["cuda"] = "ws"
CUDA_EXTRA["WS_KVP"] = 1
CUDA_EXTRA["PNOWAIT"] = 1
CUDA_EXTRA["PSMEM"] = 1
CUDA_EXTRA["KVEARLY"] = 1
CUDA_EXTRA["EPIW"] = 1
CUDA_WS_STAGES = 4
COMBINE_SP = 128
COMBINE_WARPS = 4
TUNING[("sliding", "decode")].update(cuda="cl", cl=16, cuda_max_batch=2, cuda_kvp=1, cuda_big_batch=0)
TUNING[("sliding", "verify")]["cuda"] = "ws"
