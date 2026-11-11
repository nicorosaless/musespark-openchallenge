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

# Tuning by shape range, measured on the bench (H100) and confirmed with H200 checks; diario.md
# has every number. Layer "sliding" (a window) or "full"; tile "decode" (one 16-row tile per KV
# pair: L x Hq/Hkv x 2 <= 16) or "verify" (64+ rows per KV head).
#   cuda      kernel A in CUDA (end of file: explicit cp.async pipeline), else Triton; "wg" (verify
#             tiles of up to 64 rows): the wgmma version, QK^T in E4M3 from shared memory
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
#define SMEM_BYTES (OFF_P + PRING * BN * 4)

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
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [ROWS][QP] E4M3, dims permuted (below)
    unsigned char* Kr = smem + OFF_K;   // [STAGES][BN][RP] E4M3
    unsigned char* Vr = smem + OFF_V;
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = GSWAP ? blockIdx.y : blockIdx.x, kv0 = (GSWAP ? blockIdx.x : blockIdx.y) * KVP, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
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
#pragma unroll
    for (int i = 0; i < QR; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane])
                      : make_float2(1.f, 0.f);
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
    cp_wait<0>();
    __syncthreads();
    for (int j = 0; j < STAGES - 1; ++j) { issue_kv(j); cp_commit(); }

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
    // to the program's only head on verify: a compile-time index into o.
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
    for (int idx = tid; idx < ROWS * D; idx += NT) {
        int r = idx / D, d = idx % D, mr = r / 16, rl = r % 16;
        int h = r / (ROWS / KVP), rr = r % (ROWS / KVP);
        if (rr >= L * G) continue;
        float mm = NEG_INF;
        for (int k = 0; k < WARP_K; ++k) mm = fmaxf(mm, Wm[((k * WARP_M + mr) * 16 + rl) * 2]);
        float safe = mm == NEG_INF ? 0.f : mm, den = 0.f, acc = 0.f;
        for (int k = 0; k < WARP_K; ++k) {
            int w = k * WARP_M + mr;
            float wgt = exp2f(Wm[(w * 16 + rl) * 2] - safe);
            den += wgt * Wm[(w * 16 + rl) * 2 + 1];
            acc += wgt * Wo[(w * 16 + rl) * D + d];
        }
        int j = rr / G, g = rr % G, head = (kv0 + h) * G + g;
        long slot = ((long)(b * L + j) * HQ + head) * num_splits + split;
        ws_o[slot * D + d] = den > 0.f ? acc / den * v_scale : 0.f;
        if (d == 0) ws_lse[slot] = den > 0.f ? mm + __log2f(den) : NEG_INF;
    }
}
'''

# Verify tiles (one KV head, up to 64 rows) with wgmma: experimentos/verify_wg.py, copied by
# sincroniza_cuda.py.
KERNEL_A_WG_CUDA_SOURCE = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// -D: HQ HKV L STAGES PAGE WINDOW ROPE
#define D 64
#define BN 64                          // keys per block
#define ROWS 64                        // query rows per CTA: one warpgroup m64
#define NT 128
#define G (HQ / HKV)
#define QKV_STRIDE ((2 * HQ + 2 * HKV) * D)
#define ROW (HKV * D)
#define TILE (BN * D)                  // bytes of an E4M3 tile (64 x 64)
#define VP (D + 8)                     // FP16 V rows (halves), padded: ldmatrix without bank conflicts
#define PRING (2 * STAGES)
#define NEG_INF __int_as_float(0xff800000)

#define OFF_Q 0
#define OFF_K (OFF_Q + TILE)
#define OFF_V (OFF_K + STAGES * TILE)
#define OFF_VH (OFF_V + STAGES * TILE)
#define OFF_P (OFF_VH + BN * VP * 2)
#define SMEM_BYTES (OFF_P + PRING * BN * 4)

static_assert(L * G <= ROWS, "one warpgroup holds the rows");
#ifndef REGLD
#define REGLD 0   // 1 (experiment): K/V through registers (plain loads after the barrier, stores at the end)
#endif
#define KVI (BN * (D / 16) / NT)       // 16-byte K (and V) chunks per thread and block

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

extern "C" __global__ void __launch_bounds__(NT, 4) verify_wg(
    const __nv_bfloat16* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const int* __restrict__ seq_lens, __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache,
    const float* __restrict__ k_scale_ptr, const float* __restrict__ v_scale_ptr,
    const int* __restrict__ page_table, const float* __restrict__ rope,
    float* __restrict__ ws_o, float* __restrict__ ws_lse, int pt_stride, int num_splits, float q_mult, float eps)
{
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* Qs = smem + OFF_Q;   // [64 rows x 64 B] E4M3, core-matrix layout
    unsigned char* Ks = smem + OFF_K;   // [STAGES][64 keys x 64 B] E4M3, core-matrix layout
    unsigned char* Vr = smem + OFF_V;   // [STAGES][64 keys][64 B] E4M3, row-major
    __half* Vh = (__half*)(smem + OFF_VH);   // [64 keys][VP] FP16, the block being used
    int* Pg = (int*)(smem + OFF_P);     // [PRING][BN] page-table entries

    const int split = blockIdx.x, kv0 = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, tig = lane & 3;
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
        for (int j = warp; j < L; j += NT / 32) {
            int tok = b * L + j;
            long pos = positions[tok], loc = cache_loc[tok];
            const __nv_bfloat162* row = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE);
            float2 kx = __bfloat1622float2(row[(HQ * D + kv0 * D) / 2 + lane]);
            float2 vx = __bfloat1622float2(row[((HQ + HKV) * D + kv0 * D) / 2 + lane]);
            float2 cs = ROPE ? make_float2(rope[pos * D + lane], rope[pos * D + D / 2 + lane])
                             : make_float2(1.f, 0.f);
            kx = norm_rope(kx, cs, eps);
            *(__nv_fp8x2_storage_t*)(k_cache + loc * ROW + kv0 * D + 2 * lane) =
                __nv_cvt_float2_to_fp8x2(make_float2(kx.x / k_scale, kx.y / k_scale), __NV_SATFINITE, __NV_E4M3);
            *(__nv_fp8x2_storage_t*)(v_cache + loc * ROW + kv0 * D + 2 * lane) =
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
    auto issue_kv = [&](int j) {      // K (core-matrix layout) and V (row-major) of block j, stage j % STAGES
        if (j < nb) {
            for (int idx = tid; idx < BN * (D / 16); idx += NT) {
                int key = idx / (D / 16), c16 = (idx % (D / 16)) * 16, t = hi - (first + j + 1) * BN + key;
                unsigned char* dk = Ks + (j % STAGES) * TILE + cm(key, c16);
                unsigned char* dv = Vr + (j % STAGES) * TILE + key * D + c16;
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
                int idx = tid + i * NT, key = idx / (D / 16), c16 = (idx % (D / 16)) * 16;
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
                int idx = tid + i * NT, key = idx / (D / 16), c16 = (idx % (D / 16)) * 16;
                *(uint4*)(Ks + (j % STAGES) * TILE + cm(key, c16)) = r[2 * i];
                *(uint4*)(Vr + (j % STAGES) * TILE + key * D + c16) = r[2 * i + 1];
            }
        }
    };

    for (int j = 0; j < 2 * STAGES - 1; ++j) issue_pages(j);
    cp_commit();
    // Queries meanwhile: RMSNorm, RoPE, E4M3 at unit RMS (16 rows per warp; rows past L G are zero).
    float2 qx[ROWS / 4], qcs[ROWS / 4];
    long qpos[ROWS / 4];
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = warp + i * 4;
        qx[i] = make_float2(0.f, 0.f);
        qpos[i] = 0;
        if (r < L * G) {
            int tok = b * L + r / G;
            const __nv_bfloat162* src = (const __nv_bfloat162*)(qkv + (long)tok * QKV_STRIDE + (kv0 * G + r % G) * D);
            qx[i] = __bfloat1622float2(src[lane]);
            qpos[i] = positions[tok];
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i)
        qcs[i] = ROPE ? make_float2(rope[qpos[i] * D + lane], rope[qpos[i] * D + D / 2 + lane]) : make_float2(1.f, 0.f);
#pragma unroll
    for (int i = 0; i < ROWS / 4; ++i) {
        int r = warp + i * 4;
        float2 x = norm_rope(qx[i], qcs[i], eps);
        if (r >= L * G) x = make_float2(0.f, 0.f);
        *(__nv_fp8x2_storage_t*)(Qs + cm(r, 2 * lane)) = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
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
    for (int e = 0; e < 2; ++e) ptok[e] = seq + min((warp * 16 + grp + e * 8) / G, L - 1);
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    for (int j = 0; j < nb; ++j) {
        cp_wait<STAGES - 2>();   // block j's copies have landed (and the pages STAGES - 1 ahead)
        fence_async_smem();      // ... visible to wgmma
        __syncthreads();          // ... for every thread; nobody still reads the stage refilled next, or Vh
        issue_pages(j + 2 * STAGES - 1);
        uint4 rkv[2 * KVI];
        if (REGLD) load_kv(j + STAGES - 1, rkv); else issue_kv(j + STAGES - 1);
        cp_commit();

        const unsigned char* K = Ks + (j % STAGES) * TILE;
        const unsigned char* V = Vr + (j % STAGES) * TILE;
        const int kb0 = hi - (first + j + 1) * BN;     // position of key 0 of this block
        // S = Q K^T: accumulator element nt 4 + c is row warp 16 + grp + 8 (c >> 1), key nt 8 + 2 tig + (c & 1).
        float s[32];
#if __CUDA_ARCH__ >= 900
#pragma unroll
        for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(s[i]) :: "memory");
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wg_qk(s, wg_desc(Qs), wg_desc(K), 0);
        wg_qk(s, wg_desc(Qs + 256), wg_desc(K + 256), 1);   // dims 32 .. 63: two core matrices on
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            int r = warp * 16 + grp + 8 * ((i & 3) >> 1), key = (i >> 2) * 8 + 2 * tig + (i & 1);
            float acc = 0.f;
            for (int c = 0; c < D; ++c) {
                __half_raw qh = __nv_cvt_fp8_to_halfraw(Qs[cm(r, c)], __NV_E4M3);
                __half_raw kh = __nv_cvt_fp8_to_halfraw(K[cm(key, c)], __NV_E4M3);
                acc += __half2float(*(__half*)&qh) * __half2float(*(__half*)&kh);
            }
            s[i] = acc;
        }
#endif
        // Meanwhile V to FP16, once per block: thread tid converts 32 bytes of key tid / 2.
        {
            const int key = tid >> 1, c0 = (tid & 1) * 32;
            const uint4 a = *(const uint4*)(V + key * D + c0), c = *(const uint4*)(V + key * D + c0 + 16);
            const unsigned w[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
            unsigned h[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                h[2 * i] = e4m3x2_to_h2((unsigned short)w[i]);
                h[2 * i + 1] = e4m3x2_to_h2((unsigned short)(w[i] >> 16));
            }
            uint4* dst = (uint4*)(Vh + key * VP + c0);
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
        }
#if __CUDA_ARCH__ >= 900
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(s[i]) :: "memory");
#endif
        __syncthreads();   // Vh complete

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
        // O += P V: B fragments of 16 keys x 16 dims per ldmatrix.x4.trans.
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int m = lane >> 3;
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vh + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8);
                mma_k16(o[2 * np], pa[kk], b0, b1);
                mma_k16(o[2 * np + 1], pa[kk], b2, b3);
            }
        if (REGLD) store_kv(j + STAGES - 1, rkv);
    }
    cp_wait<0>();

    // Partials: normalized o and base-2 lse per row, straight to ws (one warp per row).
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = warp * 16 + grp + e * 8;
        if (r >= L * G) continue;
        const long slot = ((long)(b * L + r / G) * HQ + kv0 * G + r % G) * num_splits + split;
        const float l = l_run[e], inv = l > 0.f ? v_scale / l : 0.f;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *(float2*)(ws_o + slot * D + n * 8 + 2 * tig) = make_float2(o[n][2 * e] * inv, o[n][2 * e + 1] * inv);
        if (tig == 0) ws_lse[slot] = l > 0.f ? m_run[e] + __log2f(l) : NEG_INF;
    }
}
'''


def _cufa_consts(cfg, stages=None, wg=False):
    hq, hkv, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    if wg:  # one warpgroup, one KV head; its own switches from CUDA_EXTRA
        return dict(HQ=hq, HKV=hkv, L=length, STAGES=stages or CUDA_STAGES, PAGE=cfg["page_size"],
                    WINDOW=cfg["window"], ROPE=int(bool(cfg["rope_theta"])), WG=1,
                    **{n: v for n, v in CUDA_EXTRA.items() if n in ("DSWAP", "REGLD")})
    g = hq // hkv
    decode = hkv % 2 == 0 and 2 * length * g <= 16
    warp_m = 1 if decode else -(-length * g // 16)
    # Verify: cap the registers at 128 for 128-thread CTAs (4 per SM; the V loads cost 7 more).
    return dict(HQ=hq, HKV=hkv, L=length, KVP=2 if decode else 1, WARP_M=warp_m,
                WARP_K=4 if decode else 1, STAGES=stages or CUDA_STAGES, PAGE=cfg["page_size"], WINDOW=cfg["window"],
                ROPE=int(bool(cfg["rope_theta"])), MINB=1 if decode else max(1, 16 // warp_m), **CUDA_EXTRA)


def _cufa_smem(k):
    if k.get("WG"):  # Q, K and V stages (E4M3), V in FP16, page ring
        return 64 * 64 + 2 * k["STAGES"] * 64 * 64 + 64 * 72 * 2 + 2 * k["STAGES"] * 64 * 4
    rows, dims = k["WARP_M"] * 16, k["KVP"] * 64
    total = rows * (dims + 16) + 2 * k["STAGES"] * 64 * (dims + 16) + 2 * k["STAGES"] * 64 * 4
    nw = k["WARP_M"] * k["WARP_K"]
    assert nw * 16 * 2 * 4 + nw * 16 * 64 * 4 <= 2 * k["STAGES"] * 64 * (dims + 16), k  # merge buffers fit
    return total


def _cufa_compile(k, arch):
    from cuda.bindings import driver, nvrtc
    key = (tuple(sorted(k.items())), arch)
    if key in _CUDA_FUNCS:
        return _CUDA_FUNCS[key]
    source, name = (KERNEL_A_WG_CUDA_SOURCE, b"verify_wg") if k.get("WG") else (KERNEL_A_CUDA_SOURCE, b"full_fa2")
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
    _CUDA_FUNCS[key] = (func, smem)
    return _CUDA_FUNCS[key]


def _cufa_threads(k):
    return 128 if k.get("WG") else k["WARP_M"] * k["WARP_K"] * 32


def _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, eps, k):
    """Kernel A of the split-KV path: per-split partials into ws_o [tokens, Hq, splits, 64] and
    ws_lse [tokens, Hq, splits], as the Triton kernel A writes them."""
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
    lc.attrs = []
    lc.numAttrs = 0
    vp = ctypes.c_void_p
    tensors = (data["qkv"], data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
               data["k_scale"], data["v_scale"], data["page_table"], rope_table, ws_o, ws_lse)
    values = tuple(t.data_ptr() for t in tensors) + (data["page_table"].stride(0), splits, q_mult, eps)
    types = (vp,) * len(tensors) + (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float)
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
    k = _cufa_consts(cfg, wg=tune["cuda"] == "wg")
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
        _cufa_launch(data, ws_o, ws_lse, splits, rope_table, q_mult, RMS_EPS, k)
    else:
        meta.update(ONE_SPLIT=splits == 1)
        _attn_kernel[(splits, pairs, batch)](*args(ws_o, ws_lse, splits), **meta)
    if splits > 1 or cuda:
        # One program per (token, query head). (HB > 1 heads per program is untested on GPU.)
        hb = 1
        _combine_kernel[(tokens, triton.cdiv(hq, hb))](
            ws_o, ws_lse, qkv, out, splits, RMS_EPS, HQ=hq, HKV=hkv, D=d, HB=hb,
            SP=min(32, triton.next_power_of_2(splits)), num_warps=2, **_PDL_LAUNCH)
    return out

TUNING[("full", "decode")]["cuda"] = True
