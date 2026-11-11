"""Compressed BF16 weights for m = 1 (gate 2 of the compression track; not a submission yet).

Format, per block of 64 rows x 128 columns of W (12 KB instead of 16):
  bytes 0-4095      sign | mantissa of columns 0-63      (64 x 64, row-major)
  bytes 4096-8191   sign | mantissa of columns 64-127
  bytes 8192-12287  exponent codes: low nibble column c, high nibble column c + 64
plus one byte per block, E (its largest exponent). A weight's exponent is E - code. Code 15
with a zero sign/mantissa byte decodes to d0 = 2^(E - 15 - 127); the weights whose exponent
is more than 14 below E, and zeros, are stored that way and fixed exactly afterwards from a
per-row list (column, true value, d0; stored as three [N, CAP] planes), at most CAP = 8 per row
(the judge's weights need at most 7).

The m = 1 kernel decodes in registers (inline PTX: prmt, per-byte subtract, lop3; 4 weights
per call), multiplies with FMA, splits K, and reduces through a PDL-launched finishing kernel
that also adds the corrections: out[r] = sum + sum_j (v_j - d0_j) * x[c_j].
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

C1_BN, C1_BK, C1_CAP = 64, 128, 8
C1_BLOCK_BYTES = 3 * 64 * 64

_DEC_LO = """{
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
_DEC_HI = _DEC_LO.replace("and.b32 c, $3, 0x0F0F0F0F;", "shr.b32 c, $3, 4;\nand.b32 c, c, 0x0F0F0F0F;")


@triton.jit
def _decode(sm, nib, e8, HIGH: tl.constexpr, ASM: tl.constexpr):
    """BF16 weights from sign|mantissa bytes, exponent-code nibbles and the block's E (u8)."""
    if ASM:
        if HIGH:
            return tl.inline_asm_elementwise(_DEC_HI_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=tl.bfloat16,
                                             is_pure=True, pack=4)
        else:
            return tl.inline_asm_elementwise(_DEC_LO_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=tl.bfloat16,
                                             is_pure=True, pack=4)
    else:  # the same arithmetic in plain Triton (the CPU interpreter cannot run PTX)
        s = sm.to(tl.int32)
        code = (nib.to(tl.int32) >> 4) & 15 if HIGH else nib.to(tl.int32) & 15
        ex = e8.to(tl.int32) - code
        return (((s & 0x80) << 8) | (ex << 7) | (s & 0x7F)).to(tl.int16).to(tl.bfloat16, bitcast=True)


_DEC_LO_C = tl.constexpr(_DEC_LO)
_DEC_HI_C = tl.constexpr(_DEC_HI)


@triton.jit(do_not_specialize=["SPLIT_K"])
def _gemv_c1(x_ptr, wc_ptr, e_ptr, acc_ptr, N, K, SPLIT_K, ASM: tl.constexpr, GDC: tl.constexpr,
             UNROLL: tl.constexpr = 1, DECODE: tl.constexpr = True, PREFETCH: tl.constexpr = False,
             PIPE: tl.constexpr = 1, corr_ptr=None, PSTRIDE=0, CORR: tl.constexpr = False,
             CAP: tl.constexpr = 8):
    """acc[64 rows of tile pid_n] += W_tile[:, its K share] @ x, x one token, W compressed.
    CORR: the pid_k = 0 CTAs also add the tile's corrections (see _finish_c1); their table loads
    go out before the loop, so the finishing kernel only converts.
    UNROLL blocks per loop step (more loads in flight); DECODE=False is a timing probe only
    (multiplies the raw bytes: wrong output). PIPE > 1: Triton pipelines the block loads that
    many stages deep through shared memory (cp.async); it does not by itself, as they feed no
    tl.dot. E is loaded as a scalar and splat: a per-element load of it made Triton pick a
    one-element-per-thread layout and shuffle all three planes through shared memory."""
    if GDC:
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
    if CORR:
        j = tl.arange(0, CAP)
        c_off = (pid_n * 64 + offs_r)[:, None] * CAP + j[None, :]
        first = (c_off >= 0) & (pid_k == 0)
        c_col = tl.load(corr_ptr + c_off, mask=first, other=0)
        c_v = tl.load(corr_ptr + PSTRIDE + c_off, mask=first, other=0)
        c_d0 = tl.load(corr_ptr + 2 * PSTRIDE + c_off, mask=first, other=0)
    if PREFETCH:
        # Software pipelining by hand: the next block's loads are issued before this block is
        # decoded and multiplied (Triton does not pipeline loads that feed no tl.dot).
        base = wc_ptr + blk.to(tl.int64) * 12288
        n_lo = tl.load(base + p_off)
        n_hi = tl.load(base + 4096 + p_off)
        n_nib = tl.load(base + 8192 + p_off)
        n_e8 = tl.zeros_like(n_lo) + tl.load(e_ptr + blk)
        for kb in range(kb_start, kb_end):
            sm_lo = n_lo
            sm_hi = n_hi
            nib = n_nib
            e8 = n_e8
            more = (p_off >= 0) & (kb + 1 < kb_end)
            nbase = wc_ptr + (blk + 1).to(tl.int64) * 12288
            n_lo = tl.load(nbase + p_off, mask=more, other=0)
            n_hi = tl.load(nbase + 4096 + p_off, mask=more, other=0)
            n_nib = tl.load(nbase + 8192 + p_off, mask=more, other=0)
            n_e8 = tl.zeros_like(n_lo) + tl.load(e_ptr + blk + 1, mask=kb + 1 < kb_end, other=0)
            x_lo = tl.load(x_ptr + kb * 128 + offs_c).to(tl.float32)
            x_hi = tl.load(x_ptr + kb * 128 + 64 + offs_c).to(tl.float32)
            w_lo = _decode(sm_lo, nib, e8, False, ASM).to(tl.float32)
            w_hi = _decode(sm_hi, nib, e8, True, ASM).to(tl.float32)
            acc += w_lo * x_lo[None, :] + w_hi * x_hi[None, :]
            blk += 1
    else:
        for kb0 in tl.range(kb_start, kb_end, UNROLL, num_stages=PIPE):
            for u in tl.static_range(UNROLL):
                kb = kb0 + u
                base = wc_ptr + (blk + u).to(tl.int64) * 12288
                sm_lo = tl.load(base + p_off)
                sm_hi = tl.load(base + 4096 + p_off)
                nib = tl.load(base + 8192 + p_off)
                x_lo = tl.load(x_ptr + kb * 128 + offs_c).to(tl.float32)
                x_hi = tl.load(x_ptr + kb * 128 + 64 + offs_c).to(tl.float32)
                if DECODE:
                  e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk + u)
                  w_lo = _decode(sm_lo, nib, e8, False, ASM).to(tl.float32)
                  w_hi = _decode(sm_hi, nib, e8, True, ASM).to(tl.float32)
                else:
                  w_lo = sm_lo.to(tl.float32) + nib.to(tl.float32)
                  w_hi = sm_hi.to(tl.float32)
                acc += w_lo * x_lo[None, :] + w_hi * x_hi[None, :]
            blk += UNROLL
    row = tl.sum(acc, axis=1)
    if CORR:
        xv = tl.load(x_ptr + (c_col.to(tl.int32) & 0xFFFF), mask=first, other=0.0).to(tl.float32)
        v = c_v.to(tl.bfloat16, bitcast=True).to(tl.float32)
        d0 = c_d0.to(tl.bfloat16, bitcast=True).to(tl.float32)
        row += tl.sum(v * xv - d0 * xv, axis=1)
    if GDC:
        gdc_wait()  # the previous call's finishing kernel has zeroed the workspace
    tl.atomic_add(acc_ptr + pid_n * 64 + offs_r, row, sem="relaxed")


@triton.jit
def _finish_c1(acc_ptr, out_ptr, corr_ptr, PSTRIDE, x_ptr, N, CAP: tl.constexpr, BLOCK: tl.constexpr, GDC: tl.constexpr,
               HAS_CORR: tl.constexpr = True, EARLY: tl.constexpr = False):
    """out[r] = bf16(acc[r] + sum_j (v_j - d0_j) * x[c_j]); acc[r] = 0. corr is int16 [3, rows, CAP]
    (rows >= N: W[:N] of a larger compressed matrix), planes PSTRIDE apart: columns, true values
    (BF16 bits), d0s (BF16 bits); padding is all zero. HAS_CORR=False when the
    main kernel added them. EARLY: the corrections (they do not depend on the main kernel) are
    loaded and summed before griddepcontrol.wait, while the main kernel still runs."""
    if GDC and not EARLY:
        gdc_wait()
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < N
    fix = tl.zeros((BLOCK,), dtype=tl.float32)
    if HAS_CORR:
        j = tl.arange(0, CAP)
        c_off = rows[:, None] * CAP + j[None, :]
        cmask = rmask[:, None]
        col = tl.load(corr_ptr + c_off, mask=cmask, other=0).to(tl.int32) & 0xFFFF
        v = tl.load(corr_ptr + PSTRIDE + c_off, mask=cmask, other=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
        d0 = tl.load(corr_ptr + 2 * PSTRIDE + c_off, mask=cmask, other=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
        xv = tl.load(x_ptr + col, mask=cmask, other=0.0).to(tl.float32)
        fix = tl.sum(v * xv - d0 * xv, axis=1)
    if GDC and EARLY:
        gdc_wait()
    a = tl.load(acc_ptr + rows, mask=rmask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + rows, (a + fix).to(out_ptr.dtype.element_ty), mask=rmask)
    tl.store(acc_ptr + rows, tl.zeros_like(a), mask=rmask)


def compress_c1(w: torch.Tensor):
    """BF16 [N, K] -> (blocks uint8 [N/64 * K/128, 12288], E uint8 [N/64 * K/128],
    corrections int16 [3, N, CAP]: columns, true values, d0s), or None if the weights do not fit
    the format."""
    n, k = w.shape
    if n % C1_BN or k % C1_BK:
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
    # [nt, nkb, 64, 192] -> per block: sm_lo (64x64), sm_hi, nib, each row-major
    blocks = blocks.view(nt, nkb, 64, 3, 64).permute(0, 1, 3, 2, 4).reshape(nt * nkb, C1_BLOCK_BYTES)
    # per-row corrections, in natural coordinates
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
        d0_bits = ((e_blk - 15) << 7).to(torch.int32)
        corr[0, r, slot] = c.to(torch.int16)
        corr[1, r, slot] = w.view(torch.int16)[r, c]
        corr[2, r, slot] = d0_bits.to(torch.int16)
    return blocks.to(torch.uint8).contiguous(), e_max.reshape(-1).to(torch.uint8).contiguous(), corr.contiguous()


def decompress_c1(c, n: int, k: int) -> torch.Tensor:
    """Inverse of compress_c1(), decoding as the kernel does: the losslessness check."""
    blocks, e_max, corr = c
    nt, nkb = n // C1_BN, k // C1_BK
    b = blocks.view(nt, nkb, 3, 64, 64).to(torch.int32)
    sm = torch.cat([b[:, :, 0], b[:, :, 1]], dim=-1)
    code = torch.cat([b[:, :, 2] & 15, b[:, :, 2] >> 4], dim=-1)
    ex = e_max.view(nt, nkb, 1, 1).to(torch.int32) - code
    bits = ((sm & 0x80) << 8) | (ex << 7) | (sm & 0x7F)
    nat = bits.permute(0, 2, 1, 3).reshape(n, k)
    # corrections: the decoded value must be d0; replace it by the true value
    cr = corr.to(torch.int32) & 0xFFFF
    used = (cr[1] != 0) | (cr[2] != 0)
    rows = torch.arange(n, device=nat.device)[:, None].expand(n, C1_CAP)
    r, cc = rows[used], cr[0][used]
    assert torch.equal(nat[r, cc], cr[2][used]), "a correction does not sit on a d0"
    nat[r, cc] = cr[1][used]
    return nat.to(torch.int16).view(torch.bfloat16)


def c1_matvec(x, comp, n, k, split, state, asm=True, gdc=True, warps=4, stages=3, unroll=1, decode=True,
              prefetch=False, pipe=1, corr_in_main=False, early=False, finish_block=128):
    """x [1, K] @ W[:n].T with W compressed by compress_c1, via _gemv_c1 + _finish_c1.
    With unroll > 1, split must leave each CTA a multiple of unroll K blocks. corr_in_main: the
    corrections go in the main kernel; early: in the finishing kernel, before its wait."""
    blocks, e_max, corr = comp
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _gemv_c1[(n // C1_BN, split)](x, blocks, e_max, state["acc"], n, k, split, ASM=asm, GDC=gdc,
                                  UNROLL=unroll, DECODE=decode, PREFETCH=prefetch, PIPE=pipe, corr_ptr=corr, PSTRIDE=corr.stride(0),
                                  CORR=corr_in_main, CAP=C1_CAP, num_warps=warps, num_stages=stages)
    kw = {"launch_pdl": True} if gdc else {}
    _finish_c1[(triton.cdiv(n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, n, CAP=C1_CAP, BLOCK=finish_block,
                                                GDC=gdc, HAS_CORR=not corr_in_main, EARLY=early,
                                                num_warps=4, **kw)
    return out


@triton.jit(do_not_specialize=["TOTAL", "NUM_CTAS"])
def _gemv_c1_sk(x_ptr, wc_ptr, e_ptr, acc_ptr, K, TOTAL, NUM_CTAS, PIPE: tl.constexpr, ASM: tl.constexpr,
                GDC: tl.constexpr):
    """_gemv_c1 with a stream-K split: the blocks are tile-major, so CTA i streams the contiguous
    run [i * TOTAL / NUM_CTAS, (i + 1) * TOTAL / NUM_CTAS) of all TOTAL = tiles * K-blocks, and adds
    one partial row vector per tile it touches. Any shape gets the same work per CTA (the plain
    split leaves qkv_gate's 36 tiles at 288 CTAs, or 1.09 waves)."""
    if GDC:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    num_kb = K // 128
    u0 = pid * TOTAL // NUM_CTAS
    u1 = (pid + 1) * TOTAL // NUM_CTAS
    offs_r = tl.arange(0, 64)
    offs_c = tl.arange(0, 64)
    p_off = offs_r[:, None] * 64 + offs_c[None, :]
    if GDC:
        gdc_wait()  # a no-op unless this kernel is itself launched with PDL
    for t in range(u0 // num_kb, (u1 + num_kb - 1) // num_kb):
        acc = tl.zeros((64, 64), dtype=tl.float32)
        for blk in tl.range(tl.maximum(u0, t * num_kb), tl.minimum(u1, (t + 1) * num_kb), num_stages=PIPE):
            kb = blk - t * num_kb
            base = wc_ptr + tl.cast(blk, tl.int64) * 12288
            sm_lo = tl.load(base + p_off)
            sm_hi = tl.load(base + 4096 + p_off)
            nib = tl.load(base + 8192 + p_off)
            e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk)
            x_lo = tl.load(x_ptr + kb * 128 + offs_c).to(tl.float32)
            x_hi = tl.load(x_ptr + kb * 128 + 64 + offs_c).to(tl.float32)
            w_lo = _decode(sm_lo, nib, e8, False, ASM).to(tl.float32)
            w_hi = _decode(sm_hi, nib, e8, True, ASM).to(tl.float32)
            acc += w_lo * x_lo[None, :] + w_hi * x_hi[None, :]
        tl.atomic_add(acc_ptr + t * 64 + offs_r, tl.sum(acc, axis=1), sem="relaxed")


@triton.jit
def _gemv_c1_direct(x_ptr, wc_ptr, e_ptr, out_ptr, corr_ptr, PSTRIDE, K, CAP: tl.constexpr, PIPE: tl.constexpr,
                    ASM: tl.constexpr):
    """One kernel for m = 1 without a K split: CTA pid_n streams all K blocks of its 64-row tile,
    adds the tile's corrections and writes the BF16 rows. No workspace, no finishing kernel (for
    short K, as o's 1024, where each split CTA had two blocks and the call paid a second kernel)."""
    pid_n = tl.program_id(0)
    num_kb = K // 128
    offs_r = tl.arange(0, 64)
    offs_c = tl.arange(0, 64)
    p_off = offs_r[:, None] * 64 + offs_c[None, :]
    j = tl.arange(0, CAP)
    c_off = (pid_n * 64 + offs_r)[:, None] * CAP + j[None, :]
    c_col = tl.load(corr_ptr + c_off)
    c_v = tl.load(corr_ptr + PSTRIDE + c_off)
    c_d0 = tl.load(corr_ptr + 2 * PSTRIDE + c_off)
    acc = tl.zeros((64, 64), dtype=tl.float32)
    for kb in tl.range(0, num_kb, num_stages=PIPE):
        blk = pid_n * num_kb + kb
        base = wc_ptr + tl.cast(blk, tl.int64) * 12288
        sm_lo = tl.load(base + p_off)
        sm_hi = tl.load(base + 4096 + p_off)
        nib = tl.load(base + 8192 + p_off)
        e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk)
        x_lo = tl.load(x_ptr + kb * 128 + offs_c).to(tl.float32)
        x_hi = tl.load(x_ptr + kb * 128 + 64 + offs_c).to(tl.float32)
        w_lo = _decode(sm_lo, nib, e8, False, ASM).to(tl.float32)
        w_hi = _decode(sm_hi, nib, e8, True, ASM).to(tl.float32)
        acc += w_lo * x_lo[None, :] + w_hi * x_hi[None, :]
    row = tl.sum(acc, axis=1)
    xv = tl.load(x_ptr + (c_col.to(tl.int32) & 0xFFFF)).to(tl.float32)
    v = c_v.to(tl.bfloat16, bitcast=True).to(tl.float32)
    d0 = c_d0.to(tl.bfloat16, bitcast=True).to(tl.float32)
    row += tl.sum(v * xv - d0 * xv, axis=1)
    tl.store(out_ptr + pid_n * 64 + offs_r, row.to(out_ptr.dtype.element_ty))


def c1d_matvec(x, comp, n, k, asm=True, warps=8, pipe=1):
    """x [1, K] @ W[:n].T in one kernel (_gemv_c1_direct)."""
    blocks, e_max, corr = comp
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _gemv_c1_direct[(n // C1_BN,)](x, blocks, e_max, out, corr, corr.stride(0), k, CAP=C1_CAP, PIPE=pipe, ASM=asm,
                                   num_warps=warps, num_stages=3)
    return out


def c1sk_matvec(x, comp, n, k, state, ctas_per_sm=4, asm=True, gdc=True, warps=None, pipe=None, early=True,
                finish_block=128, maxnreg=None):
    """x [1, K] @ W[:n].T via _gemv_c1_sk + _finish_c1. warps / pipe default to the v10 rule on
    the blocks per CTA: >= 4, 4 warps and 3 stages; else 8 warps, no pipelining. maxnreg caps the
    registers (8 warps need 64 for 4 CTAs per SM; the stream-K loop takes 80 otherwise)."""
    blocks, e_max, corr = comp
    total = (n // C1_BN) * (k // C1_BK)
    ctas = min(total, ctas_per_sm * state["num_sms"])
    deep = total // ctas >= 4
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    reg = {"maxnreg": maxnreg} if maxnreg else {}
    _gemv_c1_sk[(ctas,)](x, blocks, e_max, state["acc"], k, total, ctas, PIPE=pipe or (3 if deep else 1), ASM=asm,
                         GDC=gdc, num_warps=warps or (4 if deep else 8), num_stages=3, **reg)
    kw = {"launch_pdl": True} if gdc else {}
    _finish_c1[(triton.cdiv(n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, n, CAP=C1_CAP,
                                                BLOCK=finish_block, GDC=gdc, HAS_CORR=True, EARLY=early,
                                                num_warps=4, **kw)
    return out


@triton.jit(do_not_specialize=["M", "SPLIT_K"])
def _gemm_c1(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: tl.constexpr,
             ASM: tl.constexpr, GDC: tl.constexpr, DOT_FP32: tl.constexpr = False, PIPE: tl.constexpr = 1):
    """acc[M, 64 rows of tile pid_n] += x[M, its K share] @ W_tile^T with the decoded BF16
    weights on the tensor cores (tl.dot; m padded to BLOCK_M), instead of FP32 FMA."""
    if GDC:
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
        w_lo = _decode(sm_lo, nib, e8, False, ASM)
        w_hi = _decode(sm_hi, nib, e8, True, ASM)
        if DOT_FP32:  # the CPU interpreter's tl.dot is wrong on BF16 operands
            acc = tl.dot(w_lo.to(tl.float32), xt_lo.to(tl.float32), acc)
            acc = tl.dot(w_hi.to(tl.float32), xt_hi.to(tl.float32), acc)
        else:
            acc = tl.dot(w_lo, xt_lo, acc)
            acc = tl.dot(w_hi, xt_hi, acc)
        blk += 1
    if GDC:
        gdc_wait()
    o_off = offs_m[None, :] * N + (pid_n * 64 + offs_r)[:, None]
    tl.atomic_add(acc_ptr + o_off, acc, mask=m_mask[None, :], sem="relaxed")


@triton.jit(do_not_specialize=["M", "TOTAL", "NUM_CTAS"])
def _gemm_c1_sk(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, TOTAL, NUM_CTAS, BLOCK_M: tl.constexpr,
                PIPE: tl.constexpr, ASM: tl.constexpr, GDC: tl.constexpr, DOT_FP32: tl.constexpr = False):
    """_gemm_c1 (tensor cores, M <= BLOCK_M tokens) with the stream-K split of _gemv_c1_sk."""
    if GDC:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    num_kb = K // 128
    u0 = pid * TOTAL // NUM_CTAS
    u1 = (pid + 1) * TOTAL // NUM_CTAS
    offs_r = tl.arange(0, 64)
    offs_c = tl.arange(0, 64)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    p_off = offs_r[:, None] * 64 + offs_c[None, :]
    x_off = offs_m[None, :] * stride_xm + offs_c[:, None]
    if GDC:
        gdc_wait()  # a no-op unless this kernel is itself launched with PDL
    for t in range(u0 // num_kb, (u1 + num_kb - 1) // num_kb):
        acc = tl.zeros((64, BLOCK_M), dtype=tl.float32)
        for blk in tl.range(tl.maximum(u0, t * num_kb), tl.minimum(u1, (t + 1) * num_kb), num_stages=PIPE):
            kb = blk - t * num_kb
            base = wc_ptr + tl.cast(blk, tl.int64) * 12288
            sm_lo = tl.load(base + p_off)
            sm_hi = tl.load(base + 4096 + p_off)
            nib = tl.load(base + 8192 + p_off)
            e8 = tl.zeros_like(sm_lo) + tl.load(e_ptr + blk)
            xt_lo = tl.load(x_ptr + kb * 128 + x_off, mask=m_mask[None, :], other=0.0)
            xt_hi = tl.load(x_ptr + kb * 128 + 64 + x_off, mask=m_mask[None, :], other=0.0)
            w_lo = _decode(sm_lo, nib, e8, False, ASM)
            w_hi = _decode(sm_hi, nib, e8, True, ASM)
            if DOT_FP32:
                acc = tl.dot(w_lo.to(tl.float32), xt_lo.to(tl.float32), acc)
                acc = tl.dot(w_hi.to(tl.float32), xt_hi.to(tl.float32), acc)
            else:
                acc = tl.dot(w_lo, xt_lo, acc)
                acc = tl.dot(w_hi, xt_hi, acc)
        o_off = offs_m[None, :] * N + (t * 64 + offs_r)[:, None]
        tl.atomic_add(acc_ptr + o_off, acc, mask=m_mask[None, :], sem="relaxed")


def c1sk_matmul(x, comp, n, k, state, ctas_per_sm=4, block_m=None, asm=True, gdc=True, warps=4, pipe=3,
                finish_block=512, dot_fp32=False, maxnreg=None):
    """x [M, K] @ W[:n].T (M <= 32) via _gemm_c1_sk + _finish_c1m (corrections before its wait)."""
    blocks, e_max, corr = comp
    m = x.shape[0]
    block_m = block_m or (16 if m <= 16 else 32)
    total = (n // C1_BN) * (k // C1_BK)
    ctas = min(total, ctas_per_sm * state["num_sms"])
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    reg = {"maxnreg": maxnreg} if maxnreg else {}
    _gemm_c1_sk[(ctas,)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), total, ctas, BLOCK_M=block_m,
                         PIPE=pipe, ASM=asm, GDC=gdc, DOT_FP32=dot_fp32, num_warps=warps, num_stages=3, **reg)
    kw = {"launch_pdl": True} if gdc else {}
    _finish_c1m[(triton.cdiv(m * n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, m, n, x.stride(0),
                                                    CAP=C1_CAP, BLOCK=finish_block, GDC=gdc, EARLY=True, num_warps=4,
                                                    **kw)
    return out


@triton.jit
def _finish_c1m(acc_ptr, out_ptr, corr_ptr, PSTRIDE, x_ptr, M, N, stride_xm, CAP: tl.constexpr, BLOCK: tl.constexpr,
                GDC: tl.constexpr, EARLY: tl.constexpr = False):
    """out[t, r] = bf16(acc[t, r] + sum_j (v_j - d0_j) * x[t, c_j]) over the M x N outputs. EARLY as in
    _finish_c1."""
    if GDC and not EARLY:
        gdc_wait()
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
    if GDC and EARLY:
        gdc_wait()
    a = tl.load(acc_ptr + idx, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + idx, (a + fix).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(acc_ptr + idx, tl.zeros_like(a), mask=mask)


def c1_matmul(x, comp, n, k, split, state, block_m=16, asm=True, gdc=True, warps=4, stages=3, dot_fp32=False,
              pipe=1, early=False, finish_block=128):
    """x [M, K] @ W[:n].T (M <= 32) with W compressed by compress_c1: tensor-core variant."""
    blocks, e_max, corr = comp
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _gemm_c1[(n // C1_BN, triton.cdiv(m, block_m), split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0),
                                                          split, BLOCK_M=block_m, ASM=asm, GDC=gdc, DOT_FP32=dot_fp32,
                                                          PIPE=pipe,
                                                          num_warps=warps, num_stages=stages)
    kw = {"launch_pdl": True} if gdc else {}
    _finish_c1m[(triton.cdiv(m * n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, m, n, x.stride(0), CAP=C1_CAP,
                                                    BLOCK=finish_block, GDC=gdc, EARLY=early, num_warps=4, **kw)
    return out


# ---- v2: 32-bit word loads, decode straight to FP32, eight outputs per call ----------
# Lane j (0-3) of a word: f = (s_j & 0x80) << 24 | (E - c_j) << 23 | (s_j & 0x7F) << 16, with
# s_j byte j of the sign/mantissa word and c_j the code nibble. prmt puts s_j in bytes 2 and 3
# (the mask keeps the sign in bit 31 and the mantissa in 16-22); a second prmt puts the
# exponent byte in byte 3, and >> 1 moves it to bits 23-30.


def _dec8_asm():
    lines = [".reg .b32 cl, ch, el, eh, t, u;",
             "and.b32 cl, $10, 0x0F0F0F0F;",
             "shr.b32 ch, $10, 4;",
             "and.b32 ch, ch, 0x0F0F0F0F;",
             "sub.u32 el, $11, cl;",
             "sub.u32 eh, $11, ch;"]
    for half, src, ex in ((0, "$8", "el"), (1, "$9", "eh")):
        for j in range(4):
            out = f"${half * 4 + j}"
            lines += [f"prmt.b32 t, {src}, 0, 0x{j}{j}44;",
                      f"prmt.b32 u, {ex}, 0, 0x{j}444;",
                      "shr.b32 u, u, 1;",
                      f"lop3.b32 {out}, t, 0x807F0000, u, 0xEA;"]
    return "{\n" + "\n".join(lines) + "\n}"


_DEC8_C = tl.constexpr(_dec8_asm())


@triton.jit
def _decode8(smlo, smhi, nib, e4, ASM: tl.constexpr):
    """Eight FP32 weights per position from three 32-bit words: lanes 0-3 of the low half
    (columns 4q + j) and of the high half (columns 64 + 4q + j). e4 is E in every byte."""
    if ASM:
        return tl.inline_asm_elementwise(_DEC8_C, "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r", [smlo, smhi, nib, e4],
                                         dtype=(tl.float32,) * 8, is_pure=True, pack=1)
    else:
        e = e4 & 0xFF
        cl = nib & 0x0F0F0F0F
        ch = (nib >> 4) & 0x0F0F0F0F
        lo0 = (((smlo >> 0) & 0x80) << 24) | ((e - ((cl >> 0) & 15)) << 23) | (((smlo >> 0) & 0x7F) << 16)
        lo1 = (((smlo >> 8) & 0x80) << 24) | ((e - ((cl >> 8) & 15)) << 23) | (((smlo >> 8) & 0x7F) << 16)
        lo2 = (((smlo >> 16) & 0x80) << 24) | ((e - ((cl >> 16) & 15)) << 23) | (((smlo >> 16) & 0x7F) << 16)
        lo3 = (((smlo >> 24) & 0x80) << 24) | ((e - ((cl >> 24) & 15)) << 23) | (((smlo >> 24) & 0x7F) << 16)
        hi0 = (((smhi >> 0) & 0x80) << 24) | ((e - ((ch >> 0) & 15)) << 23) | (((smhi >> 0) & 0x7F) << 16)
        hi1 = (((smhi >> 8) & 0x80) << 24) | ((e - ((ch >> 8) & 15)) << 23) | (((smhi >> 8) & 0x7F) << 16)
        hi2 = (((smhi >> 16) & 0x80) << 24) | ((e - ((ch >> 16) & 15)) << 23) | (((smhi >> 16) & 0x7F) << 16)
        hi3 = (((smhi >> 24) & 0x80) << 24) | ((e - ((ch >> 24) & 15)) << 23) | (((smhi >> 24) & 0x7F) << 16)
        return (lo0.to(tl.float32, bitcast=True), lo1.to(tl.float32, bitcast=True),
                lo2.to(tl.float32, bitcast=True), lo3.to(tl.float32, bitcast=True),
                hi0.to(tl.float32, bitcast=True), hi1.to(tl.float32, bitcast=True),
                hi2.to(tl.float32, bitcast=True), hi3.to(tl.float32, bitcast=True))


@triton.jit(do_not_specialize=["SPLIT_K"])
def _gemv_c1w(x_ptr, wc_ptr, e_ptr, acc_ptr, N, K, SPLIT_K, ASM: tl.constexpr, GDC: tl.constexpr,
              PIPE: tl.constexpr = 1):
    """_gemv_c1 with 32-bit word loads (wc_ptr is the blocks viewed as int32: 3072 words per
    block, planes of 1024), FP32 decode and fused multiply-adds. PIPE as in _gemv_c1."""
    if GDC:
        gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    offs_r = tl.arange(0, 64)
    offs_q = tl.arange(0, 16)
    w_off = offs_r[:, None] * 16 + offs_q[None, :]
    a0 = tl.zeros((64, 16), dtype=tl.float32)
    a1 = tl.zeros((64, 16), dtype=tl.float32)
    a2 = tl.zeros((64, 16), dtype=tl.float32)
    a3 = tl.zeros((64, 16), dtype=tl.float32)
    blk = pid_n * num_kb + kb_start
    for kb in tl.range(kb_start, kb_end, num_stages=PIPE):
        base = wc_ptr + blk.to(tl.int64) * 3072
        smlo = tl.load(base + w_off)
        smhi = tl.load(base + 1024 + w_off)
        nib = tl.load(base + 2048 + w_off)
        e = tl.load(e_ptr + blk).to(tl.int32)
        e4 = tl.zeros((64, 16), dtype=tl.int32) + e * 0x01010101
        l0, l1, l2, l3, h0, h1, h2, h3 = _decode8(smlo, smhi, nib, e4, ASM)
        xb = x_ptr + kb * 128 + offs_q * 4
        a0 = tl.fma(l0, tl.load(xb + 0).to(tl.float32)[None, :], a0)
        a1 = tl.fma(l1, tl.load(xb + 1).to(tl.float32)[None, :], a1)
        a2 = tl.fma(l2, tl.load(xb + 2).to(tl.float32)[None, :], a2)
        a3 = tl.fma(l3, tl.load(xb + 3).to(tl.float32)[None, :], a3)
        a0 = tl.fma(h0, tl.load(xb + 64).to(tl.float32)[None, :], a0)
        a1 = tl.fma(h1, tl.load(xb + 65).to(tl.float32)[None, :], a1)
        a2 = tl.fma(h2, tl.load(xb + 66).to(tl.float32)[None, :], a2)
        a3 = tl.fma(h3, tl.load(xb + 67).to(tl.float32)[None, :], a3)
        blk += 1
    row = tl.sum(a0 + a1 + a2 + a3, axis=1)
    if GDC:
        gdc_wait()
    tl.atomic_add(acc_ptr + pid_n * 64 + offs_r, row, sem="relaxed")


def c1w_matvec(x, comp, n, k, split, state, asm=True, gdc=True, warps=4, stages=3, pipe=1):
    """c1_matvec with _gemv_c1w (word loads, FP32 decode, FMA)."""
    blocks, e_max, corr = comp
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _gemv_c1w[(n // C1_BN, split)](x, blocks.view(torch.int32), e_max, state["acc"], n, k, split, ASM=asm, GDC=gdc,
                                   PIPE=pipe, num_warps=warps, num_stages=stages)
    kw = {"launch_pdl": True} if gdc else {}
    _finish_c1[(triton.cdiv(n, 128),)](state["acc"], out, corr, corr.stride(0), x, n, CAP=C1_CAP, BLOCK=128, GDC=gdc,
                                       num_warps=4, **kw)
    return out
