"""c1 with tensor cores in Gluon (prototype; not a submission yet).

Triton's tl.dot on decoded c1 weights stores the BF16 tile to shared memory and runs wgmma from
there (one layout conversion per block; `ncu` 4ac615: 25 % occupancy, scoreboard stalls). Here
the compressed bytes are loaded straight into the wgmma A-operand register layout
(DotOperandLayout of an NVMMA v3 parent), decoded in registers, and wgmma runs register-A
("RS"): no BF16 round trip through shared memory. x goes through shared memory as the B operand.

Swap-AB as in submission.py: 64 rows of W (one warpgroup, wgmma M = 64) x BLOCK_M tokens.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, warpgroup_mma, warpgroup_mma_wait


_G_DEC_LO = """{
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
_G_DEC_LO_C = gl.constexpr(_G_DEC_LO)
_G_DEC_HI_C = gl.constexpr(_G_DEC_LO.replace("and.b32 c, $3, 0x0F0F0F0F;", "shr.b32 c, $3, 4;\nand.b32 c, c, 0x0F0F0F0F;"))


@gluon.jit
def _decode_asm(sm, nib, e8, HIGH: gl.constexpr):
    """The c1.py PTX decode (prmt, per-byte subtract, lop3), four weights per call. Elementwise, so
    any layout works as long as sm, nib and e8 (E splat, uint8) share it."""
    if HIGH:
        return gl.inline_asm_elementwise(_G_DEC_HI_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=gl.bfloat16, is_pure=True,
                                         pack=4)
    else:
        return gl.inline_asm_elementwise(_G_DEC_LO_C, "=r,=r,r,r,r", [sm, nib, e8], dtype=gl.bfloat16, is_pure=True,
                                         pack=4)


@gluon.jit
def _decode_rs(sm, nib, e, HIGH: gl.constexpr):
    s = sm.to(gl.int32)
    if HIGH:
        code = (nib.to(gl.int32) >> 4) & 15
    else:
        code = nib.to(gl.int32) & 15
    ex = e - code
    return (((s & 0x80) << 8) | (ex << 7) | (s & 0x7F)).to(gl.int16).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _c1_gemm_rs(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 16])
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    x_l: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    b_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K

    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, a_l))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, a_l))
    p_off = rows[:, None] * 64 + cols[None, :]
    xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_l))
    xk = gl.arange(0, 64, layout=gl.SliceLayout(0, x_l))
    x_off = xm[:, None] * stride_xm + xk[None, :]
    x_mask = (xm < M)[:, None]

    x_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, 64], b_sl)
    acc = gl.zeros([64, BLOCK_M], gl.float32, layout=mma)
    blk = pid_n * num_kb + kb_start
    # Software pipelining in registers: block kb + 1's bytes and x are loaded before block kb
    # is decoded and multiplied (masked off past the CTA's K range).
    base = wc_ptr + blk.to(gl.int64) * 12288
    n_lo = gl.load(base + p_off)
    n_hi = gl.load(base + 4096 + p_off)
    n_nib = gl.load(base + 8192 + p_off)
    n_e = gl.load(e_ptr + blk).to(gl.int32)
    n_xlo = gl.load(x_ptr + kb_start * 128 + x_off, mask=x_mask, other=0.0)
    n_xhi = gl.load(x_ptr + kb_start * 128 + 64 + x_off, mask=x_mask, other=0.0)
    for kb in range(kb_start, kb_end):
        sm_lo = n_lo
        sm_hi = n_hi
        nib = n_nib
        e = n_e
        x_lo = n_xlo
        x_hi = n_xhi
        more = kb + 1 < kb_end
        nbase = wc_ptr + (blk + 1).to(gl.int64) * 12288
        wm = (p_off >= 0) & more
        n_lo = gl.load(nbase + p_off, mask=wm, other=0)
        n_hi = gl.load(nbase + 4096 + p_off, mask=wm, other=0)
        n_nib = gl.load(nbase + 8192 + p_off, mask=wm, other=0)
        n_e = gl.load(e_ptr + blk + 1, mask=more, other=0).to(gl.int32)
        n_xlo = gl.load(x_ptr + (kb + 1) * 128 + x_off, mask=x_mask & more, other=0.0)
        n_xhi = gl.load(x_ptr + (kb + 1) * 128 + 64 + x_off, mask=x_mask & more, other=0.0)
        w_lo = _decode_rs(sm_lo, nib, e, False)
        w_hi = _decode_rs(sm_hi, nib, e, True)
        gl.barrier()  # the previous step's wgmma reads of x_smem are done (wgmma is synchronous here)
        x_smem.index(0).store(x_lo)
        x_smem.index(1).store(x_hi)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(w_lo, x_smem.index(0).permute((1, 0)), acc)
        acc = warpgroup_mma(w_hi, x_smem.index(1).permute((1, 0)), acc)
        blk += 1
    o_r = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
    o_off = o_m[None, :] * N + (pid_n * 64 + o_r)[:, None]
    gl.atomic_add(acc_ptr + o_off, acc, mask=(o_m < M)[None, :], sem="relaxed")



def c1r_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """c1 blocks [nblk, 12288] -> the same bytes with each 64 x 64 plane in wgmma A-fragment order
    (c1r): thread t of the warpgroup finds its 32 bytes at t * 32. Plane element (r, c), with
    r = 16 w + 8 h + g and c = 16 ch + 8 chi + 2 tig + p, goes to
    ((((((w * 8 + g) * 4 + tig) * 4 + ch) * 2 + h) * 2 + chi) * 2 + p). Lossless: a permutation."""
    nblk = blocks.shape[0]
    v = blocks.view(nblk, 3, 4, 2, 8, 4, 2, 4, 2)          # (blk, plane, w, h, g, ch, chi, tig, p)
    return v.permute(0, 1, 2, 4, 7, 5, 3, 6, 8).reshape(nblk, 12288).contiguous()


@gluon.jit
def _load_plane_a(ptr, t_off, j_off, a_l: gl.constexpr):
    """One c1r plane (4096 bytes at ptr) -> [64, 64] uint8 in the A-operand layout a_l: two 16-byte
    loads per thread, then a register-only permutation."""
    raw = gl.load(ptr + t_off[:, None] * 32 + j_off[None, :])
    v = gl.permute(gl.reshape(raw, [4, 8, 4, 4, 2, 2, 2]), (0, 4, 1, 3, 5, 2, 6))
    return gl.convert_layout(gl.reshape(v, [64, 64]), a_l)


@gluon.jit
def _c1_gemm_rs2(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: gl.constexpr,
                 ASM: gl.constexpr = False):
    """_c1_gemm_rs on c1r blocks (c1r_blocks): two 16-byte loads per thread and plane."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 16])
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    ld_l: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [4, 1], [1, 0])
    x_l: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    b_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    t_off = gl.arange(0, 128, layout=gl.SliceLayout(1, ld_l))
    j_off = gl.arange(0, 32, layout=gl.SliceLayout(0, ld_l))
    xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_l))
    xk = gl.arange(0, 64, layout=gl.SliceLayout(0, x_l))
    x_off = xm[:, None] * stride_xm + xk[None, :]
    x_mask = (xm < M)[:, None]

    x_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, 64], b_sl)
    acc = gl.zeros([64, BLOCK_M], gl.float32, layout=mma)
    blk = pid_n * num_kb + kb_start
    for kb in range(kb_start, kb_end):
        base = wc_ptr + blk.to(gl.int64) * 12288
        sm_lo = _load_plane_a(base, t_off, j_off, a_l)
        sm_hi = _load_plane_a(base + 4096, t_off, j_off, a_l)
        nib = _load_plane_a(base + 8192, t_off, j_off, a_l)
        x_lo = gl.load(x_ptr + kb * 128 + x_off, mask=x_mask, other=0.0)
        x_hi = gl.load(x_ptr + kb * 128 + 64 + x_off, mask=x_mask, other=0.0)
        if ASM:
            e8 = gl.zeros_like(sm_lo) + gl.load(e_ptr + blk)
            w_lo = _decode_asm(sm_lo, nib, e8, False)
            w_hi = _decode_asm(sm_hi, nib, e8, True)
        else:
            e = gl.load(e_ptr + blk).to(gl.int32)
            w_lo = _decode_rs(sm_lo, nib, e, False)
            w_hi = _decode_rs(sm_hi, nib, e, True)
        gl.barrier()  # the previous step's wgmma reads of x_smem are done (wgmma is synchronous here)
        x_smem.index(0).store(x_lo)
        x_smem.index(1).store(x_hi)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(w_lo, x_smem.index(0).permute((1, 0)), acc)
        acc = warpgroup_mma(w_hi, x_smem.index(1).permute((1, 0)), acc)
        blk += 1
    o_r = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
    o_off = o_m[None, :] * N + (pid_n * 64 + o_r)[:, None]
    gl.atomic_add(acc_ptr + o_off, acc, mask=(o_m < M)[None, :], sem="relaxed")


@gluon.jit
def _to_a(raw, a_l: gl.constexpr):
    """A c1r plane loaded as [128 threads, 32 bytes] -> [64, 64] in the A-operand layout a_l
    (a register-only permutation)."""
    v = gl.permute(gl.reshape(raw, [4, 8, 4, 4, 2, 2, 2]), (0, 4, 1, 3, 5, 2, 6))
    return gl.convert_layout(gl.reshape(v, [64, 64]), a_l)


@gluon.jit
def _c1_gemm_rs3(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: gl.constexpr):
    """_c1_gemm_rs2 with the next block's bytes and x prefetched into registers (raw, before
    the register permutation)."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 16])
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    ld_l: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [4, 1], [1, 0])
    x_l: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    b_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    t_off = gl.arange(0, 128, layout=gl.SliceLayout(1, ld_l))
    j_off = gl.arange(0, 32, layout=gl.SliceLayout(0, ld_l))
    xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_l))
    xk = gl.arange(0, 64, layout=gl.SliceLayout(0, x_l))
    x_off = xm[:, None] * stride_xm + xk[None, :]
    x_mask = (xm < M)[:, None]

    x_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, 64], b_sl)
    acc = gl.zeros([64, BLOCK_M], gl.float32, layout=mma)
    blk = pid_n * num_kb + kb_start
    ld = t_off[:, None] * 32 + j_off[None, :]
    base = wc_ptr + blk.to(gl.int64) * 12288
    n_lo = gl.load(base + ld)
    n_hi = gl.load(base + 4096 + ld)
    n_nib = gl.load(base + 8192 + ld)
    n_e = gl.load(e_ptr + blk).to(gl.int32)
    n_xlo = gl.load(x_ptr + kb_start * 128 + x_off, mask=x_mask, other=0.0)
    n_xhi = gl.load(x_ptr + kb_start * 128 + 64 + x_off, mask=x_mask, other=0.0)
    for kb in range(kb_start, kb_end):
        r_lo = n_lo
        r_hi = n_hi
        r_nib = n_nib
        e = n_e
        x_lo = n_xlo
        x_hi = n_xhi
        more = kb + 1 < kb_end
        wm = (ld >= 0) & more
        nbase = wc_ptr + (blk + 1).to(gl.int64) * 12288
        n_lo = gl.load(nbase + ld, mask=wm, other=0)
        n_hi = gl.load(nbase + 4096 + ld, mask=wm, other=0)
        n_nib = gl.load(nbase + 8192 + ld, mask=wm, other=0)
        n_e = gl.load(e_ptr + blk + 1, mask=more, other=0).to(gl.int32)
        n_xlo = gl.load(x_ptr + (kb + 1) * 128 + x_off, mask=x_mask & more, other=0.0)
        n_xhi = gl.load(x_ptr + (kb + 1) * 128 + 64 + x_off, mask=x_mask & more, other=0.0)
        sm_lo = _to_a(r_lo, a_l)
        sm_hi = _to_a(r_hi, a_l)
        nib = _to_a(r_nib, a_l)
        w_lo = _decode_rs(sm_lo, nib, e, False)
        w_hi = _decode_rs(sm_hi, nib, e, True)
        gl.barrier()  # the previous step's wgmma reads of x_smem are done (wgmma is synchronous here)
        x_smem.index(0).store(x_lo)
        x_smem.index(1).store(x_hi)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(w_lo, x_smem.index(0).permute((1, 0)), acc)
        acc = warpgroup_mma(w_hi, x_smem.index(1).permute((1, 0)), acc)
        blk += 1
    o_r = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
    o_off = o_m[None, :] * N + (pid_n * 64 + o_r)[:, None]
    gl.atomic_add(acc_ptr + o_off, acc, mask=(o_m < M)[None, :], sem="relaxed")


@gluon.jit
def _c1_gemm_rs4(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: gl.constexpr,
                 STAGES: gl.constexpr, ASM: gl.constexpr = False):
    """_c1_gemm_rs2 with a STAGES-deep cp.async ring in shared memory for the c1r bytes (12 KB) and
    x of each block; each thread then reads its 32 bytes per plane with two 16-byte shared loads."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 16])
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    ld_l: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [4, 1], [1, 0])
    x_l: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    w_sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    b_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    t_off = gl.arange(0, 128, layout=gl.SliceLayout(1, ld_l))
    j_off = gl.arange(0, 32, layout=gl.SliceLayout(0, ld_l))
    ld = t_off[:, None] * 32 + j_off[None, :]
    xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_l))
    xk = gl.arange(0, 64, layout=gl.SliceLayout(0, x_l))
    x_off = xm[:, None] * stride_xm + xk[None, :]
    x_mask = (xm < M)[:, None]

    w_s = gl.allocate_shared_memory(gl.uint8, [STAGES * 3, 128, 32], w_sl)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES * 2, BLOCK_M, 64], b_sl)
    blk0 = pid_n * num_kb + kb_start
    n_iter = kb_end - kb_start
    # prologue: blocks 0 .. STAGES - 2 in flight
    for i in gl.static_range(STAGES - 1):
        ok = i < n_iter
        base = wc_ptr + (blk0 + i).to(gl.int64) * 12288
        for p in gl.static_range(3):
            async_copy.async_load(w_s.index(i * 3 + p), base + p * 4096 + ld, mask=(ld >= 0) & ok)
        xb = x_ptr + (kb_start + i) * 128 + x_off
        async_copy.async_load(x_s.index(i * 2), xb, mask=x_mask & ok)
        async_copy.async_load(x_s.index(i * 2 + 1), xb + 64, mask=x_mask & ok)
        async_copy.commit_group()
    acc = gl.zeros([64, BLOCK_M], gl.float32, layout=mma)
    for i in range(n_iter):
        # issue block i + STAGES - 1 into the slot block i - 1 used (its wgmma finished: synchronous)
        j = i + STAGES - 1
        sj = j % STAGES
        ok = j < n_iter
        base = wc_ptr + (blk0 + j).to(gl.int64) * 12288
        for p in gl.static_range(3):
            async_copy.async_load(w_s.index(sj * 3 + p), base + p * 4096 + ld, mask=(ld >= 0) & ok)
        xb = x_ptr + (kb_start + j) * 128 + x_off
        async_copy.async_load(x_s.index(sj * 2), xb, mask=x_mask & ok)
        async_copy.async_load(x_s.index(sj * 2 + 1), xb + 64, mask=x_mask & ok)
        async_copy.commit_group()
        async_copy.wait_group(STAGES - 1)  # block i has landed
        gl.barrier()
        si = i % STAGES
        sm_lo = _to_a(w_s.index(si * 3).load(ld_l), a_l)
        sm_hi = _to_a(w_s.index(si * 3 + 1).load(ld_l), a_l)
        nib = _to_a(w_s.index(si * 3 + 2).load(ld_l), a_l)
        if ASM:
            e8 = gl.zeros_like(sm_lo) + gl.load(e_ptr + blk0 + i)
            w_lo = _decode_asm(sm_lo, nib, e8, False)
            w_hi = _decode_asm(sm_hi, nib, e8, True)
        else:
            e = gl.load(e_ptr + blk0 + i).to(gl.int32)
            w_lo = _decode_rs(sm_lo, nib, e, False)
            w_hi = _decode_rs(sm_hi, nib, e, True)
        fence_async_shared()
        acc = warpgroup_mma(w_lo, x_s.index(si * 2).permute((1, 0)), acc)
        acc = warpgroup_mma(w_hi, x_s.index(si * 2 + 1).permute((1, 0)), acc)
        gl.barrier()  # every warp is done with slot si before it is refilled
    async_copy.wait_group(0)
    o_r = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
    o_off = o_m[None, :] * N + (pid_n * 64 + o_r)[:, None]
    gl.atomic_add(acc_ptr + o_off, acc, mask=(o_m < M)[None, :], sem="relaxed")


@gluon.jit
def _c1_issue(w_s, x_s, slot, wbase, xbase, ld, x_off, ok, x_mask):
    for p in gl.static_range(3):
        async_copy.async_load(w_s.index(slot * 3 + p), wbase + p * 4096 + ld, mask=(ld >= 0) & ok)
    async_copy.async_load(x_s.index(slot * 2), xbase + x_off, mask=x_mask & ok)
    async_copy.async_load(x_s.index(slot * 2 + 1), xbase + 64 + x_off, mask=x_mask & ok)
    async_copy.commit_group()


@gluon.jit
def _c1_gemm_rs5(x_ptr, wc_ptr, e_ptr, acc_ptr, M, N, K, stride_xm, SPLIT_K, BLOCK_M: gl.constexpr,
                 STAGES: gl.constexpr):
    """_c1_gemm_rs4 (packed PTX decode) with asynchronous wgmma: block i's two MMAs run while block
    i + 1 is decoded; the slot of block i - 1 is refilled once its MMAs are done."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 16])
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    ld_l: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [4, 1], [1, 0])
    x_l: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    w_sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    b_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    num_kb = K // 128
    kb_start = pid_k * num_kb // SPLIT_K
    kb_end = (pid_k + 1) * num_kb // SPLIT_K
    t_off = gl.arange(0, 128, layout=gl.SliceLayout(1, ld_l))
    j_off = gl.arange(0, 32, layout=gl.SliceLayout(0, ld_l))
    ld = t_off[:, None] * 32 + j_off[None, :]
    xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_l))
    xk = gl.arange(0, 64, layout=gl.SliceLayout(0, x_l))
    x_off = xm[:, None] * stride_xm + xk[None, :]
    x_mask = (xm < M)[:, None]

    w_s = gl.allocate_shared_memory(gl.uint8, [STAGES * 3, 128, 32], w_sl)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [STAGES * 2, BLOCK_M, 64], b_sl)
    blk0 = pid_n * num_kb + kb_start
    n_iter = kb_end - kb_start
    for i in gl.static_range(STAGES - 1):
        _c1_issue(w_s, x_s, i, wc_ptr + (blk0 + i).to(gl.int64) * 12288, x_ptr + (kb_start + i) * 128, ld, x_off,
                  i < n_iter, x_mask)
    acc = gl.zeros([64, BLOCK_M], gl.float32, layout=mma)
    for i in range(n_iter):
        async_copy.wait_group(STAGES - 2)  # block i has landed
        gl.barrier()
        si = i % STAGES
        sm_lo = _to_a(w_s.index(si * 3).load(ld_l), a_l)
        sm_hi = _to_a(w_s.index(si * 3 + 1).load(ld_l), a_l)
        nib = _to_a(w_s.index(si * 3 + 2).load(ld_l), a_l)
        e8 = gl.zeros_like(sm_lo) + gl.load(e_ptr + blk0 + i)
        w_lo = _decode_asm(sm_lo, nib, e8, False)
        w_hi = _decode_asm(sm_hi, nib, e8, True)
        fence_async_shared()
        acc = warpgroup_mma(w_lo, x_s.index(si * 2).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma(w_hi, x_s.index(si * 2 + 1).permute((1, 0)), acc, is_async=True)
        acc, w_lo, w_hi = warpgroup_mma_wait(num_outstanding=2, deps=[acc, w_lo, w_hi])  # block i - 1's MMAs done
        gl.barrier()
        j = i + STAGES - 1
        _c1_issue(w_s, x_s, j % STAGES, wc_ptr + (blk0 + j).to(gl.int64) * 12288, x_ptr + (kb_start + j) * 128, ld,
                  x_off, j < n_iter, x_mask)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=[acc])
    async_copy.wait_group(0)
    o_r = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
    o_off = o_m[None, :] * N + (pid_n * 64 + o_r)[:, None]
    gl.atomic_add(acc_ptr + o_off, acc, mask=(o_m < M)[None, :], sem="relaxed")


def c1rs2_matmul(x, compr, n, k, split, state, block_m=16, finish_block=512, warps=4, prefetch=False, stages=0,
                 asm=False, async_mma=False):
    """c1rs_matmul on c1r blocks: compr = (c1r_blocks(blocks), e_max, corr). prefetch: _c1_gemm_rs3;
    stages > 1: _c1_gemm_rs4 with that many cp.async stages."""
    blocks, e_max, corr = compr
    m = x.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if async_mma:
        _c1_gemm_rs5[(n // 64, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split, BLOCK_M=block_m,
                                       STAGES=max(stages, 2), num_warps=warps)
    elif stages > 1:
        _c1_gemm_rs4[(n // 64, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split, BLOCK_M=block_m,
                                       STAGES=stages, ASM=asm, num_warps=warps)
    else:
        if prefetch:
            _c1_gemm_rs3[(n // 64, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split,
                                           BLOCK_M=block_m, num_warps=warps)
        else:
            _c1_gemm_rs2[(n // 64, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split,
                                           BLOCK_M=block_m, ASM=asm, num_warps=warps)
    _finish_c1m[(triton.cdiv(m * n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, m, n, x.stride(0),
                                                    CAP=C1_CAP, BLOCK=finish_block, GDC=True, EARLY=True, num_warps=4,
                                                    launch_pdl=True)
    return out

def c1rs_matmul(x, comp, n, k, split, state, block_m=16, finish_block=512, warps=4):
    """x [M, K] @ W[:n].T (M <= block_m <= 256) via _c1_gemm_rs + _finish_c1m (c1.py: the
    corrections, before griddepcontrol.wait). The main kernel is launched without PDL."""
    blocks, e_max, corr = comp
    m = x.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _c1_gemm_rs[(n // 64, split)](x, blocks, e_max, state["acc"], m, n, k, x.stride(0), split, BLOCK_M=block_m,
                                  num_warps=warps)
    _finish_c1m[(triton.cdiv(m * n, finish_block),)](state["acc"], out, corr, corr.stride(0), x, m, n, x.stride(0),
                                                    CAP=C1_CAP, BLOCK=finish_block, GDC=True, EARLY=True, num_warps=4,
                                                    launch_pdl=True)
    return out
