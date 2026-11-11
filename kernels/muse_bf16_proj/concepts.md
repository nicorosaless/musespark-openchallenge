# BF16 projections and the MoE with few tokens: the concepts

A document for learning what the kernels of the Open Kernel Challenge bf16_proj problem do, why they are
built this way, and what was tried without success. The numbers are from the judge's H200 unless
"bench" is stated (the H100 on a university cluster). Revised on 30 Oct with the final numbers; the
details of what didn't win are in [`what-didnt-win.md`](what-didnt-win.md).

## 1. The problem

Each model layer has five **projections**: multiplications of the activation `x` (one token per
row) by a weight matrix `W`:

```
out[m, n] = x[m, K] · W[n, K]ᵀ        (m tokens, n outputs, K inputs)
```

| layer | W (n × K) | type | bytes |
|---|---|---|---|
| qkv_gate | 2304 × 8192 | BF16 | 37.7 MB |
| o | 8192 × 1024 | BF16 | 16.8 MB |
| pooler | 4096 × 8192 | BF16 | 67.1 MB |
| unpooler | 8192 × 4096 (1024 rows with ≤ 8 tokens) | BF16 | 67.1 MB (8.4 MB) |
| router | 256 × 8192 | FP32 | 8.4 MB |

`m` is the token count of the step: 1, 16 and 32 in *decode* (generating text), and 8, 128 and 256
in *verify* (checking 8-token drafts). The reference is the SGLang kernel (cuBLAS underneath).

## 2. Why a BF16 GEMV is memory-bound

With a single token, each weight is read **once** and used in **one** multiply and one add: 2 FLOP
per 2 bytes. The H200 does ~989 TFLOP/s but reads only ~4.8 TB/s from its memory:

```
pooler with m = 1:  67 MB at 4.8 TB/s  = 14.0 µs   (read the weights)
                    67 MFLOP at 989 TFLOP/s = 0.07 µs   (do the arithmetic)
```

The GPU waits on memory almost all the time. In general, each byte of weights yields `m` FLOP: the
arithmetic only starts to matter when `m` approaches 989 / 4.8 ≈ 200 tokens. With m ≤ 32 **the time
is reading W**, and there are two ways to reduce it:

1. read memory as close to 4.8 TB/s as possible (occupancy, loads in flight, few launches);
2. **read fewer bytes**, without losing a single bit of the weights: lossless compression.

## 3. The c1 format: lossless BF16 in ~0.75 bytes per value

A BF16 is 16 bits: **sign (1) | exponent (8) | mantissa (7)**. The mantissa and the sign are close to
random, but the exponents are not: the weights in one region of the matrix have similar magnitudes,
so their exponents are very close to the region's maximum exponent.

c1 splits W into **blocks of 64 rows × 128 columns** and stores each weight in **12 bits**:

```
  BF16:   s eeeeeeee mmmmmmm          (16 bits)
  c1:     s mmmmmmm  +  cccc          (8 bits + 4 bits)
          └ byte ┘      └ code = E − e  (E: maximum exponent of the block, 1 byte per block)
```

Each block takes 12 KB instead of 16:

```
  block of 64 × 128 weights (12,288 bytes)
  ┌──────────────────────────────────────────────────────┐
  │ s|mantissa, columns 0-63                             │  4096 bytes  (64 × 64)
  ├──────────────────────────────────────────────────────┤
  │ s|mantissa, columns 64-127                           │  4096 bytes
  ├──────────────────────────────────────────────────────┤
  │ codes: low nibble = column c, high nibble = col c+64 │  4096 bytes
  └──────────────────────────────────────────────────────┘
  + 1 byte E per block
```

**Real example** (first block of the pooler in the judge's weights). The largest weight in the block
is 0.032958984375, with exponent **E = 122**.

| weight | BF16 bits | exp. | code E − e | s\|mantissa byte |
|---|---|---|---|---|
| 0.0146484375 | 0 01111000 1110000 | 120 | 2 | 0 1110000 |
| −0.017822265625 | 1 01111001 0010010 | 121 | 1 | 1 0010010 |
| 0.001190185546875 | 0 01110101 0011100 | 117 | 5 | 0 0011100 |

To decode: exponent = E − code, and the 16 bits are reassembled. It is exact bit for bit.

**Exponents more than 14 below E, and zeros,** do not fit: 4 bits only count from 0 to 15. In that
block, the weight 5.66·10⁻⁷ has exponent 106: 16 below E. Those weights are stored with the special
code 15 and byte 0, which decodes to a fixed value `d0 = 2^(E − 142)` (here 9.54·10⁻⁷). A per-row
**correction table** then stores (column, true value, d0), and the output adds

```
out[r] = (sum with the decoded weights) + Σ_j (v_j − d0_j) · x[col_j]
```

It is rare: in this block, 3 of 8192 weights. The block's code histogram is
`[52, 1186, 2585, 2077, 1106, 575, 309, 142, 68, 46, 19, 15, 6, 2, 1, 3]`: almost all weights are
1-4 binades below the maximum. On average there are ~1.1 corrections per row, but the per-row
maximum reaches 8 in the pooler. The v10 table allows 8 per row and the v11b table 16. If a layer
does not fit, that layer stays on plain BF16 (v9).

Result: **0.753 bytes per 2 bytes of BF16** in pooler and qkv_gate, 0.756 in unpooler and 0.774 in
o. `prepare()` compresses, decompresses and compares bit by bit; if anything differs, it does not
use c1.

**The decode fits in the kernel:** a few integer instructions per 4 weights (`prmt`, a per-byte
subtraction, `lop3`; inline PTX). It costs almost nothing: the main c1 kernel on the pooler took
14.59 µs, and 14.40 without decoding.

Effect on the H200, m = 1: pooler 18.1 → **15.3 µs**; o 5.9 → 5.7; unpooler (1024 rows) 4.5 → 4.1.
With 2-16 tokens, on the pooler, the tensor-core variant: m = 8 18.5 → 17.5. That took the kernel from
v9 (1.231x) to v10: **1.261x [1.242, 1.279]**, submit `b3445480`, its final mark (rank 663 of
1490 on 30 Oct).

The v10 pooler at m = 1 (15.5 µs in the submit) goes even below a pure read of its 64 MB in BF16
(~17.1 µs measured on the H200), because it reads 25% less. Per load, v10 runs at 56-67% of peak
bandwidth: decode 1 at 44.2 µs against a 28.9 floor; verify 16 at 76.4 against 43.9.

## 4. Split-K with a PDL finish

With m = 1, each output is a dot product of K numbers. If each CTA (a group of GPU threads) computes
64 whole output rows, the pooler has only 4096 / 64 = 64 CTAs for 132 SMs: half the GPU idle.
**Split-K** also splits K: each CTA computes its 64 rows over one slice of K, and the partial sums
are combined afterwards.

```
            K (8192) →
  rows     ┌────┬────┬────┬────┬────┬────┬────┬────┐
  0-63     │CTA0│CTA1│CTA2│CTA3│CTA4│CTA5│CTA6│CTA7│  → 8 partial sums of the same 64 rows
  64-127   │ ...                                   │
           └────┴────┴────┴────┴────┴────┴────┴────┘     64 tiles × 8 = 512 CTAs
```

The partials are summed in FP32 into a workspace with atomics (`red.add`). A finishing step remains:
convert to BF16, add the c1 corrections and reset the workspace to zero. As a normal kernel, it
would wait for the main kernel to finish and then pay its own launch. With **PDL** (*programmatic
dependent launch*), the finishing kernel is launched while the main one is still running, and it
waits inside (`griddepcontrol.wait`):

```
  no PDL:    [ main ················ ][gap][ finish ]
  with PDL:  [ main ················ ]
                  [ finish: launched, loads the correction table… | wait | converts ]
```

A detail that mattered: the corrections do not depend on the main kernel, so the finish loads them
**before** the `wait`. At first they were loaded after it, from a table laid out badly for the GPU,
and the finish cost ~3 µs. With the table reordered (one plane per field) and the load moved
earlier, the pooler at m = 1 went from ~17.4 to 15.3 µs.

## 5. What won a little: stream-K on qkv_gate

qkv_gate has 36 tiles of 64 rows. With split 8 that is 288 CTAs for ~528 slots: the GPU stays half
empty. **Stream-K** spreads the work like a continuous belt: all (tile, K slice) pairs in a row, and
each CTA takes an equal stretch, even if it starts or ends in the middle of a tile.

H200, qkv_gate at m = 1: 11.7 → **10.1-10.9 µs**. It is in v11b, together with cuBLAS "NN" (W
transposed and stored in `prepare`) for pooler and unpooler, which on the pooler at m = 128 goes
from 26.1 to 23.2 µs. v11b gave 1.271x [1.252, 1.290] in check, against 1.261x for v10. It is +1%
and does not clear the upper end of v10's 95% CI, so it was not submitted. It has been in the
decode_layer layer since the layer's v41.

## 6. What was tried and didn't win on the H200

| idea | what it is | H200 numbers (µs) | why not |
|---|---|---|---|
| stream-K in the tensor-core kernels (BF16 and c1) | the same as in §5, with m > 1 | qkv m8 11.46 against 11.33; o m8-32 ≥ 7 against 5.8 | with tensor cores the limit is not the work split |
| c1 with tensor cores (Triton) for m = 8-32 | decompress and use `tl.dot` | qkv m8 13.0 against 11.6; pooler m32 25.2 against 18.9 | Triton routes the decompressed BF16 through shared memory before the MMA |
| c1 in Gluon with the MMA reading from registers (and PTX decoding) | without that round trip through shared memory | best variant (rs2a) 17.3 against 17.6 on pooler m8 (tie); the pipelined ones, 19-30 against 17; at m = 1, 18.4 against 15.2 | a single warpgroup decodes and multiplies in series: latency |
| single kernel without split for m = 1 | no workspace and no finish | o 6.3 against 5.4; pooler 34 against 15 | few CTAs, few bytes in flight |
| custom GEMM for 128-256 tokens (Triton, CuTe DSL, TMA) | beat cuBLAS where it runs at ~60% | TMA: o m128 14.0 against 6.6, m256 1.3-2.3 times worse; CuTe: pooler only, 1.05-1.12x | cuBLAS (CUTLASS sm90) is already well built |
| **warp-specialized GEMM (WS)** in Gluon | 1 warp loads by TMA, others multiply; TMA epilogue | qkv m128 **16.7 against 17.0-17.2** (tie); pooler m128 24.1 against 21.9; m256 33 against 28 | on par with cuBLAS, not above it |
| DeepGEMM, FlashInfer (shipped in the image) | other libraries | tie with cuBLAS in every shape | — |
| c1 + WS (compression inside the 128-256 token GEMM) | read 25% less with large m | not done | in the layer, those projections are 13-24% of their shape: ~+1% at most |

General lesson: on the H200, with 4.8 TB/s, a kernel that trades bytes for instructions or for
latency can win on the H100 (1.3 TB/s, where everything is memory) and lose on the H200. **Only the
H200 decides.**

## 7. The MoE with few tokens and the PDL chain

After the projections, each token goes through **8 of 256 experts** (MoE). Each expert is two
matrices with **NVFP4** weights (4 bits plus one scale per 16 values): `h = x·W13ᵀ`, SwiGLU,
`y = h·W2ᵀ`, a norm and the weighted sum. This GPU holds 32 experts; with 1-32 tokens the work is
again reading weights (as in §2). The starting kernel (v24 of the MoE work) is **five kernels**:

```
  route prep → gemm1 → swiglu_reduce → gemm2 → combine
```

On the H200, with 1 token, the MoE takes ~37-39 µs against an 11.4 floor: the two GEMMs ~32 µs and
the three small kernels ~8.4. What was done:

- **PDL chain:** each kernel lets the next one start, and the next one waits before reading what it
  needs, as in §4 but along a chain of five. The CUDA kernels are launched with `cuLaunchKernelEx`
  and the PDL attribute. Bench: −1.5 to −3.5 µs per call. **decode_layer integrated it in its v45**
  (submit `b76b7763`, 1.675x, the layer's mark on 30 Oct).
- **Measured on the H200** (lab of check `af6d1b9c`, 28 Oct; full table in
  [`what-didnt-win.md`](what-didnt-win.md) §5), µs per call without the empty graph (4.3 µs), with
  2, 7, 9 and 12 distinct local experts:

  | idea | bs 1 | bs 8 | bs 16 | bs 32 |
  |---|---|---|---|---|
  | v24 (no PDL) | 33.8 | 71.1 | 94.3 | **115.5** |
  | **PDL chain** | **31.7** | **68.2** | **93.4** | 118.4 |
  | double buffering by half groups | 32.4 | 71.1 | 96.2 | 128.1 |
  | `moe_bs1`: the whole 1-token MoE in one cooperative kernel | 32.0-36.2 | — | — | — |
  | per-warp `cp.async` ring of 2-4 groups | 38.6-46.7 | 90-125 | 106-159 | 134-216 |
  | SwiGLU inside gemm1 (no atomics) | 33.0 | 72.7 | 100.4 | 135.4 |
  | gridded combine with the norm computed in gemm2 | 32.3 | 73.0 | 103.5 | 133.2 |

The PDL chain wins up to bs 16 (−2 to −3 µs at bs 1 and 8) and **loses ~3 µs at bs 32**, where it
tied on the bench. That is why the final file,
[`experiments/small_m_clean.py`](experiments/small_m_clean.py), is v24 plus the
PDL chain only for bs ≤ 16 (`PDL_MAX_BS = 16`), without any of the paths that lost. It is the base of
the single expert file: the MoE kernel builds on top of it and decode_layer integrates it. Everything
else loses or ties. The small kernels cost almost nothing with PDL, and what remains at bs 1 is the
**latency of the GEMMs** (~32 µs of an MoE whose whole floor is ~11), which none of the ideas tried
(double buffering, `cp.async`, single kernel) reduced on the H200.
