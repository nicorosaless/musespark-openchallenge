# muse_bf16_proj: where the headroom is (roofline per load, H200)

Table for the mark, v10 (submit `b3445480`, 1.261x), recomputed on 30 Oct from
[`evidence/runs/muse_bf16_proj/b34454802aa64dc4.json`](../../evidence/runs/muse_bf16_proj/b34454802aa64dc4.json).
Each load is the sum of the medians of its 5 layers, and the floor is the one in TASK.md
(4.8 TB/s). "Floor/v10" is the fraction of peak bandwidth I use. In parentheses, v3 (`4eb830e7`,
1.207x), which is the source of [`evidence/roofline-bf16.json`](../../evidence/roofline-bf16.json).

| load | SGLang µs | v10 µs | speedup | floor µs | floor/v10 | v3 µs (speedup; floor/v3) |
|---|---|---|---|---|---|---|
| decode 1 | 59.5 | **44.2** | 1.35 | 28.9 | 65% | 49.8 (1.19; 58%) |
| decode 16 | 76.5 | 61.6 | 1.24 | 41.4 | 67% | 63.2 (1.20; 66%) |
| decode 32 | 77.3 | 62.8 | 1.23 | 41.8 | 66% | 64.9 (1.20; 64%) |
| verify 1 | 76.7 | 48.2 | **1.59** | 29.0 | 60% | 49.9 (1.52; 58%) |
| verify 16 | 85.9 | 76.4 | 1.12 | 43.9 | 57% | 77.9 (1.11; 56%) |
| verify 32 | 101.2 | 92.7 | 1.09 | 51.7 | 56% | 95.0 (1.07; 54%) |

**Reading.**
- **The weakest loads are verify 16 and verify 32 (m 128 and 256).** There, every layer except the
  router runs on cuBLAS at ~1.1x. Four custom GEMMs did not beat it on the H200: pointer-based
  Triton, the CuTe DSL (pooler only, 1.05-1.12x), TMA in Triton, and warp specialization in Gluon,
  which ties on qkv_gate m=128. Neither did DeepGEMM nor FlashInfer. The details are in
  [`what-didnt-win.md`](what-didnt-win.md).
- Per layer, with m ≤ 32, in v10:
  - pooler: 15.5 µs at m=1 against a 14.0 floor (91%, thanks to c1), and 17.7-19.1 at m=8-32
    (74-79%);
  - unpooler at m=16/32, still on cuBLAS: 18.4-18.6 against 14.1 (76%);
  - qkv_gate: 12.1-12.6 against 7.9 (63-65%); v11b brings it down to 10.9 at m=1;
  - o: 6.5-7.0 against 3.5 (~52%).
- **The fixed cost per kernel is measured.** A kernel that only reads N bytes with a cold L2
  (check `62cd23e1`, 26 Oct) gives a line with an intercept of ~0.2-0.8 µs and a slope of
  ~3.9 TB/s (64 MB in 17.1 µs). My kernels ran 1.6-3.0 µs above it. v9 (PDL finish) and v10 (c1,
  fewer bytes) cut part of that. The v10 pooler at m=1 goes below a pure BF16 read because it reads
  25% less.
- **Correction (26 Oct 05:38):** an earlier version of this document attributed the 4 µs of
  qkv_gate at m=1 (8.4 µs in the profile against 12.5 in the shape) to something outside the
  kernel. That was wrong. The judge's profile measures with a warm L2 and the shape with a cold L2:
  the difference is reading from HBM the weights that were already in L2 in the profile.
- **Board reference (1 Nov):** rank 40 is at 1.586x and rank 300 at 1.529x. For a 1.59x geometric
  mean, every load would have to run at ~80% of peak.
