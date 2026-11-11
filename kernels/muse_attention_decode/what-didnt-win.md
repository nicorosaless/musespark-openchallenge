# What did not win on the H200 (attention)

Each row is one idea measured with a judge check on the H200, compared with the reference version
of that moment. All checks passed 16/16. The µs are the judge's scored median on the shape the idea
targeted. The evidence for each check is in
[`evidence/runs/muse_attention_decode/<check>.json`](../../evidence/runs/muse_attention_decode/).

The "bench" column is the H100 of the university cluster (runs from [`scripts/bench/bench`](../../scripts/bench/bench), ~1.5 TB/s). It is included
because for several ideas it predicted the opposite of what happened on the H200.

## Verify full (batch 16 / 32)

Reference: 4 stages + `PNOWAIT` + `PSMEM`, one KV head per CTA and 4 CTAs/SM: **166-170 / 317-322 µs**.

| idea | what it changed | CTAs/SM | bench | H200 verify 16 / 32 (µs) | check |
|---|---|---|---|---|---|
| `KVP = 2` (wgmma, two heads per CTA) | whole 128 B rows | 2 | −31% | **205.1 / 397.4** | `ed6b7812` |
| `ws` with `KVP = 2` | coalesced producer, two consumers | 2 | | 200.6 / 386.0 | `ae516eee` |
| `PP` | two warpgroups on alternate blocks (ping-pong) | 2 | | 202.0 / 394.6 | `0b5d089e` |
| `PVA` | PV with asynchronous wgmma; **serialized** (C7515), invalid measurement | 4 | | 188.2 / 358.7 | `3474bcfc` |
| `PVA`, not serialized | PV with wgmma, `MINB = 3`, 136 registers | 3 | −1% | **187.8 / 363.0** | `aaf0450b` |
| `PVA = 2` | PV in E4M3 as in FA3 | 4 | +1.5% | **246.7 / 481.3** | `b58fb252` |
| `PVA = 2` + `VTROT` | no bank conflicts when transposing V | 4 | +3.5% | 236.6 / 459.3 | `ed1ddb27` |
| `OVL` | QKᵀ(j + 1) in flight during PV(j), 2 stages | 4 | +3% | 184.4 / 353.7 | `e909e028` |
| BN = 32 | blocks of 32 keys | 5 | | 207.9 / 401.6 | `3882cd18` |
| BN = 128 | blocks of 128 keys (two n64 wgmma) | 3 | | 179 / 331 | `02748ef7` |
| `EXPH` + `DEFERL` | exponentials in FP16 (`ex2.approx.f16x2`) | 4 | | 172.2 / 325.4 | `9d292aee` |
| `CLW` | cluster of 2 that shares each 128 B row through DSMEM | 4 | **−30%** | **200.7 / 384.2** | `127933f5` |

## Decode full (batch 16 / 32)

Reference: `full_fa2`, 3 stages, 3 CTAs/SM: **106-110 / 196-203 µs** (run-to-run variation is ±3%).

| idea | what it changed | bench | H200 decode 16 / 32 (µs) | check |
|---|---|---|---|---|
| 5 stages, 2 CTAs/SM | more blocks in flight | | 115.0 / 215.6 | `429178c1` |
| 2 stages, 4 CTAs/SM | more CTAs, less pipeline | = | 109.0 / 203.1 (tie) | `d1ea671a` |
| `KVP = 1`, 5 CTAs/SM | half a 128 B row per CTA | +55% | **154.7 / 295.5** | `9e8c4012` |

## Sliding and latency

| idea | what it changed | bench | H200 | check |
|---|---|---|---|---|
| `cl` sized to one wave | sliding decode at batch 16 / 32 in one kernel (cluster of 8 / 4) | +2-7% | 12.9 / 18.4 µs, against 12.4 / 16.9 | `6256b262` |
| `KVEARLY` | K/V copies before the q chain | −10% (decode 1 sliding) | total 1.697x, within the noise | `aa9dd3ea` |
| `KVEARLY` + `EPIW` | plus the epilogue with the weights once per row | | total 1.697x, within the noise | `22bb24e8` |
| `GDCQ` (in the layer) | PDL behind the qkv projection | = | layer 1.667x, against 1.685x | `3bdcac84` (decode_layer) |
| `LASTM` | no cluster: the last CTA merges | +17% (11.9 against 10.2 µs) | not measured | — |

`KVEARLY` and `EPIW` ended up in the mark (`f31f6572`) together with `cl` with one head per CTA and
the single-pass combine. Alone they stayed within the noise.

## The lesson

On the bench (HBM2e, ~1.5 TB/s), verify full is limited by DRAM efficiency with random rows:
anything that reads whole 128 B rows wins. On the H200 (4.8 TB/s) what rules is **the CTAs per SM
and the length of the dependency chain of each block**. `KVP = 2`, `PVA` and `CLW` won on the bench
and lost on the H200.

| change | decided on the bench? |
|---|---|
| latency: prologue, combine, launches | yes (decode 1 sliding with `cl` and one head per CTA; combine) |
| CTAs/SM, MMA path, row split, gather shape | no: it needs a check on the H200 |

The other trap is ptxas: it serializes **every** `wgmma` in the kernel (warnings C7514 / C7515) if
an ordinary instruction reads or defines an accumulator while a `wgmma` is in flight, even on a path
that never runs at the same time. Look for those warnings when you assemble each variant with the
judge's ptxas, before measuring it.
