"""Time breakdown of the layer per shape and per component, from an okc check's H200 profile.

    python3 kernels/muse_decode_layer/tools/breakdown.py <job>

Reads evidence/runs/muse_decode_layer/<job>.{json,txt}; run it from the repository root.

Each okc profile lists a shape's 12 heaviest kernels and the GPU total; the remainder goes to
"other". GPU time adds both streams (router and routing overlap the pooler), so it exceeds the
wall time; shares within a shape are taken on GPU time and applied to the measured wall time.
Prints markdown tables: per shape and component (us), the share of each component in the score
(each load counts the same in the geometric mean) and the floors from the TASK.md tables.
"""
import json
import math
import re
import sys

job = sys.argv[1]
base = f"evidence/runs/muse_decode_layer/{job}"
txt = open(base + ".txt").read()
res = json.load(open(base + ".json"))
res = res.get("result", res)

COMP = [  # (component, regex on the kernel name)
    ("attention", r"_attn_kernel|_combine_kernel|full_fa2|verify_ws|verify_wg|flash"),
    ("MoE", r"moe_gemm|moe_route|_route_prep|_swiglu_reduce|^_combine$|marlin"),
    ("proj. c1", r"_c1_"),
    ("proj. split-K", r"_proj_kernel|_splitk_finish"),
    ("proj. cuBLAS", r"nvjet|cublas|gemm|xmma"),
    ("routing + pre-expert", r"_route_pre_expert"),
    ("boundaries (norm + residual)", r"norm_residual|_ffn_boundary"),
]

# Floors, H200, from the TASK.md tables of the single-op problems (us).
PROJ_FLOOR = {1: 28.9, 16: 41.5, 32: 41.7, 8: 29.0, 128: 43.8, 256: 51.8}  # qkv+o+pooler+unpooler+router
MOE_FLOOR = {1: 11.4, 16: 52.5, 32: 70.8, 8: 37.3, 128: 101.3, 256: 108.9}
ATT_FLOOR = {("decode 1", "sliding"): 0.1, ("decode 1", "full"): 4.3, ("decode 16", "sliding"): 1.6,
             ("decode 16", "full"): 63.2, ("decode 32", "sliding"): 3.3, ("decode 32", "full"): 126.4,
             ("verify 1", "sliding"): 0.1, ("verify 1", "full"): 4.3, ("verify 16", "sliding"): 1.9,
             ("verify 16", "full"): 63.4, ("verify 32", "sliding"): 3.7, ("verify 32", "full"): 126.8}
TOKENS = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}

bench = {}
for x in res["records"]:
    if x.get("event") == "benchmark":
        c = x["case"]
        bench[(c["load"], c["layer"])] = (x["baseline_ns"] / 1000, x["median_ns"] / 1000, c["weight"])

shapes = {}
for blk in re.split(r"\n(?=profile \d+:)", txt)[1:]:
    head = blk.splitlines()[0]
    load = re.search(r"load=(\w+ \d+)", head).group(1)
    layer = re.search(r"layer=(\w+)", head).group(1)
    comps = {name: 0.0 for name, _ in COMP}
    listed = 0.0
    total = float(re.search(r"total\s+([\d.]+) µs", blk).group(1))
    for line in blk.splitlines()[2:]:
        m = re.match(r"(\S.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)$", line)
        if not m:
            continue
        name, t = m.group(1), float(m.group(2))
        listed += t
        for comp, rx in COMP:
            if re.search(rx, name):
                comps[comp] += t
                break
        else:
            comps.setdefault("other", 0.0)
            comps["other"] += t
    comps["other (outside the top 12)"] = total - listed
    shapes[(load, layer)] = (total, comps)

names = [c for c, _ in COMP] + ["other (outside the top 12)"]
loads = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]

print("### Per shape (wall-clock µs, split by each component's GPU time)\n")
print("| shape | SGLang | mine | speedup | layer floor | " + " | ".join(names) + " |")
print("|---|" + "---|" * (4 + len(names)))
load_tot = {}
load_comp = {}
for load in loads:
    for layer in ("sliding", "full"):
        sg, mine, w = bench[(load, layer)]
        gpu, comps = shapes[(load, layer)]
        scale = mine / gpu
        vals = [comps.get(n, 0.0) * scale for n in names]
        lt = load_tot.setdefault(load, [0.0, 0.0])
        lt[0] += w * sg
        lt[1] += w * mine
        lc = load_comp.setdefault(load, {n: 0.0 for n in names})
        for n, v in zip(names, vals):
            lc[n] += w * v
        print(f"| {load} {layer} | {sg:.1f} | {mine:.1f} | {sg / mine:.3f} | | "
              + " | ".join(f"{v:.1f}" for v in vals) + " |")

print("\n### Per load: each component's share of the load's time, and the floors\n")
print("| load | speedup | " + " | ".join(names) + " | proj. floor | MoE floor | attention floor |")
print("|---|---|" + "---|" * (len(names) + 3))
share_avg = {n: 0.0 for n in names}
for load in loads:
    sg, mine = load_tot[load]
    t = TOKENS[load]
    sh = {n: load_comp[load][n] / mine for n in names}
    for n in names:
        share_avg[n] += sh[n] / len(loads)
    att_floor = (46 * ATT_FLOOR[(load, "sliding")] + 16 * ATT_FLOOR[(load, "full")]) / 62
    print(f"| {load} | {sg / mine:.3f} | " + " | ".join(f"{100 * sh[n]:.0f} %" for n in names)
          + f" | {PROJ_FLOOR[t]:.1f} | {MOE_FLOOR[t]:.1f} | {att_floor:.1f} (weighted mean) |")
g = math.exp(sum(math.log(load_tot[l][0] / load_tot[l][1]) for l in loads) / len(loads))
print(f"\nGeometric mean of this check: {g:.4f}x.\n")
print("### Mean share of the score (mean over the six loads, each counting the same)\n")
print("| component | mean share | gain if it were halved |")
print("|---|---|---|")
for n in names:
    half = math.exp(-sum(math.log(1 - 0.5 * load_comp[l][n] / load_tot[l][1]) for l in loads) / len(loads))
    print(f"| {n} | {100 * share_avg[n]:.1f} % | x{half:.3f} |")


def scenario(target):
    """Geometric-mean speedup if each component in target takes target[c](load, layer) us, the
    rest staying as measured."""
    out = 0.0
    for load in loads:
        sg_tot, ours_tot = 0.0, 0.0
        for layer in ("sliding", "full"):
            sg, mine, w = bench[(load, layer)]
            gpu, comps = shapes[(load, layer)]
            scale = mine / gpu
            t = mine
            for comp, f in target.items():
                cur = sum(comps.get(n, 0.0) for n in comp) * scale
                t += min(f(load, layer), cur) - cur
            sg_tot += w * sg
            ours_tot += w * t
        out += math.log(sg_tot / ours_tot)
    return math.exp(out / len(loads))


PROJ = ("proj. c1", "proj. split-K", "proj. cuBLAS")
moe = lambda k: {("MoE",): lambda l, y: k * MOE_FLOOR[TOKENS[l]]}  # noqa: E731
att = lambda k: {("attention",): lambda l, y: max(k * ATT_FLOOR[(l, y)], 5.0)}  # noqa: E731
proj = lambda k: {PROJ: lambda l, y: k * PROJ_FLOOR[TOKENS[l]]}  # noqa: E731
small = {("routing + pre-expert", "boundaries (norm + residual)", "other (outside the top 12)"): lambda l, y: 4.0}
print("\n### Scenarios: geometric mean if each part reached k times its floor (the rest as measured)\n")
print("| scenario | geometric mean | vs this check |")
print("|---|---|---|")
for label, tgt in (("measured", {}),
                   ("MoE at 2 x floor", moe(2)), ("MoE at 1.5 x floor", moe(1.5)), ("MoE at the floor", moe(1)),
                   ("attention at 2 x floor (min. 5 µs)", att(2)), ("attention at the floor (min. 5 µs)", att(1)),
                   ("projections at 1.3 x floor", proj(1.3)), ("projections at the floor", proj(1)),
                   ("small kernels at 4 µs per shape", small),
                   ("MoE 2x + attention 2x + proj. 1.3x", {**moe(2), **att(2), **proj(1.3)}),
                   ("MoE 1.5x + attention 2x + proj. 1.3x + small 4 µs", {**moe(1.5), **att(2), **proj(1.3), **small}),
                   ("everything at the floor", {**moe(1), **att(1), **proj(1), **small})):
    v = scenario(tgt)
    print(f"| {label} | {v:.3f}x | x{v / g:.3f} |")
