"""Breakdown of a judge run per load and per path, and scenarios on the geometric mean.
Usage: python3 breakdown.py <run.json>"""
import json
import math
import sys

d = json.load(open(sys.argv[1]))
r = d["result"]
bench = [x for x in r["records"] if x["event"] == "benchmark"]
loads = {}
for x in bench:
    c = x["case"]
    loads.setdefault(c["load"], {})[c["layer"]] = dict(w=c["weight"], base=x["baseline_ns"] / 1e3, mine=x["median_ns"] / 1e3,
                                                       ratio=x["speedup"])


def load_speedup(l, mult=None):
    mult = mult or {}
    num = sum(v["w"] * v["base"] for v in l.values())
    den = sum(v["w"] * v["mine"] * mult.get(k, 1.0) for k, v in l.items())
    return num / den


def geo(mults):
    """mults: {(phase, layer): factor on my time}."""
    s = []
    for name, l in loads.items():
        phase = name.split()[0]
        s.append(load_speedup(l, {layer: mults.get((phase, layer), 1.0) for layer in l}))
    return math.exp(sum(math.log(x) for x in s) / len(s)), s


print(f"reported {r['speedup']:.4f}  CI95 {r['speedup_ci']}")
g0, s0 = geo({})
print(f"reproduced (sum w base / sum w mine per load, geometric mean) {g0:.4f}\n")
print(f"{'load':10s} {'layer':8s} {'wgt':>4s} {'SGLang µs':>10s} {'mine µs':>10s} {'x':>6s} {'% of load':>10s}")
for (name, l), sp in zip(loads.items(), s0):
    tot = sum(v["w"] * v["mine"] for v in l.values())
    for layer, v in l.items():
        print(f"{name:10s} {layer:8s} {v['w']:4d} {v['base']:10.2f} {v['mine']:10.2f} {v['base'] / v['mine']:6.3f} "
              f"{100 * v['w'] * v['mine'] / tot:9.1f}%")
    print(f"{name:10s} {'load':8s} {'':4s} {'':10s} {'':10s} {sp:6.3f}")
print()
for label, m in (("(a) verify full x0.7", {("verify", "full"): 0.7}),
                 ("(b) sliding decode -20 %", {("decode", "sliding"): 0.8}),
                 ("    sliding verify -20 %", {("verify", "sliding"): 0.8}),
                 ("    sliding (all) -20 %", {("decode", "sliding"): 0.8, ("verify", "sliding"): 0.8}),
                 ("    full decode -10 %", {("decode", "full"): 0.9}),
                 ("    verify full at decode full's rate", None)):
    if m is None:
        # verify full at decode full's µs per shape (same bytes): factor per load
        f = {}
        for n in ("16", "32", "1"):
            f[n] = loads[f"decode {n}"]["full"]["mine"] / loads[f"verify {n}"]["full"]["mine"]
        s = []
        for name, l in loads.items():
            phase, n = name.split()
            s.append(load_speedup(l, {"full": f[n]} if phase == "verify" else {}))
        g = math.exp(sum(math.log(x) for x in s) / len(s))
        print(f"{label:40s} {g:.4f}  ({g - g0:+.4f})  factors {', '.join(f'{k}: {v:.2f}' for k, v in f.items())}")
        continue
    g, _ = geo(m)
    print(f"{label:40s} {g:.4f}  ({g - g0:+.4f})")
