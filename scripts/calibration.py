#!/usr/bin/python3 -I
"""Calibration of the H100 (bench) against the H200 (judge) with the untouched SGLang baseline.

  scripts/calibration.py archive <okc-status.json>...     copy JSON, profile and log to evidence/runs/
  scripts/calibration.py match <okc-status.json>... --bench <bench.json>...

`match` pairs shapes by (load, shape fields) and writes evidence/calibration.json: per
shape, the SGLang time on the H200 (okc), on the H100 (bench) and factor = h200 / h100.
A bench time multiplied by the factor estimates the judge's.
"""

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORED = {"seed", "weight", "load", "repeats"}


def shape_key(load, fields):
    return load + " | " + ", ".join(f"{k} {fields[k]}" for k in sorted(fields) if k not in IGNORED)


def parse_shape(text):
    fields = {}
    for part in text.split(","):
        k, _, v = part.strip().partition(" ")
        fields[k] = v.strip()
    return fields


def archive(paths):
    for p in map(Path, paths):
        r = json.loads(p.read_text())
        dest = ROOT / "evidence/runs" / r["board"]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest / f"{r['id']}.json")
        for kind, ext in (("text", "txt"), ("log", "log")):
            source = (r.get("profile") or {}).get(kind) if kind == "text" else (r.get("output") or {}).get("log")
            if source and Path(source).is_file():
                shutil.copy(source, dest / f"{r['id']}.{ext}")
        print(f"{r['board']}/{r['id']}: {r['result'].get('status')} speedup {r['result'].get('speedup')}")


def match(okc_paths, bench_paths):
    output = ROOT / "evidence/calibration.json"
    cal = json.loads(output.read_text()) if output.exists() else {}
    benches = {}
    for p in map(Path, bench_paths):
        b = json.loads(p.read_text())
        benches[(b.get("meta") or {}).get("problem") or b["problem"]] = (p, b)
    for p in map(Path, okc_paths):
        r = json.loads(p.read_text())
        problem = r["board"]
        if problem not in benches:
            print(f"{problem}: no bench run, skipping it")
            continue
        bpath, b = benches[problem]
        h100 = {shape_key(f["load"], parse_shape(f["shape"])): f for f in b["shapes"]}
        shapes = []
        for rec in r["result"]["records"]:
            if rec.get("event") != "benchmark":
                continue
            case = {k: str(v) for k, v in rec["case"].items()}
            k = shape_key(case["load"], case)
            f = h100.get(k)
            h200_us = rec["baseline_ns"] / 1000
            shapes.append({
                "shape": k, "weight": int(rec["case"].get("weight", 1)),
                "sglang_h200_us": round(h200_us, 3),
                "sglang_h100_us": round(f["base_us"], 3) if f else None,
                "factor_h200_h100": round(h200_us / f["base_us"], 4) if f else None,
            })
        matched = [f for f in shapes if f["factor_h200_h100"]]
        logs = [math.log(f["factor_h200_h100"]) for f in matched]
        cal[problem] = {
            "okc_job": r["id"],
            "bench_run": str(bpath.relative_to(ROOT)) if bpath.is_relative_to(ROOT) else str(bpath),
            "judge_gpu": r["result"].get("gpu"), "bench_gpu": b["environment"]["gpu"],
            "bench_driver": b["environment"]["driver"],
            "matched_shapes": f"{len(matched)}/{len(shapes)}",
            "factor_geomean": round(math.exp(sum(logs) / len(logs)), 4) if logs else None,
            "factor_min": min((f["factor_h200_h100"] for f in matched), default=None),
            "factor_max": max((f["factor_h200_h100"] for f in matched), default=None),
            "speedup_okc_baseline": r["result"].get("speedup"),
            "speedup_bench_baseline": b.get("speedup"),
            "shapes": shapes,
        }
        print(f"{problem}: {len(matched)}/{len(shapes)} shapes, geomean factor {cal[problem]['factor_geomean']}")
    output.write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    print(output)


def main(argv):
    if argv[:1] == ["archive"]:
        return archive(argv[1:])
    if argv[:1] == ["match"] and "--bench" in argv:
        i = argv.index("--bench")
        return match(argv[1:i], argv[i + 1:])
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
