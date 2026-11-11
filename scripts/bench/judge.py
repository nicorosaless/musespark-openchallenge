"""Local judge: runs a submission.py against the SGLang baseline the way okc does.

Runs inside the judge's image (SIF) on an H100 of the university cluster. It does not replace
`okc check`: it is the cheap iteration tool between checks.

What it imitates (TASK.md, "How a run is timed"):
- the pre-flight tests listed in TASK.md and correctness with reference.check_implementation;
- each scored shape captured as a CUDA graph and replayed on fresh inputs
  (a set of seeds that is copied into the static buffers before each replay);
- cold L2: a buffer larger than the L2 is read before each timed replay;
- up to 16 input copies per shape and the judge's repetitions per side; baseline and submission
  alternate and the order flips every repetition; one ratio per repetition;
- speedup per shape = median of ratios; per load = Σ weight·t_base / Σ weight·t_sub;
  per problem = geometric mean over loads; 95% CI by bootstrap, at least ±1.5 %;
- the fixed cost of a replay is subtracted, 2·t(1-node graph) − t(2-node graph).

Known differences from the judge: its own seeds; muse_decode_layer has no `router_load`
(it lives in the judge's task.yml), so the default router is used; the GPU is an H100,
not an H200 (see evidence/calibration.json).

Usage:
  python judge.py --problem DIR --submission FILE --output OUT.json
                  [--baseline FILE] [--no-tests] [--no-timing] [--reps N] [--copies N]
                  [--filter TEXT] [--no-profile] [--keep-going]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import sys
import time
import traceback

T0 = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

LIMITS_S = {"muse_bf16_proj": 600}  # every other problem: 900 (TASK.md: "A run has N minutes in all")
# Repetitions per side the judge used in the calibration checks (records[].runs); 60 by default.
REPS = {"muse_moe_w4a16": 100, "muse_prefill_layer": 6}
TEST_SEED = 1000
SHAPE_SEED = 2000
FLUSH_BYTES = 512 << 20  # the H100 and the H200 both have 50 MB of L2
MIN_CI = 0.015
COPY_BUDGET_BYTES = 2 << 30


# ---------------------------------------------------------------- TASK.md

def _value(v: str):
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def parse_shape(text: str) -> dict:
    """"batch 1, ctx 80000, layer sliding, page 1" -> {"batch": 1, ...}."""
    shape = {}
    for part in text.split(","):
        key, _, value = part.strip().partition(" ")
        shape[key] = _value(value.strip())
    return shape


def read_task(path: str) -> tuple[list[dict], list[dict]]:
    """(scored shapes, tests) from the tables in TASK.md."""
    text = open(path).read()
    shapes = []
    for row in re.findall(r"^\| *((?:decode|verify|turn|first)[^|]*)\|([^|]+)\|([^|]+)\|([^|]+)\|", text, re.M):
        load, shape, weight, floor = (c.strip() for c in row)
        floor_us = float(floor.replace("µs", "").replace(",", "").strip())
        shapes.append({"load": load, "shape": shape, "weight": int(weight), "floor_us": floor_us})
    block = re.search(r"must pass these \d+ tests[^`]*```\n(.*?)```", text, re.S)
    tests = [{"shape": line.strip()} for line in block.group(1).splitlines() if line.strip()]
    return shapes, tests


# ---------------------------------------------------------------- modules

def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone(v) for v in x)
    return copy.deepcopy(x)


def copy_into(dst, src):
    """Copy the tensors of src into those of dst (same shapes), without allocating."""
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v, non_blocking=True)
        elif isinstance(v, dict) and k != "config":
            copy_into(dst[k], v)


def nbytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    return 0


def n_params(fn) -> int:
    return len(inspect.signature(fn).parameters)


class Problem:
    def __init__(self, directory: str):
        sys.path.insert(0, directory)
        self.dir = directory
        # The name comes from the TASK.md header ("# ... (`muse_x`)"): on the worker the
        # directory is called `problem/`.
        header = open(os.path.join(directory, "TASK.md")).readline()
        m = re.search(r"\(`(muse_\w+)`\)", header)
        self.name = m.group(1) if m else os.path.basename(os.path.normpath(directory))
        self.ref = load_module("reference", os.path.join(directory, "reference.py"))
        self.shapes, self.tests = read_task(os.path.join(directory, "TASK.md"))
        self.weights = None
        if hasattr(self.ref, "generate_weights"):
            self.weights = self.ref.generate_weights(self.ref.WEIGHTS_SEED)
        self.check_takes_weights = n_params(self.ref.check_implementation) == 3

    def make_input(self, shape: dict, seed: int) -> dict:
        case = {**shape, "seed": seed}
        if hasattr(self.ref, "case_fields"):
            case.update(self.ref.case_fields(case, self.weights))
        return clone(self.ref.generate_input(**case))

    def check(self, data, out) -> tuple[bool, str]:
        raw = self.ref.collect(data, out) if hasattr(self.ref, "collect") else out
        args = (data, raw, self.weights) if self.check_takes_weights else (data, raw)
        ok, msg = self.ref.check_implementation(*args)
        return bool(ok), str(msg)


class Kernel:
    """A prepared submission.py or baseline.py: kernel_fn(data[, state])."""

    def __init__(self, name: str, path: str, problem: Problem):
        self.path = path
        self.mod = load_module(name, path)
        self.state = None
        self.prepare_s = 0.0
        self.takes_state = n_params(self.mod.kernel_fn) == 2
        if hasattr(self.mod, "prepare"):
            t = time.time()
            self.state = self.mod.prepare(clone(problem.weights))
            torch.cuda.synchronize()
            self.prepare_s = time.time() - t
        elif self.takes_state:
            self.state = clone(problem.weights)

    def __call__(self, data):
        return self.mod.kernel_fn(data, self.state) if self.takes_state else self.mod.kernel_fn(data)


# ---------------------------------------------------------------- timing

def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


class Timer:
    def __init__(self):
        self.buf = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda").fill_(1.0)
        self.acc = torch.empty((), dtype=torch.float32, device="cuda")
        # Fixed cost of a replay, as the judge computes it: 2·t(1 node) − t(2 nodes).
        trivial = torch.zeros(1, device="cuda")
        g1, _ = capture(lambda: trivial.add_(1))
        g2, _ = capture(lambda: (trivial.add_(1), trivial.add_(1)))
        one = float(np.median(self.measure(lambda: None, g1, 200)))
        two = float(np.median(self.measure(lambda: None, g2, 200)))
        self.graph_overhead_us = max(0.0, 2 * one - two)

    def flush_l2(self):
        torch.sum(self.buf, dim=0, out=self.acc)

    def measure(self, load_input, g, n) -> list[float]:
        evs = []
        for _ in range(n):
            load_input()
            self.flush_l2()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            evs.append((a, b))
        torch.cuda.synchronize()
        return [1000.0 * a.elapsed_time(b) for a, b in evs]


def bootstrap(per_shape, shapes, rng, n=1000):
    """95% CI of the problem speedup, resampling the repetitions of each shape."""
    samples = []
    for _ in range(n):
        speeds = []
        for f, timed in zip(shapes, per_shape):
            r = timed["ratios"]
            speeds.append(float(np.median(r[rng.integers(0, len(r), len(r))])))
        samples.append(problem_speedup(shapes, speeds)[0])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def problem_speedup(shapes, speeds):
    """(speedup, loads) with the aggregation from TASK.md."""
    loads = {}
    for f, s in zip(shapes, speeds):
        d = loads.setdefault(f["load"], {"base": 0.0, "sub": 0.0})
        d["base"] += f["weight"] * f["base_us"]
        d["sub"] += f["weight"] * f["base_us"] / s
    sp = [d["base"] / d["sub"] for d in loads.values()]
    return float(math.exp(sum(map(math.log, sp)) / len(sp))), loads


def kernel_profile(g, load_input, timer, reps=3) -> str:
    """Kernels of one replay with a cold L2: only the replay is inside the profile."""
    from torch.profiler import ProfilerActivity, profile
    times, calls = {}, {}
    for _ in range(reps):
        load_input()
        timer.flush_l2()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            g.replay()
            torch.cuda.synchronize()
        for ev in prof.key_averages():
            t = getattr(ev, "self_device_time_total", None)
            t = getattr(ev, "self_cuda_time_total", 0) if t is None else t
            if t > 0:
                times[ev.key] = times.get(ev.key, 0.0) + t / reps
                calls[ev.key] = calls.get(ev.key, 0) + ev.count
    total = sum(times.values()) or 1.0
    rows = sorted(((t, calls[k] // reps, k) for k, t in times.items()), reverse=True)
    return "\n".join(f"{t:9.2f} µs/rep {100 * t / total:5.1f}%  x{n:<3d} {k[:110]}" for t, n, k in rows[:15])


def time_shape(problem, base, sub, f, reps, copies, timer, with_profile):
    shape = parse_shape(f["shape"])
    pool = [problem.make_input(shape, SHAPE_SEED)]
    # Like the judge (16 copies in bf16_proj, 2 in decode_layer): as many as fit in the budget.
    copies = max(2, min(copies, COPY_BUDGET_BYTES // max(1, nbytes(pool[0]))))
    pool += [problem.make_input(shape, SHAPE_SEED + 17 * i) for i in range(1, copies)]
    static = clone(pool[0])
    g_base, _ = capture(lambda: base(static))
    g_sub, out_sub = capture(lambda: sub(static))
    # Every timed call is also checked: here, one replay per input.
    checks = []
    for p in pool:
        copy_into(static, p)
        g_sub.replay()
        torch.cuda.synchronize()
        checks.append(problem.check(static, out_sub))
    tb, ts = [], []
    for r in range(reps):
        p = pool[r % len(pool)]
        load_input = lambda p=p: copy_into(static, p)  # noqa: E731
        order = [(g_base, tb), (g_sub, ts)] if r % 2 == 0 else [(g_sub, ts), (g_base, tb)]
        for g, dest in order:
            dest.extend(timer.measure(load_input, g, 1))
    c = timer.graph_overhead_us
    tb = np.maximum(np.array(tb) - c, 0.05)
    ts = np.maximum(np.array(ts) - c, 0.05)
    # One ratio per repetition (the order alternates between repetitions), like the judge.
    ratios = tb / ts
    speed = float(np.median(ratios))
    base_us = float(np.median(tb))
    res = {
        **f, "base_us": base_us, "sub_us": base_us / speed, "speedup": speed,
        "ratios": ratios, "reps": int(len(ratios)), "copies": len(pool),
        "ratio_iqr": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
        "check_ok": all(ok for ok, _ in checks), "check_msg": next((m for ok, m in checks if not ok), checks[0][1]),
    }
    if with_profile:
        res["profile"] = kernel_profile(g_sub, lambda: copy_into(static, pool[0]), timer)
    del g_base, g_sub, static, pool
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--filter", default="")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    res = {"judge_version": 1, "arguments": vars(args), "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def save():
        res["duration_s"] = round(time.time() - T0, 1)
        lim = res.get("limit_s")
        res["within_limit"] = lim is None or res["duration_s"] <= lim
        with open(args.output, "w") as fh:
            json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    try:
        sub_src = open(args.submission, "rb").read()
        res["submission_sha256"] = hashlib.sha256(sub_src).hexdigest()
        res["environment"] = {
            "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "driver": _driver(), "python": platform.python_version(),
        }
        problem = Problem(args.problem)
        res["problem"] = problem.name
        res["limit_s"] = LIMITS_S.get(problem.name, 900)
        base_path = args.baseline or os.path.join(args.problem, "baseline.py")
        sub = Kernel("okc_submission", args.submission, problem)
        res["prepare_s"] = {"sub": round(sub.prepare_s, 2)}

        res["tests"] = []
        if not args.no_tests:
            for i, t in enumerate(problem.tests):
                t0 = time.time()
                try:
                    data = problem.make_input(parse_shape(t["shape"]), TEST_SEED + i)
                    out = sub(data)
                    torch.cuda.synchronize()
                    ok, msg = problem.check(data, out)
                    del data, out
                except Exception as e:  # noqa: BLE001
                    ok, msg = False, f"{type(e).__name__}: {e}"
                    res.setdefault("tracebacks", []).append(traceback.format_exc())
                res["tests"].append({"shape": t["shape"], "ok": ok, "msg": msg, "s": round(time.time() - t0, 2)})
                print(f"[test] {'OK ' if ok else 'FAIL'} {t['shape']}: {msg}", flush=True)
                if not ok and not args.keep_going:
                    break
            torch.cuda.empty_cache()
        res["tests_ok"] = all(t["ok"] for t in res["tests"]) and (
            args.no_tests or len(res["tests"]) == len(problem.tests))
        save()

        if args.no_timing or not res["tests_ok"]:
            res["status"] = "tests" if res["tests_ok"] else "tests_failed"
            return save()

        base = Kernel("okc_baseline", base_path, problem)
        res["prepare_s"]["base"] = round(base.prepare_s, 2)
        timer = Timer()
        res["graph_overhead_us"] = timer.graph_overhead_us
        rng = np.random.default_rng(0)
        shapes = [f for f in problem.shapes if args.filter in f["load"] + " | " + f["shape"]]
        res["shapes"] = []
        for f in shapes:
            reps = args.reps or REPS.get(problem.name, 60)
            r = time_shape(problem, base, sub, f, reps, args.copies, timer, not args.no_profile)
            res["shapes"].append(r)
            print(f"[shape] {f['load']} | {f['shape']}: base {r['base_us']:.1f} µs, sub {r['sub_us']:.1f} µs,"
                  f" x{r['speedup']:.3f} check={'ok' if r['check_ok'] else 'FAIL'}", flush=True)
            save()
        sp, loads = problem_speedup(res["shapes"], [r["speedup"] for r in res["shapes"]])
        lo, hi = bootstrap(res["shapes"], res["shapes"], rng)
        lo, hi = min(lo, sp * (1 - MIN_CI)), max(hi, sp * (1 + MIN_CI))
        bases = [d["base"] for d in loads.values()]
        step_base = math.exp(sum(map(math.log, bases)) / len(bases))
        res.update({
            "speedup": sp, "ci95": [lo, hi],
            "loads": [{"load": k, "base_us": d["base"], "sub_us": d["sub"], "speedup": d["base"] / d["sub"]}
                      for k, d in loads.items()],
            "step_base_us": step_base, "step_sub_us": step_base / sp,
            "timing_checks_ok": all(r["check_ok"] for r in res["shapes"]),
            "complete": len(shapes) == len(problem.shapes),
        })
        for r in res["shapes"]:
            r.pop("ratios")
        res["status"] = "ok" if res["timing_checks_ok"] else "timing_checks_failed"
        print(f"[total] speedup x{sp:.3f} CI95 [{lo:.3f}, {hi:.3f}]", flush=True)
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        res.setdefault("tracebacks", []).append(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    for r in res.get("shapes", []):
        r.pop("ratios", None)
    save()


def _driver() -> str:
    try:
        import pynvml
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else v
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
