#!/usr/bin/env python3
"""Portfolio charts for the Open Kernel Challenge (entry nicorosaless).

Usage: python3 charts.py <repo_root>

Reads only <repo_root>/evidence/ and <repo_root>/kernels/muse_decode_layer/time-breakdown.md.
Writes <repo_root>/charts/<name>.png and <name>.svg. Matplotlib only, no network,
byte-stable output for the same input.
"""
import csv
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Patch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator  # noqa: E402

TEAM = "nicorosaless"
PROBLEMS = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]
FINAL_MARKS = {
    "muse_bf16_proj": "b34454802aa64dc4",
    "muse_attention_decode": "28dd5b809b434132",
    "muse_moe_w4a16": "db79106e9d524b1d",
    "muse_decode_layer": "b76b7763d6eb45b6",
}
CET = dt.timezone(dt.timedelta(hours=1))

# Palette: reference instance of the dataviz skill (light mode), validated with
# validate_palette.js: categorical slots 1-4 (adjacent), slots 1-2 (all pairs), ordinal blue 250/550.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
BLUE_LIGHT = "#86b6ef"  # ordinal step 250
BLUE_DARK = "#1c5cab"  # ordinal step 550
PX = 1 / 96  # one CSS pixel in inches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.facecolor": SURFACE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "svg.hashsalt": "okc-charts",
    "path.simplify": False,
})
DPI = 200


# ---------------------------------------------------------------- helpers

def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=DPI, facecolor=SURFACE,
                metadata={"Software": None})
    fig.savefig(os.path.join(outdir, name + ".svg"), facecolor=SURFACE,
                metadata={"Date": None, "Creator": None})
    plt.close(fig)
    print("wrote", os.path.join(outdir, name) + ".{png,svg}")


def px_to_data(ax, px):
    """Convert CSS pixels to data units along x and y for a linear axis."""
    fig = ax.figure
    bbox = ax.get_position()
    w_in = bbox.width * fig.get_figwidth()
    h_in = bbox.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return px * PX * abs(x1 - x0) / w_in, px * PX * abs(y1 - y0) / h_in


def hbar(ax, y, x0, x1, thick, color, rounded=True, radius_px=4, zorder=2):
    """Horizontal bar from x0 to x1, square at x0, 4px rounded corners at x1.

    Axis limits must be final before calling (radius is converted from pixels).
    """
    if x1 <= x0:
        return
    rx, ry = px_to_data(ax, radius_px)
    if ax.yaxis_inverted():
        ry = abs(ry)
    rx = min(rx, (x1 - x0)) if rounded else 0
    ry = min(ry, thick / 2) if rounded else 0
    yb, yt = y - thick / 2, y + thick / 2
    verts = [(x0, yb), (x1 - rx, yb), (x1, yb), (x1, yb + ry), (x1, yt - ry), (x1, yt),
             (x1 - rx, yt), (x0, yt), (x0, yb)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def vbar(ax, x, y1, thick, color, radius_px=4, zorder=2):
    """Vertical bar from 0 to y1, square at the baseline, rounded at the top."""
    if y1 <= 0:
        return
    rx, ry = px_to_data(ax, radius_px)
    rx = min(rx, thick / 2)
    ry = min(ry, y1)
    xl, xr = x - thick / 2, x + thick / 2
    verts = [(xl, 0), (xr, 0), (xr, y1 - ry), (xr, y1), (xr - rx, y1), (xl + rx, y1), (xl, y1),
             (xl, y1 - ry), (xl, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def header(fig, title, subtitle, top=0.975):
    fig.text(0.012, top, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, top - 0.032 * (8 / fig.get_figheight()), subtitle, fontsize=9, color=INK2,
             va="top")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- data: runs

RUN_FILE = re.compile(r"^[0-9a-f]{16}\.json$")


def load_runs(ev, problem):
    runs = []
    for path in sorted(glob.glob(os.path.join(ev, "runs", problem, "*.json"))):
        if RUN_FILE.match(os.path.basename(path)):  # skips <id>.status.json sidecars
            runs.append(read_json(path))
    return runs


def case_key(case):
    return tuple(sorted((k, v) for k, v in case.items() if k != "seed"))


def per_shape(run):
    out = OrderedDict()
    for rec in run["result"]["records"]:
        if rec.get("event") == "benchmark":
            out[case_key(rec["case"])] = rec
    return out


def shape_label(problem, case):
    c = dict(case)
    if problem == "muse_bf16_proj":
        return f"{c['layer']} · {c['load']} (m={c['m']})"
    if problem == "muse_moe_w4a16":
        return f"{c['load']} (bs {c['bs']})"
    return f"{c['load']} · {c['layer']}"


def shape_order(problem, keys):
    loads = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
    if problem == "muse_bf16_proj":
        layers = ["qkv_gate", "o", "pooler", "unpooler", "router"]
        return sorted(keys, key=lambda k: (layers.index(dict(k)["layer"]), loads.index(dict(k)["load"])))
    if problem == "muse_moe_w4a16":
        return sorted(keys, key=lambda k: loads.index(dict(k)["load"]))
    return sorted(keys, key=lambda k: (loads.index(dict(k)["load"]),
                                       ["sliding", "full"].index(dict(k)["layer"])))


# ---------------------------------------------------------------- chart 1

def chart_speedup_by_shape(ev, outdir):
    data = {}
    for p in PROBLEMS:
        runs = load_runs(ev, p)
        submits = sorted((r for r in runs if r.get("mode") == "run"
                          and (r.get("result") or {}).get("status") == "ok"),
                         key=lambda r: r["created_at"])
        first = submits[0]
        final = next(r for r in runs if r["id"] == FINAL_MARKS[p])
        f_sh, m_sh = per_shape(first), per_shape(final)
        keys = shape_order(p, [k for k in m_sh if k in f_sh])
        data[p] = dict(first=first, final=final, keys=keys,
                       rows=[(shape_label(p, k), f_sh[k]["speedup"], m_sh[k]["speedup"]) for k in keys])

    fig = plt.figure(figsize=(13.5, 12.6))
    header(fig, "Speedup vs SGLang per scored shape, first submit to final mark",
           "H200 judge, per-shape speedup from the submit records (baseline_ns / median_ns). "
           "1x = SGLang, right is faster. Log scale.")
    gs = fig.add_gridspec(1, 2, left=0.17, right=0.985, top=0.885, bottom=0.05, wspace=0.42)
    right = gs[0, 1].subgridspec(3, 1, height_ratios=[12, 12, 6], hspace=0.42)
    axes = {
        "muse_bf16_proj": fig.add_subplot(gs[0, 0]),
        "muse_attention_decode": fig.add_subplot(right[0]),
        "muse_decode_layer": fig.add_subplot(right[1]),
        "muse_moe_w4a16": fig.add_subplot(right[2]),
    }
    handles = [Line2D([], [], ls="none", marker="o", markersize=7, color=BLUE_LIGHT,
                      markeredgecolor=SURFACE, label="First submit"),
               Line2D([], [], ls="none", marker="o", markersize=7, color=BLUE_DARK,
                      markeredgecolor=SURFACE, label="Final mark")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.955), ncol=2,
               fontsize=9, handletextpad=0.3, columnspacing=1.4)

    for p, ax in axes.items():
        d = data[p]
        rows = d["rows"]
        # bf16: an empty row between projection layers
        ys, y = [], 0.0
        prev_group = None
        for k in d["keys"]:
            group = dict(k).get("layer") if p == "muse_bf16_proj" else None
            if prev_group is not None and group != prev_group:
                y += 0.6
            ys.append(y)
            y += 1.0
            prev_group = group
        lo = min(min(r[1], r[2]) for r in rows)
        hi = max(max(r[1], r[2]) for r in rows)
        # speedups are ratios: a log axis gives 0.8x and 1.25x the same distance from 1x
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(0.85, lo * 0.95), hi * 1.08)
        ax.set_ylim(ys[-1] + 0.7, -0.7)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_locator(FixedLocator([0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(1.0, color=INK2, lw=1.0, zorder=1)
        for yy, (_, s_first, s_final) in zip(ys, rows):
            ax.plot([s_first, s_final], [yy, yy], color=AXIS, lw=2, zorder=2, solid_capstyle="butt")
            ax.scatter([s_first], [yy], s=34, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1.2,
                       zorder=3)
            ax.scatter([s_final], [yy], s=40, color=BLUE_DARK, edgecolor=SURFACE, linewidth=1.2,
                       zorder=4)
        ax.set_xlabel("speedup vs SGLang (line at 1x = SGLang)", fontsize=8)
        fs, ms = d["first"]["result"]["speedup"], d["final"]["result"]["speedup"]
        ax.set_title(f"{p}   {fs:.2f}x → {ms:.2f}x", loc="left", fontsize=10.5, fontweight="bold",
                     pad=18)
        ax.text(0, 1.0, f"first submit {d['first']['id'][:8]} · final mark {d['final']['id'][:8]}",
                transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom", ha="left")
    save(fig, outdir, "speedup_by_shape")


# ---------------------------------------------------------------- chart 2

def parse_md_tables(text):
    """Yield (heading, header, rows) for each markdown table in text."""
    heading = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-|:]+\|$", lines[i + 1]):
            hdr = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            yield heading, hdr, rows
            continue
        i += 1


def num(s):
    return float(s.replace(",", "."))


def chart_layer_time_breakdown(repo, outdir):
    path = os.path.join(repo, "kernels", "muse_decode_layer", "time-breakdown.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    table = None
    for heading, hdr, rows in parse_md_tables(text):
        if hdr and hdr[0] == "shape" and "mine" in hdr and "MoE" in hdr:
            table = (heading, hdr, rows)
            break
    if table is None:
        sys.exit("time-breakdown.md: no per-shape table (columns shape / mine / MoE) found")
    heading, hdr, rows = table
    col = {h: i for i, h in enumerate(hdr)}
    groups = [
        ("Attention", AQUA, ["attention"]),
        ("MoE", BLUE, ["MoE"]),
        ("BF16 projections (c1, split-K, cuBLAS)", ORANGE, ["proj. c1", "proj. split-K", "proj. cuBLAS"]),
        ("Small kernels (routing, norm + residual, other)", YELLOW,
         ["routing + pre-expert", "boundaries (norm + residual)", "others (outside the top 12)"]),
    ]
    for _, _, cols in groups:
        for c in cols:
            if c not in col:
                sys.exit(f"time-breakdown.md: column {c!r} missing")
    shapes = []
    for r in rows:
        parts = [sum(num(r[col[c]]) for c in cols) for _, _, cols in groups]
        mine, sgl, spd = num(r[col["mine"]]), num(r[col["SGLang"]]), num(r[col["speedup"]])
        if abs(sum(parts) - mine) > 0.6:
            sys.exit(f"time-breakdown.md: components of {r[0]!r} add to {sum(parts):.1f}, not {mine}")
        shapes.append((r[col["shape"]], parts, mine, sgl, spd))
    m = re.search(r"`([0-9a-f]{16})`", text)
    check = m.group(1)[:8] if m else "?"

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.78, bottom=0.08)
    header(fig, "Where decode_layer time goes, per scored shape",
           f"Wall time on the H200 (µs), split by each component's share of GPU time. Profile of check "
           f"{check} (v54),\nwithin noise of the board mark. Table: kernels/muse_decode_layer/time-breakdown.md.",
           top=0.965)
    n = len(shapes)
    xmax = max(s[2] for s in shapes) * 1.0
    ax.set_xlim(0, 1100)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(range(n), [s[0] for s in shapes])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("µs per call (wall time)", fontsize=8.5)
    gap = px_to_data(ax, 2)[0]
    thick = 0.55
    renderer = fig.canvas.get_renderer()
    for i, (name, parts, mine, sgl, spd) in enumerate(shapes):
        x = 0.0
        nz = [j for j, v in enumerate(parts) if v > 0]
        for j, v in enumerate(parts):
            if v <= 0:
                continue
            start = x + (gap if x > 0 else 0)
            end = x + v
            hbar(ax, i, start, end, thick, groups[j][1], rounded=(j == nz[-1]))
            # inline value only when it fits with padding
            label = f"{v:.0f}"
            t = ax.text((start + end) / 2, i, label, ha="center", va="center", fontsize=7,
                        color=INK if groups[j][1] == YELLOW else "white", zorder=4)
            bb = t.get_window_extent(renderer)
            seg_px = (ax.transData.transform((end, 0))[0] - ax.transData.transform((start, 0))[0])
            if bb.width + 8 > seg_px:
                t.remove()
            x = end
        ax.text(mine + 8, i, f"{mine:.0f} µs · {spd:.2f}x vs SGLang ({sgl:.0f} µs)", va="center",
                ha="left", fontsize=7.5, color=INK2)
    handles = [Patch(color=c, label=l) for l, c, _ in groups]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.15, 0.86), ncol=4, fontsize=8.5,
               handlelength=1.2, handleheight=0.9, columnspacing=1.6)
    save(fig, outdir, "layer_time_breakdown")


# ---------------------------------------------------------------- chart 3

BOARD_ORDER = ["overall"] + PROBLEMS


def snapshot_time(path, d):
    name = os.path.basename(path)
    if isinstance(d, dict) and d.get("capturado"):
        return dt.datetime.fromisoformat(d["capturado"]).astimezone(CET).replace(tzinfo=None), "capture"
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})-", name)
    if m:
        return dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}"), "file name"
    # no time recorded: the snapshot is no older than its newest submission
    latest = max((r.get("submitted_at", 0) for b in d.values() if isinstance(b, dict)
                  for r in b.get("rows", [])), default=0)
    if latest:
        return dt.datetime.fromtimestamp(latest, CET).replace(tzinfo=None), "newest submission"
    return dt.datetime.fromisoformat(name[:10] + "T12:00"), "date only"


def our_rank(board):
    you = board.get("you")
    if you and you.get("team") == TEAM:
        return you["rank"]
    rows = board.get("rows", [])
    for r in rows:
        if r.get("team") == TEAM:
            if r.get("rank") is not None:
                return r["rank"]
            # The boards after the close give speedups and no ranks: rank = 1 + teams faster.
            return 1 + sum(1 for o in rows if o.get("speedup", 0) > r["speedup"])
    return None


def board_size(board):
    return board.get("ranked") or len(board.get("rows", [])) or None


def load_boards(ev):
    snaps = defaultdict(dict)  # time -> board -> (rank, size)
    for path in sorted(glob.glob(os.path.join(ev, "boards", "*.json"))):
        d = read_json(path)
        when, _ = snapshot_time(path, d)
        if "general" in d:
            boards = {"overall": d["general"]}
            boards.update({k: v for k, v in d.items() if k in PROBLEMS})
        else:
            boards = {d["board"]: d}
        for name, b in boards.items():
            if name in BOARD_ORDER:
                snaps[when][name] = (our_rank(b), board_size(b))
    return OrderedDict(sorted(snaps.items()))


def chart_rank_by_day(ev, outdir):
    """Share of the board ahead of me at each snapshot. Ranks alone mislead: the boards grew
    from ~1,000 to ~2,000 teams while my speedups stayed put."""
    snaps = load_boards(ev)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.11, wspace=0.12)
    header(fig, "My position on each board, as the share of teams ahead of me",
           "Rank divided by the number of teams on the board at each snapshot. Top of the axis = first "
           "place. The boards kept growing (overall: 1,190 to 2,014 teams)\nand my speedups did not "
           "change after the last improving submit (30 Oct), so the position slid. Shaded: after that submit. Times in CET.", top=0.965)
    times = list(snaps)
    t0 = min(times) - dt.timedelta(hours=10)
    t1 = max(times) + dt.timedelta(hours=10)
    last_submit = dt.datetime(2026, 10, 30, 23, 7)
    for ax, name in zip(axes, BOARD_ORDER):
        pts = [(t, r / s * 100, r, s) for t, (r, s) in
               ((t, v[name]) for t, v in snaps.items() if name in v) if r and s]
        ax.axvspan(last_submit, t1, color=MUTED, alpha=0.08, lw=0, zorder=0)
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=BLUE, lw=2, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
        ax.scatter([q[0] for q in pts], [q[1] for q in pts], s=36, color=BLUE, edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        t_f, pc_f, r_f, s_f = pts[0]
        ax.text(0.02, 0.03, f"first ranked: {r_f:,} of {s_f:,} (top {pc_f:.0f}%)", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.5, color=INK2)
        t_l, pc_l, r_l, s_l = pts[-1]
        ax.annotate(f"{r_l:,} of {s_l:,}\ntop {pc_l:.0f}%", (t_l, pc_l), xytext=(0, 9),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK,
                    fontweight="bold")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(t0, t1)
        ax.set_ylim(70, 0)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        d_first = min(times).date()
        n_days = (max(times).date() - d_first).days + 1
        ticks = [dt.datetime.combine(d_first + dt.timedelta(days=i), dt.time()) for i in range(0, n_days, 3)]
        ax.xaxis.set_major_locator(FixedLocator([matplotlib.dates.date2num(t) for t in ticks]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelsize=7.5, rotation=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"top {v:.0f}%" if v else "1st"))
    save(fig, outdir, "rank_by_day")


def chart_final_boards(ev, outdir):
    """Distribution of every team's final speedup on each problem board, with mine marked."""
    boards = {}
    for p in PROBLEMS:
        paths = sorted(glob.glob(os.path.join(ev, "boards", f"*-{p}.json")))
        d = read_json(paths[-1])
        speeds = sorted((r["speedup"] for r in d["rows"] if r.get("speedup")), reverse=True)
        mine = next(r["speedup"] for r in d["rows"] if r.get("team") == TEAM)
        boards[p] = (speeds, mine, os.path.basename(paths[-1])[:10])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.08, hspace=0.55, wspace=0.12)
    header(fig, "Where my final marks sit on each board at the close",
           "Every team's best speedup vs SGLang on the H200, in 0.05x bins. Blue: my mark. "
           "Lines: the board median and 40th place.\nFinal boards, fetched after the 5 Nov close "
           f"({boards[PROBLEMS[0]][2]}).", top=0.97)
    order = ["muse_decode_layer", "muse_moe_w4a16", "muse_attention_decode", "muse_bf16_proj"]
    for ax, p in zip(axes.flat, order):
        speeds, mine, _ = boards[p]
        n = len(speeds)
        rank = 1 + sum(s > mine for s in speeds)
        median = speeds[n // 2]
        top40 = speeds[39]
        width = 0.05
        lo = 1.0
        hi = (int(max(speeds) / width) + 1) * width
        nb = int(round((hi - lo) / width))
        counts = [0] * nb
        for s in speeds:
            counts[min(nb - 1, max(0, int((s - lo) / width)))] += 1
        ymax = max(counts) * 1.32
        ax.set_xlim(lo - 0.02, hi + 0.02)
        ax.set_ylim(0, ymax)
        ours_bin = min(nb - 1, int((mine - lo) / width))
        bw = width - px_to_data(ax, 2)[0]
        for i, c in enumerate(counts):
            vbar(ax, lo + (i + 0.5) * width, c, bw, BLUE if i == ours_bin else "#cfcdc4", radius_px=2)
        for x, label, color, weight in ((median, f"median {median:.2f}x", INK2, "normal"),
                                        (top40, f"40th {top40:.2f}x", INK2, "normal")):
            ax.axvline(x, color=color, lw=1.0, zorder=1)
            ax.text(x, ymax * 0.98, " " + label, ha="left", va="top", fontsize=7.5, color=color,
                    fontweight=weight)
        left = mine < median  # put the label on the side away from the median line
        ax.annotate(f"me: {mine:.3f}x\nrank {rank:,} of {n:,}",
                    (lo + (ours_bin + 0.5) * width, counts[ours_bin]),
                    xytext=(lo + (ours_bin + (-1.5 if left else 2.5)) * width, ymax * 0.62),
                    textcoords="data", ha="right" if left else "left", va="bottom", fontsize=8,
                    color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=2, shrinkB=3))
        ax.set_title(p, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.tick_params(labelsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("teams", fontsize=8)
    save(fig, outdir, "final_boards")


# ---------------------------------------------------------------- chart 4

def chart_h100_hours(ev, outdir):
    per_day = defaultdict(float)
    with open(os.path.join(ev, "h100-hours.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                per_day[dt.date.fromisoformat(r["start"][:10])] += r["gpu_hours"]
    d0, d1 = min(per_day), max(per_day)
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    hours = [per_day.get(d, 0.0) for d in days]
    cum = []
    acc = 0.0
    for h in hours:
        acc += h
        cum.append(acc)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.08, right=0.9, top=0.78, bottom=0.1)
    header(fig, "H100 bench usage: GPU-hours per day and cumulative",
           "H100 bench jobs on the university cluster, attributed to the day each job started.", top=0.96)
    xs = list(range(len(days)))
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.set_ylim(0, (int(acc / 5) + 1) * 5)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    bar_w = px_to_data(ax, 24)[0]
    for x, h, c in zip(xs, hours, cum):
        vbar(ax, x, h, bar_w, BLUE)
        if abs(c - h) < 1.0:  # the cumulative marker sits on the bar top: label beside it
            ax.annotate(f"{h:.1f}", (x, h), xytext=(-16, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=INK2)
        else:
            ax.annotate(f"{h:.1f}", (x, h), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color=INK2)
    ax.plot(xs, cum, color=ORANGE, lw=2, zorder=3, solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(xs, cum, s=36, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate(f"{acc:.1f} h total", (xs[-1], cum[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(xs, [d.strftime("%b %d") for d in days])
    ax.set_ylabel("GPU-hours")
    handles = [Patch(color=BLUE, label="GPU-hours that day"),
               Line2D([], [], color=ORANGE, lw=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                      label="Cumulative")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.86), ncol=2, fontsize=8.5)
    save(fig, outdir, "h100_hours")


# ---------------------------------------------------------------- chart 5

MARKERS = {"muse_attention_decode": "o", "muse_moe_w4a16": "s", "muse_bf16_proj": "^",
           "muse_decode_layer": "D"}
LINTHRESH = 2.0


def chart_bench_vs_h200(ev, outdir):
    path = os.path.join(ev, "bench_vs_h200.csv")
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        r["x"], r["y"] = float(r["bench_pct"]), float(r["h200_pct"])
        r["flip"] = (r["x"] < 0 < r["y"]) or (r["y"] < 0 < r["x"])

    lim = (-100, 200)
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.07)
    n_flip = sum(r["flip"] for r in rows)
    header(fig, "The H100 bench vs the H200 judge: same change, different verdict",
           f"Each point is one change: relative time change predicted on the H100 bench (x) and measured "
           f"on the H200 judge (y).\nNegative = faster. {n_flip} of {len(rows)} changes flipped sign. "
           f"Symmetric log axes, linear within ±{LINTHRESH:g}%.", top=0.975)
    for axis in ("x", "y"):
        getattr(ax, f"set_{axis}scale")("symlog", linthresh=LINTHRESH, linscale=0.45)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100, 200]
    fmt = FuncFormatter(lambda v, _: f"{v:+g}%" if v else "0")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(FixedLocator([]))
    ax.tick_params(labelsize=8)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # flipped quadrants
    ax.add_patch(Rectangle((lim[0], 0), -lim[0], lim[1], facecolor=ORANGE, alpha=0.08, lw=0, zorder=0))
    ax.add_patch(Rectangle((0, lim[0]), lim[1], -lim[0], facecolor=ORANGE, alpha=0.08, lw=0, zorder=0))
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.plot(lim, lim, color=MUTED, lw=1.0, zorder=1)
    ax.text(150, 150, "bench = H200 ", rotation=45, rotation_mode="anchor", ha="right", va="bottom",
            fontsize=8, color=MUTED, transform_rotates_text=True)
    ax.text(-92, 185, "Bench said faster,\nH200 was slower", ha="left", va="top", fontsize=9.5,
            color=INK, fontweight="bold")
    ax.text(185, -92, "Bench said slower,\nH200 was faster", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    ax.text(-92, -1.2, "Both faster", ha="left", va="top", fontsize=9, color=INK2)
    ax.text(185, 1.2, "Both slower", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_xlabel("H100 bench: predicted time change", fontsize=9)
    ax.set_ylabel("H200 judge: measured time change", fontsize=9)

    for r in rows:
        ax.scatter([r["x"]], [r["y"]], marker=MARKERS.get(r["problem"], "o"), s=58,
                   color=ORANGE if r["flip"] else BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=5)
    place_labels(ax, rows)

    shape_h = [Line2D([], [], ls="none", marker=MARKERS[p], markersize=7, color=INK2,
                      markeredgecolor=SURFACE, label=p) for p in
               ["muse_attention_decode", "muse_moe_w4a16", "muse_bf16_proj", "muse_decode_layer"]]
    color_h = [Line2D([], [], ls="none", marker="o", markersize=8, color=ORANGE, markeredgecolor=SURFACE,
                      label="sign flipped"),
               Line2D([], [], ls="none", marker="o", markersize=8, color=BLUE, markeredgecolor=SURFACE,
                      label="sign carried over")]
    fig.legend(handles=color_h + shape_h, loc="upper left", bbox_to_anchor=(0.085, 0.885), ncol=6,
               fontsize=8, handletextpad=0.3, columnspacing=1.1)
    save(fig, outdir, "bench_vs_h200")


def place_labels(ax, rows):
    """Greedy, deterministic label placement: first candidate offset that overlaps nothing."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    taken = []
    for r in rows:  # every marker is an obstacle
        cx, cy = ax.transData.transform((r["x"], r["y"]))
        taken.append(matplotlib.transforms.Bbox.from_bounds(cx - 11, cy - 11, 22, 22))
    for t in ax.texts:
        taken.append(t.get_window_extent(renderer).expanded(1.02, 1.1))
    cands = []
    for dist in (8, 18, 28, 40, 55):
        for dx, dy, ha, va in ((1, 0, "left", "center"), (-1, 0, "right", "center"),
                               (0, 1, "center", "bottom"), (0, -1, "center", "top"),
                               (0.8, 0.8, "left", "bottom"), (-0.8, 0.8, "right", "bottom"),
                               (0.8, -0.8, "left", "top"), (-0.8, -0.8, "right", "top")):
            cands.append((dx * dist, dy * dist, ha, va, dist))
    # place points in a dense-first order so crowded ones get the close spots
    order = sorted(range(len(rows)), key=lambda i: (abs(rows[i]["x"]) + abs(rows[i]["y"]), i))
    for i in order:
        r = rows[i]
        best = None
        for dx, dy, ha, va, dist in cands:
            t = ax.annotate(r["change"], (r["x"], r["y"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=7.5, color=INK, zorder=6)
            bb = t.get_window_extent(renderer).expanded(1.04, 1.15)
            inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(o) for o in taken):
                best = (t, bb, dist)
                break
            t.remove()
        if best is None:  # last resort: far right with a leader line
            t = ax.annotate(r["change"], (r["x"], r["y"]), xytext=(60, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=7.5, color=INK, zorder=6)
            best = (t, t.get_window_extent(renderer), 60)
        t, bb, dist = best
        if dist > 8:
            t.arrow_patch = None
            t.remove()
            t = ax.annotate(r["change"], (r["x"], r["y"]), xytext=t.xyann, textcoords="offset points",
                            ha=t.get_ha(), va=t.get_va(), fontsize=7.5, color=INK, zorder=6,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=5))
        taken.append(bb)


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 charts.py <repo_root>")
    repo = os.path.abspath(sys.argv[1])
    ev = os.path.join(repo, "evidence")
    outdir = os.path.join(repo, "charts")
    os.makedirs(outdir, exist_ok=True)
    chart_speedup_by_shape(ev, outdir)
    chart_layer_time_breakdown(repo, outdir)
    chart_rank_by_day(ev, outdir)
    chart_final_boards(ev, outdir)
    chart_h100_hours(ev, outdir)
    chart_bench_vs_h200(ev, outdir)


if __name__ == "__main__":
    main()
