#!/usr/bin/env python
"""Show where the true value falls outside the range of the posterior samples, in 1D and 2D.

1D: truth outside [min, max] of a parameter's samples (the in_minmax check in grade.py).
2D: truth outside the convex hull of a pair's samples, the 2D analogue of [min, max].

The hull sits inside the min-max box, so a pair whose truth misses either 1D range also
misses the hull. Such pairs are drawn separately from "2D-only" misses, where the truth is
inside both 1D ranges but still off the sample cloud.

Uses the same layout as grade.py (see its docstring).

Usage:
    python plot_range.py ds_Sep2_2026                  # use all samples
    python plot_range.py ds_Sep2_2026 --n_sample 100   # use only the first 100 samples

Outputs, per submission, under <submission>/grades/n_<tag>/:
    range_check.csv                     one row per parameter (1D) and per pair (2D)
    plots/range_summary.png             in/out grid, one panel per dataset
    plots/range_detail_<dataset>.png    only the missed parameters and 2D-only pairs
and, across submissions, under ROOT/leaderboard/n_<tag>/:
    range_leaderboard.csv, range_leaderboard_by_dataset.csv, range_leaderboard.png

The range leaderboard is information, not a ranking: both checks get easier with more
samples, so it follows the order of grade.py's leaderboard.csv when that file exists.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull, QhullError

from grade import (GRADES_DIR, GRID, INK, INK_2, MUTED, SURFACE, BLUE,
                   find_datasets, find_submissions, load_samples, load_truth, short_name)

HULL_TOL = 1e-12  # truth on the hull boundary counts as inside
MAX_DETAIL_PANELS = 40

# Summary grid cell states
INSIDE, OUT_VIA_1D, OUT_2D_ONLY, OUT_1D, NO_HULL = 0, 1, 2, 3, 4
STATE_COLORS = ["#e9e8e2", "#f5c4c3", "#e34948", "#9e1b1a", "#c9d3de"]
STATE_LABELS = ["inside", "pair outside (a 1D range already misses)",
                "pair outside, 2D only (inside both 1D ranges)", "parameter outside 1D range",
                "n/a (samples have no 2D spread)"]


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def outside_1d(s, t):
    """Distance from the truth to the [min, max] range of the samples (0 = inside)."""
    return float(max(s.min() - t, t - s.max(), 0.0))


def build_hull(P):
    """Convex hull of 2D points, or None when the points are a point or a line (no area)."""
    try:
        return ConvexHull(P)
    except (QhullError, ValueError):
        return None


def outside_hull(hull, t):
    """Largest distance of the truth past a hull edge line (0 = inside).

    Each row of hull.equations is a unit normal and offset with normal . x + offset <= 0 inside.
    """
    return float(max(np.max(hull.equations[:, :2] @ t + hull.equations[:, 2]), 0.0))


def check_dataset(S, truth):
    """Return one row per parameter (1D) and one per pair (2D), plus the hulls for plotting."""
    names = list(truth.index)
    y = truth.to_numpy(float)
    d = len(names)
    rows, hulls = [], {}
    dist_1d = [outside_1d(S[:, k], y[k]) for k in range(d)]
    for k in range(d):
        rows.append(dict(kind="1d", param_i=names[k], param_j="", outside=dist_1d[k] > 0,
                         distance=dist_1d[k], x_true_i=y[k], x_true_j=np.nan))
    for i in range(d):
        for j in range(i + 1, d):
            hull = build_hull(S[:, [i, j]])
            hulls[i, j] = hull
            via_1d = dist_1d[i] > 0 or dist_1d[j] > 0
            if hull is None:  # no hull, but a 1D miss still puts the truth outside
                dist, out = np.nan, (True if via_1d else np.nan)
            else:
                dist = outside_hull(hull, y[[i, j]])
                out = dist > HULL_TOL
            rows.append(dict(kind="2d", param_i=names[i], param_j=names[j], outside=out,
                             distance=dist, x_true_i=y[i], x_true_j=y[j],
                             via_1d=via_1d))
    return pd.DataFrame(rows), hulls


def misses_1d(rc):
    return (rc.kind == "1d") & (rc.outside == True)  # noqa: E712  (outside holds True/False/NaN)


def misses_2d_only(rc):
    return (rc.kind == "2d") & (rc.outside == True) & (rc.via_1d == False)  # noqa: E712


def summarize(rc):
    """Per-dataset counts and fractions of misses for the range leaderboard."""
    n_params, n_pairs = int((rc.kind == "1d").sum()), int((rc.kind == "2d").sum())
    n1, n2 = int(misses_1d(rc).sum()), int(misses_2d_only(rc).sum())
    n_na = int(((rc.kind == "2d") & rc.outside.isna()).sum())
    return dict(n_miss_1d=n1, n_miss_2d_only=n2, n_na_2d=n_na,
                frac_miss_1d=n1 / n_params, frac_miss_2d_only=n2 / n_pairs, frac_na_2d=n_na / n_pairs,
                max_distance_1d=rc.loc[rc.kind == "1d", "distance"].max())


def state_grid(rc, names):
    """d x d grid of cell states: diagonal = 1D, lower triangle = 2D, upper triangle = NaN."""
    idx = {n: k for k, n in enumerate(names)}
    grid = np.full((len(names), len(names)), np.nan)
    for r in rc.itertuples():
        if r.kind == "1d":
            grid[idx[r.param_i], idx[r.param_i]] = OUT_1D if r.outside else INSIDE
        elif pd.isna(r.outside):
            grid[idx[r.param_j], idx[r.param_i]] = NO_HULL
        else:
            grid[idx[r.param_j], idx[r.param_i]] = ((OUT_VIA_1D if r.via_1d else OUT_2D_ONLY)
                                                    if r.outside else INSIDE)
    return grid


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_summary(checks, names, title, path):
    """One panel per dataset: diagonal = 1D check, lower triangle = 2D hull check."""
    n = len(checks)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.9 * nrow + 0.8), squeeze=False)
    cmap = ListedColormap(STATE_COLORS)
    cmap.set_bad(SURFACE)
    d = len(names)
    for ax, (ds, rc) in zip(axes.flat, checks.items()):
        grid = state_grid(rc, names)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=len(STATE_COLORS) - 0.5)
        ax.set_xticks(np.arange(-0.5, d), minor=True)
        ax.set_yticks(np.arange(-0.5, d), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=0.8)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels([p.lstrip("x") for p in names], fontsize=6)
        ax.set_yticklabels([p.lstrip("x") for p in names], fontsize=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        n1, n2 = int(misses_1d(rc).sum()), int(misses_2d_only(rc).sum())
        n_pairs = int((rc.kind == "2d").sum())
        ax.set_title(f"{short_name(ds)}\n1D misses: {n1}/{d}   2D-only misses: {n2}/{n_pairs}",
                     fontsize=8, color=INK)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles = [Patch(color=c, label=l) for c, l in zip(STATE_COLORS, STATE_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title + "\ndiagonal: truth vs [min, max] of samples  |  lower triangle: truth vs "
                 "convex hull of the pair's samples", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_detail(S, truth, rc, hulls, title, path, lim):
    """One panel per 1D miss and per 2D-only miss, showing how far outside the truth is."""
    names = list(truth.index)
    y = truth.to_numpy(float)
    idx = {n: k for k, n in enumerate(names)}
    panels = ([("1d", r) for r in rc[misses_1d(rc)].itertuples()]
              + [("2d", r) for r in rc[misses_2d_only(rc)].itertuples()])
    shown = panels[:MAX_DETAIL_PANELS]
    ncol = min(len(shown), 5)
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.6 * nrow + 0.9), squeeze=False)
    for ax, (kind, r) in zip(axes.flat, shown):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
        if kind == "1d":
            k = idx[r.param_i]
            s = S[:, k]
            ax.axvspan(s.min(), s.max(), color=GRID, lw=0)
            ax.hist(s, bins=25, range=lim, color=BLUE, histtype="stepfilled", lw=0)
            ax.axvline(y[k], color=STATE_COLORS[OUT_1D], lw=1.5)
            ax.set_xlim(lim)
            ax.set_yticks([])
            ax.set_xlabel(r.param_i, fontsize=8, color=INK_2)
            ax.set_title(f"1D: {r.param_i}, {r.distance:.3g} outside", fontsize=8, color=INK)
        else:
            i, j = idx[r.param_i], idx[r.param_j]
            P = S[:, [i, j]]
            ax.scatter(P[:, 0], P[:, 1], s=2, color=BLUE, alpha=0.3, linewidths=0, rasterized=True)
            ring = np.append(hulls[i, j].vertices, hulls[i, j].vertices[0])
            ax.plot(P[ring, 0], P[ring, 1], color=INK_2, lw=0.8)
            ax.scatter([y[i]], [y[j]], marker="x", s=40, color=STATE_COLORS[OUT_2D_ONLY], lw=1.5, zorder=3)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(r.param_i, fontsize=8, color=INK_2)
            ax.set_ylabel(r.param_j, fontsize=8, color=INK_2)
            ax.set_title(f"2D: {r.param_i}-{r.param_j}, {r.distance:.3g} outside", fontsize=8, color=INK)
    for ax in axes.flat[len(shown):]:
        ax.set_visible(False)
    extra = f"  (first {MAX_DETAIL_PANELS} of {len(panels)} shown)" if len(panels) > len(shown) else ""
    fig.suptitle(f"{title}{extra}\n1D: histogram, gray = [min, max], red line = truth  |  "
                 f"2D-only: samples, gray outline = convex hull, red × = truth", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


RANGE_PANELS = [
    ("frac_miss_1d", "Parameters with truth outside [min, max]", OUT_1D),
    ("frac_miss_2d_only", "Pairs with truth outside the hull,\ninside both 1D ranges (2D only)", OUT_2D_ONLY),
    ("frac_na_2d", "Pairs with no hull (n/a)", NO_HULL),
]


def plot_range_leaderboard(by_dataset, overall, path):
    """One panel per miss fraction: bar = mean over datasets, dots = each dataset."""
    methods = list(overall.index)
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, len(RANGE_PANELS), figsize=(4.2 * len(RANGE_PANELS), 0.45 * len(methods) + 1.6),
                             sharey=True)
    for ax, (col, label, state) in zip(axes, RANGE_PANELS):
        ax.barh(y, overall[col], height=0.6, color=STATE_COLORS[state], lw=0)
        for k, m in enumerate(methods):
            vals = by_dataset.loc[by_dataset.method == m, col]
            ax.scatter(vals, np.full(len(vals), k), s=10, color=INK_2, zorder=3, linewidths=0)
        ax.set_xlim(0, max(1e-3, by_dataset[col].max()) * 1.08)
        ax.set_title(label, fontsize=9, color=INK)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Is the truth inside the sample range? (lower is better; information only, "
                 "not a ranking)\nbar = mean over datasets, dots = each dataset", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def leaderboard_order(root, tag, methods):
    """Methods in the order of grade.py's leaderboard when it exists, else alphabetical."""
    path = root / "leaderboard" / tag / "leaderboard.csv"
    ranked = list(pd.read_csv(path, index_col=0).index) if path.is_file() else []
    return [m for m in ranked if m in methods] + sorted(m for m in methods if m not in ranked)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset collection folder, e.g. ds_Sep2_2026")
    ap.add_argument("-n", "--n_sample", default="all",
                    help="'all' or an integer: use only the first N samples of each results file")
    ap.add_argument("--prior_low", type=float, default=-1.0)
    ap.add_argument("--prior_high", type=float, default=1.0)
    args = ap.parse_args()

    n_sample = None if args.n_sample == "all" else int(args.n_sample)
    tag = f"n_{args.n_sample}"
    pad = 0.025 * (args.prior_high - args.prior_low)
    lim = (args.prior_low - pad, args.prior_high + pad)
    root = args.root.resolve()
    datasets = find_datasets(root)

    all_rows = []
    for method, method_dir in find_submissions(root):
        print(f"=== {method}")
        out = method_dir / GRADES_DIR / tag
        (out / "plots").mkdir(parents=True, exist_ok=True)
        checks, names = {}, None
        for ds_dir in datasets:
            truth = load_truth(ds_dir)
            S = load_samples(method_dir, ds_dir.name, list(truth.index), n_sample)
            if S is None:
                continue
            names = list(truth.index)
            rc, hulls = check_dataset(S, truth)
            checks[ds_dir.name] = rc
            all_rows.append(dict(method=method, dataset=ds_dir.name, n_samples=S.shape[0], **summarize(rc)))
            n1, n2 = int(misses_1d(rc).sum()), int(misses_2d_only(rc).sum())
            print(f"  {ds_dir.name}: {n1} 1D misses, {n2} 2D-only misses")
            if n1 or n2:
                plot_detail(S, truth, rc, hulls, f"{method} | {ds_dir.name} | {S.shape[0]} samples",
                            out / "plots" / f"range_detail_{ds_dir.name}.png", lim)
        if not checks:
            print("  nothing checked\n")
            continue
        table = pd.concat([rc.assign(dataset=ds) for ds, rc in checks.items()], ignore_index=True)
        table = table[["dataset"] + [c for c in table.columns if c != "dataset"]]
        table.to_csv(out / "range_check.csv", index=False)
        plot_summary(checks, names, f"{method}: is the truth inside the sample range? ({args.n_sample} samples)",
                     out / "plots" / "range_summary.png")
        print(f"  -> {out}\n")

    if not all_rows:
        print("No results checked.")
        return

    by_dataset = pd.DataFrame(all_rows)
    overall = by_dataset.drop(columns=["dataset"]).groupby("method").mean(numeric_only=True)
    overall.insert(0, "n_datasets_checked", by_dataset.groupby("method").size())
    overall = overall.loc[leaderboard_order(root, tag, list(overall.index))]
    lb = root / "leaderboard" / tag
    lb.mkdir(parents=True, exist_ok=True)
    by_dataset.to_csv(lb / "range_leaderboard_by_dataset.csv", index=False)
    overall.to_csv(lb / "range_leaderboard.csv")
    plot_range_leaderboard(by_dataset, overall, lb / "range_leaderboard.png")
    cols = ["n_datasets_checked", "n_samples", "frac_miss_1d", "frac_miss_2d_only", "frac_na_2d"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("Range check (same order as leaderboard.csv; information only):")
        print(overall[cols].to_string())
    print(f"\n-> {lb}")


if __name__ == "__main__":
    main()
