#!/usr/bin/env python
"""Grade parameter-estimation results against the ground truth.

Layout expected under ROOT (e.g. ds_Sep2_2026/):
    <dataset>/x_true.csv                  one folder per dataset (not starting with "Results")
    Results_<method>/results_<dataset>.csv  posterior samples, one column per parameter

Usage:
    python grade.py ds_Sep2_2026                  # use all samples
    python grade.py ds_Sep2_2026 --n_sample 100   # use only the first 100 samples
    python grade.py ds_Sep2_2026 --no_plots

Outputs:
    Results_<method>/grades/n_<tag>/   per_parameter.csv, pairwise.csv, summary.csv, plots/
    ROOT/leaderboard/n_<tag>/          leaderboard.csv, leaderboard_by_dataset.csv, leaderboard.png

See METRICS.md for what each metric means and which direction is better.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS_PREFIX = "Results"
RESULTS_FILE = "results_{dataset}.csv"
TRUTH_FILE = "x_true.csv"
LEVELS = (0.50, 0.90, 0.95)

# Chart colors (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------------------
# Discovery and loading
# ----------------------------------------------------------------------------
def find_datasets(root):
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith(RESULTS_PREFIX) and (p / TRUTH_FILE).is_file())


def find_methods(root):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(RESULTS_PREFIX))


def load_truth(dataset_dir):
    return pd.read_csv(dataset_dir / TRUTH_FILE, index_col=0).iloc[:, 0]


def load_samples(method_dir, dataset_name, param_names, n_sample):
    """Return an (n, d) sample array in the order of param_names, or None if unavailable."""
    path = method_dir / RESULTS_FILE.format(dataset=dataset_name)
    if not path.is_file():
        print(f"  [skip] {dataset_name}: {path.name} not found")
        return None
    df = pd.read_csv(path, index_col=0)
    missing = [p for p in param_names if p not in df.columns]
    if missing:
        print(f"  [skip] {dataset_name}: missing columns {missing}")
        return None
    if n_sample is not None:
        if len(df) < n_sample:
            print(f"  [warn] {dataset_name}: only {len(df)} samples (< {n_sample}), using all")
        df = df.iloc[:n_sample]
    return df[param_names].to_numpy(float)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def crps(samples, y):
    """Empirical CRPS = E|X - y| - 0.5 E|X - X'|, using the sorted-sample identity for the second term."""
    x = np.sort(samples)
    n = len(x)
    i = np.arange(1, n + 1)
    return np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n**2


def energy_score(S, y):
    """Multivariate CRPS: E||X - y|| - 0.5 E||X - X'||."""
    n = len(S)
    return np.mean(np.linalg.norm(S - y, axis=1)) - pdist(S).sum() / n**2


def kde_hpd_level(points, truth):
    """Smallest highest-density region (as a probability) that contains the truth.

    Density is a Gaussian KDE; sample densities are leave-one-out so a sample
    does not count its own kernel. Returns NaN if the KDE cannot be built.
    """
    n, d = points.shape
    try:
        kde = stats.gaussian_kde(points.T)
    except np.linalg.LinAlgError:
        return np.nan
    self_term = 1.0 / np.sqrt((2 * np.pi) ** d * np.linalg.det(kde.covariance))
    dens_loo = (kde(points.T) - self_term / n) * n / (n - 1)
    return float(np.mean(dens_loo > kde(truth[:, None])[0]))


def kde_entropy_1d(samples, low, high):
    """Differential entropy (nats) = -mean log f(x_i), with f a leave-one-out Gaussian KDE
    reflected at the prior bounds. Robust to repeated samples (e.g. MCMC rejections)."""
    n = len(samples)
    if np.std(samples) == 0:
        return np.nan
    kde = stats.gaussian_kde(samples)
    h2 = kde.covariance[0, 0]
    dens = kde(samples) + kde(2 * low - samples) + kde(2 * high - samples)
    dens_loo = (dens - 1.0 / (n * np.sqrt(2 * np.pi * h2))) * n / (n - 1)
    return float(-np.mean(np.log(dens_loo)))


def mahalanobis_level(S, y):
    """Fraction of samples closer to the sample mean (in Mahalanobis distance) than the truth."""
    mu = S.mean(axis=0)
    prec = np.linalg.pinv(np.cov(S, rowvar=False))
    D = S - mu
    d2 = np.einsum("ij,jk,ik->i", D, prec, D)
    dt = y - mu
    return float(np.mean(d2 < dt @ prec @ dt))


def gaussian_entropy(S):
    d = S.shape[1]
    sign, logdet = np.linalg.slogdet(np.cov(S, rowvar=False))
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet) if sign > 0 else np.nan


def pct(level):
    return f"{round(level * 100)}"


def grade_dataset(S, truth, prior_low, prior_high):
    """Return (per_parameter rows, pairwise rows, summary dict) for one dataset."""
    names = list(truth.index)
    y = truth.to_numpy(float)
    n, d = S.shape
    prior_std = (prior_high - prior_low) / np.sqrt(12)
    prior_entropy = np.log(prior_high - prior_low)

    per_param = []
    for k, name in enumerate(names):
        s, t = S[:, k], y[k]
        mean, median, std = s.mean(), np.median(s), s.std(ddof=1)
        entropy = kde_entropy_1d(s, prior_low, prior_high)
        row = dict(param=name, x_true=t, mean=mean, median=median, std=std,
                   bias=mean - t, abs_err_mean=abs(mean - t), abs_err_median=abs(median - t),
                   z_score=(mean - t) / std if std > 0 else np.nan,
                   percentile_rank=(np.sum(s < t) + 0.5 * np.sum(s == t)) / n,
                   in_minmax=bool(s.min() <= t <= s.max()))
        for lv in LEVELS:
            lo, hi = np.quantile(s, [(1 - lv) / 2, (1 + lv) / 2])
            row[f"in_ci{pct(lv)}"] = bool(lo <= t <= hi)
        row.update(contraction=std / prior_std, entropy=entropy,
                   entropy_reduction=prior_entropy - entropy, crps=crps(s, t))
        per_param.append(row)
    pp = pd.DataFrame(per_param)

    pairwise = []
    for i in range(d):
        for j in range(i + 1, d):
            level = kde_hpd_level(S[:, [i, j]], y[[i, j]])
            row = dict(param_i=names[i], param_j=names[j], hpd_level=level,
                       corr=np.corrcoef(S[:, i], S[:, j])[0, 1])
            for lv in LEVELS:
                row[f"in_hpd{pct(lv)}"] = bool(level <= lv) if np.isfinite(level) else np.nan
            pairwise.append(row)
    pw = pd.DataFrame(pairwise)

    joint_level = mahalanobis_level(S, y)
    joint_entropy = gaussian_entropy(S)
    summary = dict(
        n_samples=n, n_params=d,
        crps_mean=pp.crps.mean(),
        energy_score=energy_score(S, y),
        rmse_mean=np.sqrt(np.mean(pp.bias**2)),
        mae_median=pp.abs_err_median.mean(),
        mean_abs_z=pp.z_score.abs().mean(),
        max_abs_z=pp.z_score.abs().max(),
        frac_abs_z_gt2=(pp.z_score.abs() > 2).mean(),
        cov1d_minmax=pp.in_minmax.mean(),
        **{f"cov1d_ci{pct(lv)}": pp[f"in_ci{pct(lv)}"].mean() for lv in LEVELS},
        rank_ks=stats.kstest(pp.percentile_rank, "uniform").statistic,
        **{f"cov2d_hpd{pct(lv)}": pw[f"in_hpd{pct(lv)}"].astype(float).mean() for lv in LEVELS},
        joint_level_mahalanobis=joint_level,
        **{f"in_joint{pct(lv)}": joint_level <= lv for lv in LEVELS},
        contraction_mean=pp.contraction.mean(),
        entropy_reduction_1d_mean=pp.entropy_reduction.mean(),
        joint_entropy_gauss=joint_entropy,
        joint_entropy_reduction_gauss=d * prior_entropy - joint_entropy,
    )
    return pp, pw, summary


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def short_name(dataset):
    return dataset.replace("_strerr", "\nstrerr")


def plot_corner(S, truth, pw, title, path):
    """Lower: samples with the truth marked. Diagonal: marginals with 90% interval.
    Upper: KDE HPD level of the truth for that pair (x = truth outside the 90% region)."""
    names = list(truth.index)
    y = truth.to_numpy(float)
    d = len(names)
    level = np.full((d, d), np.nan)
    idx = {n: k for k, n in enumerate(names)}
    for r in pw.itertuples():
        level[idx[r.param_i], idx[r.param_j]] = r.hpd_level

    fig, axes = plt.subplots(d, d, figsize=(1.1 * d, 1.1 * d))
    lim = (-1.05, 1.05)
    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r > c:
                ax.scatter(S[:, c], S[:, r], s=1, color=BLUE, alpha=0.25, linewidths=0, rasterized=True)
                ax.axvline(y[c], color=INK, lw=0.6)
                ax.axhline(y[r], color=INK, lw=0.6)
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_visible(True)
            elif r == c:
                s = S[:, c]
                lo, hi = np.quantile(s, [0.05, 0.95])
                ax.axvspan(lo, hi, color=GRID, lw=0)
                ax.hist(s, bins=25, range=(-1, 1), color=BLUE, histtype="stepfilled", lw=0)
                ax.axvline(y[c], color=INK, lw=1.2)
                ax.set_xlim(lim)
                ax.spines["bottom"].set_visible(True)
            else:
                v = level[r, c]
                ax.set_facecolor(SEQ_BLUE(v) if np.isfinite(v) else GRID)
                label = "n/a" if not np.isfinite(v) else (f"{v:.2f}" + ("\n×" if v > 0.9 else ""))
                ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=7,
                        color="#ffffff" if np.isfinite(v) and v > 0.55 else INK, transform=ax.transAxes)
            if r == d - 1:
                ax.set_xlabel(names[c], fontsize=9)
            if c == 0 and r > 0:
                ax.set_ylabel(names[r], fontsize=9)
    fig.suptitle(title + "\nlower: samples, black lines = truth  |  diagonal: marginal, gray = 90% interval  |  "
                 "upper: HPD level of truth (× = outside 90% region; lower is better)",
                 fontsize=12, color=INK, y=0.995)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.955, wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_overview(per_param, title, path):
    """Small multiples: one column per dataset, one row per per-parameter metric."""
    datasets = list(dict.fromkeys(per_param.dataset))
    rows = [("crps", "CRPS (lower better)", None),
            ("contraction", "Contraction (std / prior std)", [1.0]),
            ("z_score", "z-score of mean (|z| > 2 = confidently wrong)", [-2.0, 0.0, 2.0]),
            ("percentile_rank", "Percentile rank of truth", [0.05, 0.5, 0.95])]
    fig, axes = plt.subplots(len(rows), len(datasets), figsize=(3.2 * len(datasets), 2.4 * len(rows)),
                             sharey="row", squeeze=False)
    for c, ds in enumerate(datasets):
        sub = per_param[per_param.dataset == ds]
        x = np.arange(len(sub))
        for r, (col, label, refs) in enumerate(rows):
            ax = axes[r, c]
            ax.bar(x, sub[col], width=0.7, color=BLUE, lw=0)
            for ref in refs or []:
                ax.axhline(ref, color=MUTED, lw=0.8, ls="--" if ref != 0 else "-")
            ax.tick_params(labelsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([p.lstrip("x") for p in sub.param] if r == len(rows) - 1 else [])
            if c == 0:
                ax.set_ylabel(textwrap.fill(label, 22), fontsize=8)
            if r == 0:
                ax.set_title(short_name(ds), fontsize=8, color=INK)
        axes[-1, c].set_xlabel("parameter index", fontsize=8)
    axes[3, 0].set_ylim(0, 1)
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


LEADERBOARD_PANELS = [
    ("crps_mean", "Mean CRPS (lower better)", None),
    ("energy_score", "Energy score (lower better)", None),
    ("rmse_mean", "RMSE of posterior mean (lower better)", None),
    ("mean_abs_z", "Mean |z| (lower better)", None),
    ("cov1d_ci90", "1D coverage, 90% interval (near 0.9 best)", 0.9),
    ("cov2d_hpd90", "2D coverage, 90% HPD (near 0.9 best)", 0.9),
    ("contraction_mean", "Mean contraction (lower = sharper)", 1.0),
    ("entropy_reduction_1d_mean", "Mean 1D entropy reduction, nats (higher = sharper)", None),
]


def plot_leaderboard(by_dataset, overall, path):
    methods = list(overall.index[:len(SERIES)])  # fixed colors; extra methods stay in the CSVs only
    datasets = list(dict.fromkeys(by_dataset.dataset))
    bar_h = 0.8 / len(methods)
    fig, axes = plt.subplots(2, 4, figsize=(18, 2.0 + 0.45 * len(datasets) * max(1, len(methods) / 2) * 2),
                             sharey=True)
    for ax, (col, label, ref) in zip(axes.flat, LEADERBOARD_PANELS):
        for m_i, m in enumerate(methods):
            sub = by_dataset[by_dataset.method == m].set_index("dataset").reindex(datasets)
            ypos = np.arange(len(datasets)) + (m_i - (len(methods) - 1) / 2) * bar_h
            ax.barh(ypos, sub[col], height=bar_h * 0.9, color=SERIES[m_i], lw=0, label=m)
        if ref is not None:
            ax.axvline(ref, color=MUTED, lw=0.8, ls="--")
        ax.set_title(textwrap.fill(label, 32), fontsize=9, color=INK)
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([short_name(d) for d in datasets], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    axes[0, 0].invert_yaxis()  # shared y: invert once so the first dataset is on top
    title = "Leaderboard by dataset"
    if len(overall) > len(methods):
        title += f" (top {len(methods)} of {len(overall)} methods shown)"
    fig.suptitle(title if len(methods) > 1 else f"{title}: {methods[0]}", fontsize=12, color=INK)
    if len(methods) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset collection folder, e.g. ds_Sep2_2026")
    ap.add_argument("-n", "--n_sample", default="all",
                    help="'all' or an integer: use only the first N samples of each results file")
    ap.add_argument("--no_plots", action="store_true", help="skip figures")
    ap.add_argument("--prior_low", type=float, default=-1.0)
    ap.add_argument("--prior_high", type=float, default=1.0)
    args = ap.parse_args()

    n_sample = None if args.n_sample == "all" else int(args.n_sample)
    tag = f"n_{args.n_sample}"
    root = args.root.resolve()
    datasets = find_datasets(root)
    methods = find_methods(root)
    print(f"Root: {root}\nDatasets ({len(datasets)}): {[d.name for d in datasets]}\n"
          f"Methods ({len(methods)}): {[m.name for m in methods]}\nSamples used: {args.n_sample}\n")

    all_summaries = []
    for method_dir in methods:
        method = method_dir.name
        print(f"=== {method}")
        out = method_dir / "grades" / tag
        (out / "plots").mkdir(parents=True, exist_ok=True)
        pps, pws, sums = [], [], []
        for ds_dir in datasets:
            truth = load_truth(ds_dir)
            S = load_samples(method_dir, ds_dir.name, list(truth.index), n_sample)
            if S is None:
                continue
            print(f"  grading {ds_dir.name} ({S.shape[0]} samples)")
            pp, pw, summary = grade_dataset(S, truth, args.prior_low, args.prior_high)
            pps.append(pp.assign(dataset=ds_dir.name))
            pws.append(pw.assign(dataset=ds_dir.name))
            sums.append(dict(method=method, dataset=ds_dir.name, **summary))
            if not args.no_plots:
                plot_corner(S, truth, pw, f"{method} | {ds_dir.name} | {S.shape[0]} samples",
                            out / "plots" / f"corner_{ds_dir.name}.png")
        if not sums:
            print("  nothing graded\n")
            continue
        per_param = pd.concat(pps, ignore_index=True)
        per_param = per_param[["dataset"] + [c for c in per_param.columns if c != "dataset"]]
        pairwise = pd.concat(pws, ignore_index=True)
        pairwise = pairwise[["dataset"] + [c for c in pairwise.columns if c != "dataset"]]
        summary = pd.DataFrame(sums)
        per_param.to_csv(out / "per_parameter.csv", index=False)
        pairwise.to_csv(out / "pairwise.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        if not args.no_plots:
            plot_overview(per_param, f"{method}: per-parameter metrics ({args.n_sample} samples)",
                          out / "plots" / "metrics_overview.png")
        all_summaries.append(summary)
        print(f"  -> {out}\n")

    if not all_summaries:
        print("No results graded.")
        return

    by_dataset = pd.concat(all_summaries, ignore_index=True)
    numeric = by_dataset.drop(columns=["dataset"]).groupby("method").mean(numeric_only=True)
    numeric.insert(0, "n_datasets_graded", by_dataset.groupby("method").size())
    # Complete submissions (graded on the most datasets) rank ahead of incomplete ones
    overall = numeric.sort_values(["n_datasets_graded", "crps_mean"], ascending=[False, True])
    overall.insert(0, "rank", np.arange(1, len(overall) + 1))
    lb = root / "leaderboard" / tag
    lb.mkdir(parents=True, exist_ok=True)
    by_dataset.to_csv(lb / "leaderboard_by_dataset.csv", index=False)
    overall.to_csv(lb / "leaderboard.csv")
    if not args.no_plots:
        plot_leaderboard(by_dataset, overall, lb / "leaderboard.png")

    key = ["crps_mean", "energy_score", "rmse_mean", "mean_abs_z", "cov1d_ci90", "cov2d_hpd90",
           "in_joint90", "contraction_mean", "entropy_reduction_1d_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print("Per dataset:")
        print(by_dataset.set_index(["method", "dataset"])[["n_samples"] + key].to_string())
        print("\nLeaderboard (complete submissions first, then by mean CRPS, lower is better):")
        print(overall[["rank", "n_datasets_graded"] + key].to_string())
    if overall.n_datasets_graded.nunique() > 1:
        print("\n[warn] methods graded on different numbers of datasets; incomplete ones rank last "
              "and their averages are not directly comparable")
    print(f"\n-> {lb}")


if __name__ == "__main__":
    main()
