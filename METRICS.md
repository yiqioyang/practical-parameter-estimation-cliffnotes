# Grading parameter estimates (`grade.py`)

`grade.py` compares each method's posterior samples with the true parameters (`x_true.csv`)
and ranks methods on a leaderboard.

## 1. How to run

```bash
cd datasets
python grade.py ds_Sep2_2026                 # use all samples in each results file
python grade.py ds_Sep2_2026 --n_sample 100  # use only the first 100 samples (short: -n 100)
python grade.py ds_Sep2_2026 --no_plots      # tables only (faster)
```

Other options: `--prior_low` / `--prior_high` (default −1 and 1), the parameter range used for
contraction and entropy.

A full run on `ds_Sep2_2026` takes about 30 s on a login node. No PBS job is needed.

### Why `--n_sample`?
Methods can return different numbers of samples (e.g. 342 vs 1000). Some metrics depend on
sample count. The min–max range, for example, always grows with more samples.
`--n_sample N` takes the **first N rows** of every results file so all methods are compared on
equal footing. A file with fewer than N rows uses all its rows and prints a warning.

## 2. What the grader expects

```
ds_Sep2_2026/
├── <dataset>/x_true.csv                 # any folder NOT starting with "Results" that has x_true.csv
├── Results_<who>/                       # one folder per competitor
│   └── results_<dataset>.csv            # must match the dataset folder name exactly
└── Results_<who>/                       # ... or, when one person submits several methods:
    └── <method>/                        # one subfolder per method, each graded on its own
        └── results_<dataset>.csv
```

A competitor who tried several methods puts each one in its own subfolder; the grader finds them
and ranks each as a separate entry named `Results_<who>/<method>`. The two layouts can be mixed,
both across folders and inside one folder (loose results files *and* method subfolders).

**Results file format:** CSV, first column is a row index, then one column per parameter
named as in `x_true.csv` (`x0 … x19`). Each row is one posterior sample, and all rows are equally
weighted. Extra columns are ignored. A missing file means that dataset is skipped for that
method.

## 3. Outputs

Every run writes into a folder tagged by the sample setting (`n_all`, `n_100`, …), so runs
with different settings do not overwrite each other.

| File | Content |
|---|---|
| `<submission>/grades/n_<tag>/per_parameter.csv` | One row per (dataset, parameter): 1D metrics |
| `<submission>/grades/n_<tag>/pairwise.csv` | One row per (dataset, parameter pair): 2D metrics |
| `<submission>/grades/n_<tag>/summary.csv` | One row per dataset: aggregated and joint metrics |
| `<submission>/grades/n_<tag>/plots/corner_<dataset>.png` | Corner plot (see §6) |
| `<submission>/grades/n_<tag>/plots/metrics_overview.png` | Per-parameter CRPS, contraction, z-score, rank |
| `leaderboard/n_<tag>/leaderboard_by_dataset.csv` | All methods' `summary.csv` rows combined |
| `leaderboard/n_<tag>/leaderboard.csv` | One row per method, averaged over datasets, ranked |
| `leaderboard/n_<tag>/leaderboard.png` | Key metrics per dataset, one color per method |

### Ranking rule
1. Methods graded on more datasets rank ahead (an incomplete submission ranks last).
2. Ties are broken by **mean CRPS** (lower is better).

CRPS is the primary score because it rewards being accurate *and* sharp while penalizing overconfidence
at the same time (see §4.1). The other metrics explain *why* a method scores the way it does.

---

## 4. Metrics

Notation: for one parameter, `s` = the samples, `t` = the true value, `n` = number of samples.
Prior = Uniform[−1, 1].

**Direction key:** ↓ lower is better · ↑ higher is better · ◎ closer to a target is better ·
ℹ informational (no better/worse on its own)

### 4.1 Headline scores

| Metric | Where | Direction | Meaning |
|---|---|---|---|
| `crps` / `crps_mean` | per-parameter / summary | ↓ (0 = perfect) | **Continuous Ranked Probability Score.** `E|X − t| − ½·E|X − X′|`. The first term is the distance from the samples to the truth; the second is a bonus for spread. A posterior that sits tightly on the truth scores 0. A posterior that is tight but in the wrong place scores badly. A posterior that is very wide scores in between. Same units as the parameter. |
| `energy_score` | summary | ↓ (0 = perfect) | The multivariate version of CRPS on the full 20-D vector: `E‖X − t‖ − ½·E‖X − X′‖`. Unlike averaging CRPS, it also rewards getting the **correlations** between parameters right. |

### 4.2 Point accuracy

| Metric | Where | Direction | Meaning |
|---|---|---|---|
| `mean`, `median`, `std` | per-parameter | ℹ | Posterior summary statistics. |
| `bias` | per-parameter | ◎ 0 | `mean − t`. The sign shows the direction of the error. |
| `abs_err_mean`, `abs_err_median` | per-parameter | ↓ | `|mean − t|`, `|median − t|`. |
| `rmse_mean` | summary | ↓ | Root-mean-square of `bias` over all parameters. |
| `mae_median` | summary | ↓ | Mean of `abs_err_median` over all parameters. |
| `z_score` | per-parameter | ◎ 0 | `(mean − t) / std`: the error measured in posterior standard deviations. **|z| > 2 means "confidently wrong"**: the posterior claims to be precise but misses. |
| `mean_abs_z`, `max_abs_z` | summary | ↓ | Average and worst |z| over parameters. |
| `frac_abs_z_gt2` | summary | ↓ | Fraction of parameters that are confidently wrong. |

### 4.3 Coverage: is the truth inside the estimate?

For a well-calibrated method, a 90% interval should contain the truth about 90% of the time.
So coverage is **◎ closest to the nominal level**, not "higher is better":

- **Coverage above nominal** (e.g. 1.0 for a 90% interval) means the posterior may be *too wide*
  (under-confident). A posterior equal to the prior always gets 1.0, which is why coverage must be
  read together with CRPS, contraction and entropy.
- **Coverage below nominal** means the posterior is *too narrow* (overconfident).

> **Noise warning:** each dataset has only one true vector. With 20 parameters, 1D coverage moves in
> steps of 0.05, so treat differences of ±0.1 as noise. The pairwise and cross-dataset averages are
> more stable.

| Metric | Where | Direction | Meaning |
|---|---|---|---|
| `in_minmax` / `cov1d_minmax` | per-param / summary | ℹ | Truth within [min, max] of samples. Weak test: it grows with sample count. |
| `in_ci50`, `in_ci90`, `in_ci95` | per-parameter | ℹ | Truth within the central 50/90/95% quantile interval. |
| `cov1d_ci50`, `cov1d_ci90`, `cov1d_ci95` | summary | ◎ 0.5 / 0.9 / 0.95 | Fraction of parameters whose interval contains the truth. |
| `percentile_rank` | per-parameter | ◎ not near 0 or 1 | Fraction of samples below the truth. Near 0 or 1 means the truth is in the tail. For a calibrated method, ranks across parameters look uniform. |
| `rank_ks` | summary | ↓ | Kolmogorov–Smirnov distance between the 20 percentile ranks and Uniform(0, 1). 0 means perfectly uniform. Noisy with 20 values. |
| `hpd_level` | pairwise | ↓ | For a pair (x_i, x_j): the probability mass of the smallest high-density region that still contains the truth. 0.1 means the truth is near the posterior's peak. 0.97 means it is in the outer tail. Density comes from a 2D Gaussian KDE, which handles **non-Gaussian** shapes such as banana or multi-modal. |
| `in_hpd50`, `in_hpd90`, `in_hpd95` | pairwise | ℹ | `hpd_level ≤ 0.5 / 0.9 / 0.95`. |
| `cov2d_hpd50`, `cov2d_hpd90`, `cov2d_hpd95` | summary | ◎ 0.5 / 0.9 / 0.95 | Fraction of the 190 pairs whose HPD region contains the truth. |
| `joint_level_mahalanobis` | summary | ↓ | The same idea in all 20 dimensions: the fraction of samples closer to the posterior mean (Mahalanobis distance) than the truth. The level is taken from the samples themselves rather than a χ² table, which reduces the Gaussian assumption. |
| `in_joint50`, `in_joint90`, `in_joint95` | summary | ℹ (True is good) | `joint_level_mahalanobis ≤ 0.5 / 0.9 / 0.95`. In the leaderboard this becomes the fraction of datasets where the result is True. |

### 4.4 Sharpness: how much did the method learn?

These measure how narrow the posterior is **compared with the prior**. Sharper is only better if the
truth is still covered (§4.3) and CRPS is low (§4.1). A sharp but wrong posterior shows high
contraction *and* large |z|.

| Metric | Where | Direction | Meaning |
|---|---|---|---|
| `contraction` | per-parameter | ↓ (if coverage OK) | `std / prior_std`, with prior std = 2/√12 ≈ 0.577. **1 means nothing was learned** about this parameter. 0.2 means the spread was cut to 20% of the prior. Values slightly above 1 are possible when samples pile up at both bounds. |
| `contraction_mean` | summary | ↓ (if coverage OK) | Average over parameters. |
| `entropy` | per-parameter | ℹ | Differential entropy of the 1D posterior (nats), estimated with a leave-one-out KDE that is reflected at the prior bounds. It handles repeated samples such as MCMC rejections. The prior's entropy is log 2 ≈ 0.693. |
| `entropy_reduction` | per-parameter | ↑ (if coverage OK) | `log 2 − entropy`: information gained over the prior, in nats. 0 means nothing was learned. Unlike std, it also captures non-Gaussian narrowing, e.g. a posterior with two sharp modes. |
| `entropy_reduction_1d_mean` | summary | ↑ (if coverage OK) | Average over parameters. |
| `joint_entropy_gauss` | summary | ℹ | Entropy of a Gaussian with the same 20×20 covariance as the samples. It is an **upper bound** on the true joint entropy, since the Gaussian has the most entropy for a given covariance. |
| `joint_entropy_reduction_gauss` | summary | ↑ (if coverage OK) | `20·log 2 − joint_entropy_gauss`. It is a lower bound on the information gained, and it includes what was learned from **correlations** between parameters. |

### 4.5 Bookkeeping

| Column | Meaning |
|---|---|
| `n_samples` | Samples actually used (after `--n_sample`). |
| `n_params` | Number of parameters graded. |
| `n_datasets_graded` | (leaderboard) Datasets the method submitted results for. |
| `degenerate` / `frac_degenerate` | Per-parameter / fraction of parameters where the posterior collapsed to a single repeated value (spread below 1e-9 of the prior width). Grid and brute-force searches do this. Such a parameter has no density to estimate, so `entropy` and `z_score` are left empty; CRPS, RMSE and coverage still score it, and coverage will fail unless the point is exactly right. A high `frac_degenerate` means the leaderboard's `mean_abs_z` and entropy columns rest on fewer parameters for that method. |

---

## 5. Reading the metrics together

| Pattern | Interpretation |
|---|---|
| Low CRPS, coverage ≈ nominal, high entropy reduction | Accurate, calibrated and informative. The goal. |
| Coverage = 1.0, contraction ≈ 1, entropy reduction ≈ 0 | The method returned roughly the prior. Safe but uninformative. |
| High contraction (small), large |z|, low coverage | Overconfident: sharp but in the wrong place. |
| Low `rmse_mean` but high CRPS | The point estimate is good, but the uncertainty is badly sized. |
| Average CRPS fine but `energy_score` relatively poor | Marginals are fine, but the correlations between parameters are wrong. |
| Pairwise coverage much lower than 1D coverage | Each parameter looks fine alone, but the joint shape (correlations, curvature) misses the truth. |

## 6. Plots

- **`corner_<dataset>.png`**
  - *Lower triangle:* scatter of samples for each pair. Black lines cross at the truth.
  - *Diagonal:* histogram of each parameter. The gray band is the central 90% interval and the black line is the truth.
  - *Upper triangle:* `hpd_level` for that pair (darker = truth further in the tail). An `×` marks
    pairs where the truth is outside the 90% region.
- **`metrics_overview.png`**: one column per dataset. Rows show CRPS, contraction (dashed line = prior),
  z-score (dashed lines = ±2) and percentile rank (dashed lines = 5% / 50% / 95%) for every parameter.
- **`leaderboard.png`**: key summary metrics per dataset, one color per method (up to 8 methods drawn;
  all methods are always in the CSVs). Dashed lines mark the target (0.9 for coverage, 1 for
  contraction).

## 7. Implementation notes

- CRPS uses the exact sample formula (sorted-sample identity), so there is no binning.
- KDE bandwidth is Scott's rule (`scipy.stats.gaussian_kde` default).
- The 2D HPD level uses leave-one-out densities for the samples, so a sample does not count its own kernel.
- A KDE that cannot be built (e.g. perfectly collinear samples for a pair) gives `NaN` for that pair.
  Such pairs are left out of `cov2d_*`.
