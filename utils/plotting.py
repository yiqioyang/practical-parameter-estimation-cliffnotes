import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BAR_COLOR = "#2a78d6"
_HIGHLIGHT = "#4a3aa7"


def plot_correlation(X, Y, sensitive_param_idx=None, sensitive_output_idx=None):
    """Heatmap of the correlation between each column of Y and each column of X.
    No axis tick labels and no per-cell numeric labels -- color only.
    sensitive_param_idx / sensitive_output_idx (if given) outline the rows/
    columns the generator marked as sensitive, so you can see whether the
    amplified coefficients actually show up as stronger correlation.
    """
    n_params = X.shape[1]
    combined = np.hstack([X, Y])
    corr = np.corrcoef(combined, rowvar=False)
    corr_yx = corr[n_params:, :n_params]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(corr_yx, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", color=_INK_SECONDARY)
    ax.set_ylabel("y", color=_INK_SECONDARY)
    ax.set_title("Correlation between y's and x's", color=_INK_PRIMARY)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")

    if sensitive_param_idx is not None and len(sensitive_param_idx) > 0:
        for j in sensitive_param_idx:
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, corr_yx.shape[0],
                                    fill=False, edgecolor=_HIGHLIGHT, linewidth=1.5))
    if sensitive_output_idx is not None and len(sensitive_output_idx) > 0:
        for i in sensitive_output_idx:
            ax.add_patch(Rectangle((-0.5, i - 0.5), corr_yx.shape[1], 1,
                                    fill=False, edgecolor=_HIGHLIGHT, linewidth=1.5))

    fig.tight_layout()
    plt.show()
    return corr_yx


def sparsity_per_output(coefficients, tol=1e-8):
    """Fraction of near-zero coefficient entries per output/state column.

    `coefficients` is the dict stored on a dataset (e.g. {"coefficients": B}
    for linear/exponential, {"coefficients_linear": B1, "coefficients_quad": B2}
    for quadratic, {"A": A} for the linear ODE, {"forcing": f} for lorenz96)
    -- all arrays in the dict are pooled together. The last axis of every
    array is treated as the output/state axis; 1-D arrays (e.g. lorenz96's
    per-state forcing) count as a single row each.
    """
    arrays = list(coefficients.values())
    n_outputs = arrays[0].shape[-1]
    zero_counts = np.zeros(n_outputs)
    total = 0
    for arr in arrays:
        if arr.ndim == 1:
            zero_counts += (np.abs(arr) < tol).astype(float)
            total += 1
        else:
            zero_counts += (np.abs(arr) < tol).sum(axis=0)
            total += arr.shape[0]
    return zero_counts / total


def nonlinearity_per_output(X, Y_true):
    """1 - R^2 of the best linear (OLS) fit of Y_true from X, per output column.

    Model-agnostic nonlinearity proxy: ~0 for a linear generator, higher for
    quadratic/exponential generators in proportion to how much of the output's
    variance a straight line can't capture.
    """
    n_samples = X.shape[0]
    X_design = np.hstack([np.ones((n_samples, 1)), X])
    beta, *_ = np.linalg.lstsq(X_design, Y_true, rcond=None)
    Y_hat = X_design @ beta

    ss_res = ((Y_true - Y_hat) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)

    return np.clip(ss_res / ss_tot, 0, 1)


def interaction_strength_per_output(X, Y_true, additive_degree=2):
    """1 - R^2 of the best fully-additive (per-parameter, no cross terms)
    polynomial fit of Y_true from X, per output column.

    Unlike nonlinearity_per_output -- which a straight line can't capture at
    all, curved-but-still-independent included -- this only credits an
    output as "needing interaction" if no amount of per-parameter curvature
    can explain it, i.e. it genuinely requires cross terms like x_i * x_j
    (interacting parameters, or coupled ODE states). ~0 for linear/diagonal
    generators, positive once pairwise/triplet terms (or ODE coupling) are
    introduced.
    """
    n_samples, n_params = X.shape
    basis = [np.ones((n_samples, 1))] + [X ** d for d in range(1, additive_degree + 1)]
    X_design = np.hstack(basis)

    beta, *_ = np.linalg.lstsq(X_design, Y_true, rcond=None)
    Y_hat = X_design @ beta

    ss_res = ((Y_true - Y_hat) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)

    return np.clip(ss_res / ss_tot, 0, 1)


def plot_data_overview(dataset):
    """Bar charts of per-output coefficient sparsity, nonlinearity, and
    interaction strength -- the "how sparse / how nonlinear / how much do
    parameters interact" summary view for a generated dataset dict."""
    X = dataset["X"]
    Y = dataset["Y"]
    coefficients = dataset["coefficients"]
    meta = dataset.get("metadata", {})

    sparsity = sparsity_per_output(coefficients)
    if "n_states" in meta and "t_eval" in meta:
        # ODE/lorenz96: coefficients are per-state (n_states columns) but Y
        # is flattened time-major (n_states * n_times columns) -- the same
        # (time-invariant) coefficients apply at every observed time, so tile.
        sparsity = np.tile(sparsity, len(meta["t_eval"]))
    nonlinearity = nonlinearity_per_output(X, Y)
    interaction = interaction_strength_per_output(X, Y)

    n_outputs = Y.shape[1]
    idx = np.arange(n_outputs)

    fig, axes = plt.subplots(3, 1, figsize=(max(6, n_outputs * 0.35), 8.5), sharex=True)
    fig.patch.set_facecolor(_SURFACE)

    panels = [
        (sparsity, "Coefficient sparsity per output", "fraction zero"),
        (nonlinearity, "Nonlinearity per output (1 - R² of best linear fit)", "1 - R²"),
        (interaction, "Interaction strength per output (1 - R² of best additive fit)", "1 - R²"),
    ]
    for ax, (values, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8)
        ax.bar(idx, values, color=_BAR_COLOR, width=0.7, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_title(title, color=_INK_PRIMARY, loc="left", fontsize=11)
        ax.set_ylabel(ylabel, color=_INK_MUTED, fontsize=9)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("output index", color=_INK_MUTED, fontsize=9)
    axes[-1].set_xticks(idx)
    fig.tight_layout()
    plt.show()
    return sparsity, nonlinearity, interaction


def plot_scatter_examples(dataset, n_examples=3, seed=None):
    """Small-multiples scatter of Y vs. its most correlated X column, for
    a few outputs -- a quick visual check of the generator's shape (roughly
    straight for linear, curved for quadratic, hockey-stick for exponential).
    Prefers the generator's marked "sensitive" outputs if any were set."""
    X = dataset["X"]
    Y = dataset["Y"]
    n_outputs = Y.shape[1]

    outputs_to_vary = dataset.get("outputs_to_vary")
    if outputs_to_vary is not None and len(outputs_to_vary) > 0:
        chosen = list(outputs_to_vary[:n_examples])
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n_outputs, size=min(n_examples, n_outputs), replace=False).tolist()

    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 3.5))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(_SURFACE)

    for ax, k in zip(axes, chosen):
        corrs = [np.corrcoef(X[:, i], Y[:, k])[0, 1] for i in range(X.shape[1])]
        best_param = int(np.argmax(np.abs(corrs)))
        ax.set_facecolor(_SURFACE)
        ax.scatter(X[:, best_param], Y[:, k], s=10, color=_BAR_COLOR, alpha=0.6, edgecolors="none")
        ax.set_title(f"output {k} vs param {best_param}", color=_INK_PRIMARY, fontsize=10, loc="left")
        ax.set_xlabel(f"x[{best_param}]", color=_INK_MUTED, fontsize=9)
        ax.set_ylabel(f"y[{k}]", color=_INK_MUTED, fontsize=9)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_generator_spectrum(datasets):
    """Bar chart comparing several generate_*_dataset outputs side by side on
    two axes computed directly from X/Y (not from generator internals):
    mean nonlinearity (1 - R² of the best linear fit) and mean interaction
    strength (1 - R² of the best per-parameter-additive fit), each averaged
    over the dataset's outputs. The "spectrum from linear to nonlinear to
    interacting to chaotic" summary view for comparing generator families.

    `datasets` is a dict of {label: dataset}. Returns {label: (nonlinearity, interaction)}.
    """
    names = list(datasets.keys())
    nonlin = [nonlinearity_per_output(d["X"], d["Y"]).mean() for d in datasets.values()]
    inter = [interaction_strength_per_output(d["X"], d["Y"]).mean() for d in datasets.values()]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 4.5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8)
    ax.bar(x - width / 2, nonlin, width, color=_BAR_COLOR, label="nonlinearity (1 - R² linear fit)", zorder=3)
    ax.bar(x + width / 2, inter, width, color=_HIGHLIGHT, label="interaction (1 - R² additive fit)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", color=_INK_MUTED)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean across outputs", color=_INK_MUTED, fontsize=9)
    ax.set_title("Generator spectrum: linear → nonlinear → interacting → chaotic",
                 color=_INK_PRIMARY, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(colors=_INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()
    return dict(zip(names, zip(nonlin, inter)))


def plot_trajectories(dataset, state_idx=0, n_examples=8, seed=None):
    """Line plot of one state's trajectory over t_eval for a few sample rows
    -- the ODE/Lorenz-96-specific diagnostic. (plot_correlation/plot_data_overview/
    plot_scatter_examples all still work on these datasets unchanged, since they
    only look at X/Y/coefficients shapes -- this one adds the time axis those
    don't show.)
    """
    meta = dataset["metadata"]
    t_eval = np.asarray(meta["t_eval"])
    n_states = meta["n_states"]
    n_times = len(t_eval)

    Y = dataset["Y"].reshape(-1, n_times, n_states)
    n_samples = Y.shape[0]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_samples, size=min(n_examples, n_samples), replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for s in chosen:
        ax.plot(t_eval, Y[s, :, state_idx], color=_BAR_COLOR, alpha=0.7, linewidth=1.2)
    ax.set_title(f"state {state_idx} trajectories -- {meta.get('model', '')}",
                 color=_INK_PRIMARY, loc="left", fontsize=11)
    ax.set_xlabel("t", color=_INK_MUTED, fontsize=9)
    ax.set_ylabel(f"x[{state_idx}](t)", color=_INK_MUTED, fontsize=9)
    ax.tick_params(colors=_INK_MUTED)
    ax.grid(color=_GRIDLINE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_block_structure(dataset):
    """Heatmap of |coefficients| for a structure="blocks" linear/exponential
    dataset, with lines marking the block boundaries from param_group_id/
    output_group_id -- the direct visual check that "param group g only (or
    mostly) drives output group g" actually holds in the generated matrix.
    """
    param_group_id = dataset.get("param_group_id")
    output_group_id = dataset.get("output_group_id")
    if param_group_id is None or output_group_id is None:
        raise ValueError(
            "dataset has no param_group_id/output_group_id -- was it generated with structure='blocks'?")

    coefficients = dataset["coefficients"]
    if isinstance(coefficients, dict):
        coefficients = coefficients["coefficients"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(np.abs(coefficients), cmap="Blues", aspect="auto")
    ax.set_xlabel("output", color=_INK_SECONDARY)
    ax.set_ylabel("param", color=_INK_SECONDARY)
    ax.set_title("Block structure: |coefficients|", color=_INK_PRIMARY, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|coefficient|")

    for boundary in np.flatnonzero(np.diff(output_group_id)) + 0.5:
        ax.axvline(boundary, color=_HIGHLIGHT, linewidth=1.2)
    for boundary in np.flatnonzero(np.diff(param_group_id)) + 0.5:
        ax.axhline(boundary, color=_HIGHLIGHT, linewidth=1.2)

    fig.tight_layout()
    plt.show()
    return coefficients


def plot_overlapping_block_structure(dataset):
    """Heatmap of |coefficients| for a structure="overlapping_blocks" dataset (see
    linear.generate_matrix_overlapping_blocks), with output-group boundaries marked
    -- like plot_block_structure, but there's no single param_group_id to draw
    horizontal boundaries from, since a param can belong to more than one group.
    Instead, a left-margin strip shows how many groups each param belongs to
    (param_group_membership.sum(axis=1)): 1 for an ordinary group-exclusive param,
    >=2 for one shared across groups -- those rows should visibly light up in more
    than one output-group column in the main heatmap.
    """
    param_group_membership = dataset.get("param_group_membership")
    output_group_id = dataset.get("output_group_id")
    if param_group_membership is None or output_group_id is None:
        raise ValueError(
            "dataset has no param_group_membership/output_group_id -- "
            "was it generated with structure='overlapping_blocks'?")

    coefficients = dataset["coefficients"]
    if isinstance(coefficients, dict):
        coefficients = coefficients["coefficients"]
    n_groups_shared = np.asarray(param_group_membership).sum(axis=1)

    fig, (ax_strip, ax) = plt.subplots(
        1, 2, figsize=(7.5, 5), gridspec_kw={"width_ratios": [0.5, 10]}, sharey=True)
    fig.patch.set_facecolor(_SURFACE)

    ax.set_facecolor(_SURFACE)
    im = ax.imshow(np.abs(coefficients), cmap="Blues", aspect="auto")
    for boundary in np.flatnonzero(np.diff(output_group_id)) + 0.5:
        ax.axvline(boundary, color=_HIGHLIGHT, linewidth=1.2)
    ax.set_xlabel("output", color=_INK_SECONDARY)
    ax.set_title("Overlapping block structure: |coefficients|", color=_INK_PRIMARY, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|coefficient|")

    ax_strip.set_facecolor(_SURFACE)
    ax_strip.imshow(n_groups_shared[:, None], cmap="Purples", aspect="auto", vmin=0)
    ax_strip.set_xticks([])
    ax_strip.set_ylabel("param", color=_INK_SECONDARY)
    ax_strip.set_title("# groups", color=_INK_MUTED, fontsize=8)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.show()
    return coefficients, n_groups_shared


def plot_conflicting_truth(dataset):
    """For a generate_conflicting_truth_dataset output: heatmap of
    cross_model_mismatch[i, j] -- the normalized RMS gap between model i's true
    observation and what model i would predict at model j's true input. A
    near-zero diagonal (each model reproduces its own truth) with large
    off-diagonal entries (no model's truth satisfies any other model) is the
    visual signature of the conflict: there is no single input that would
    make every block's residual small at once.
    """
    mismatch = dataset["cross_model_mismatch"]
    names = dataset["metadata"]["model_names"]
    n_models = len(names)

    fig, ax = plt.subplots(figsize=(max(4, n_models * 1.2), max(3.5, n_models * 1.0)))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(mismatch, cmap="Reds", vmin=0)
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(names, color=_INK_MUTED)
    ax.set_yticklabels(names, color=_INK_MUTED)
    ax.set_xlabel("true input used (x_true of model j)", color=_INK_SECONDARY, fontsize=9)
    ax.set_ylabel("model i's prediction evaluated", color=_INK_SECONDARY, fontsize=9)
    ax.set_title("Conflicting truth: normalized RMS mismatch", color=_INK_PRIMARY, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized RMS gap")

    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{mismatch[i, j]:.2f}", ha="center", va="center",
                    color=_INK_PRIMARY if mismatch[i, j] < mismatch.max() / 2 else "white", fontsize=9)

    fig.tight_layout()
    plt.show()
    return mismatch


def plot_regime_residuals(dataset, output_idx=0):
    """For a generate_structural_regime_dataset output: fit one OLS model to
    the *pooled* data (as a naive single-regime calibration would) and plot
    its residuals for output `output_idx`, colored by the (normally hidden)
    regime_label. A single correctly-specified model fit to genuinely
    single-regime data would give patternless residuals; here the residuals
    should visibly split by regime -- the visual signature of structural
    error: the misfit is systematic, not random noise, and no single
    parameter vector resolves it.
    """
    X, Y = dataset["X"], dataset["Y"]
    regime_label = dataset["regime_label"]

    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(X_design, Y[:, output_idx], rcond=None)
    residuals = Y[:, output_idx] - X_design @ beta

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for regime, color, label in [(0, _BAR_COLOR, "regime 1"), (1, _HIGHLIGHT, "regime 2")]:
        mask = regime_label == regime
        ax.scatter(np.flatnonzero(mask), residuals[mask], s=12, color=color, alpha=0.7,
                   edgecolors="none", label=label)
    ax.axhline(0, color=_INK_MUTED, linewidth=0.8)
    ax.set_title(f"Pooled OLS residuals, output {output_idx} (colored by hidden regime)",
                 color=_INK_PRIMARY, loc="left", fontsize=11)
    ax.set_xlabel("sample index", color=_INK_MUTED, fontsize=9)
    ax.set_ylabel("residual", color=_INK_MUTED, fontsize=9)
    ax.tick_params(colors=_INK_MUTED)
    ax.legend(frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()
    return residuals
