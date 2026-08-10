import numpy as np

"""Coefficient generators for the exponential-link model:

    y_k = exp( sum_i coefficients[i, k] * x_i )

Structurally identical to linear.py (single coefficient matrix, same
full/diag/triu sparsity options) -- the only difference is `scale`, which
shrinks the raw N(0,1) draw before use. Summing n_params unscaled N(0,1)
terms and exponentiating overflows quickly once n_params grows past a
handful, so coefficients need to live on a smaller scale than in the linear
model.
"""


def generate_matrix_full(n_params, n_outputs, n_sensitive_para=0, n_sensitive_output=0, sensitivity_multiplier=2.0, scale=0.5, seed=None):
    """Dense coefficients. Returns (coefficients, paras_to_vary, outputs_to_vary)."""
    rng = np.random.default_rng(seed)
    coefficients = scale * rng.normal(size=(n_params, n_outputs))

    paras_to_vary, outputs_to_vary = [], []
    if n_sensitive_para > 0 and n_sensitive_output > 0:
        paras_to_vary = rng.choice(n_params, size=n_sensitive_para, replace=False).tolist()
        outputs_to_vary = rng.choice(n_outputs, size=n_sensitive_output, replace=False).tolist()
        for i in outputs_to_vary:
            coefficients[paras_to_vary, i] *= sensitivity_multiplier

    return coefficients, paras_to_vary, outputs_to_vary


def generate_matrix_diag(n_params, n_outputs, n_sensitive_para=0, window_size=2, sensitivity_multiplier=2.0, scale=0.5, seed=None):
    """Banded coefficients (see linear.generate_matrix_diag). Returns (coefficients, paras_to_vary)."""
    rng = np.random.default_rng(seed)
    coefficients = np.zeros((n_params, n_outputs))

    max_windows = max(0, n_params - window_size + 1)
    n_windowed = min(n_outputs, max_windows)

    for j in range(n_windowed):
        coefficients[j:j + window_size, j] = scale * rng.normal(size=window_size)
    for j in range(n_windowed, n_outputs):
        coefficients[:, j] = scale * rng.normal(size=n_params)

    paras_to_vary = []
    if n_sensitive_para > 0:
        paras_to_vary = rng.choice(n_windowed, size=n_sensitive_para, replace=False).tolist()
        coefficients[paras_to_vary, :n_windowed] *= sensitivity_multiplier

    return coefficients, paras_to_vary


def generate_matrix_triu(n_params, n_outputs, n_sensitive_para, n_sensitive_outputs, scale=0.5, seed=None):
    """Sparsity-masked coefficients (see linear.generate_matrix_triu). Returns coefficients."""
    rng = np.random.default_rng(seed)

    mask_a = np.zeros((n_params, n_outputs))
    mask_a[:n_sensitive_para, :] = 1
    mask_b = np.triu(np.ones((n_params, n_outputs)), n_sensitive_outputs - n_sensitive_para)
    mask = ((mask_a + mask_b) > 0).astype(int)

    coefficients = mask * (scale * rng.normal(size=(n_params, n_outputs)))
    return coefficients


def generate_matrix_blocks(n_params, n_outputs, param_group_sizes, output_group_sizes,
                            sensitivity_multiplier=2.0, background_scale=0.0, scale=0.5, seed=None):
    """Block-diagonal sensitivity structure (see linear.generate_matrix_blocks):
    param group g exclusively (if background_scale=0) drives output group g.
    Same `scale` shrinkage as the other exponential.py generators, applied on
    top of both the background and within-block coefficient draws.

    Returns (coefficients, param_group_id, output_group_id)."""
    if sum(param_group_sizes) != n_params:
        raise ValueError(f"param_group_sizes must sum to n_params ({n_params}), got {sum(param_group_sizes)}")
    if sum(output_group_sizes) != n_outputs:
        raise ValueError(f"output_group_sizes must sum to n_outputs ({n_outputs}), got {sum(output_group_sizes)}")
    if len(param_group_sizes) != len(output_group_sizes):
        raise ValueError("param_group_sizes and output_group_sizes must have the same number of groups")

    rng = np.random.default_rng(seed)
    n_groups = len(param_group_sizes)
    coefficients = scale * background_scale * rng.normal(size=(n_params, n_outputs))

    param_group_id = np.repeat(np.arange(n_groups), param_group_sizes)
    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    param_starts = np.concatenate([[0], np.cumsum(param_group_sizes)])
    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])

    for g in range(n_groups):
        p0, p1 = param_starts[g], param_starts[g + 1]
        o0, o1 = output_starts[g], output_starts[g + 1]
        coefficients[p0:p1, o0:o1] = scale * sensitivity_multiplier * rng.normal(size=(p1 - p0, o1 - o0))

    return coefficients, param_group_id, output_group_id


def generate_matrix_overlapping_blocks(n_params, n_outputs, param_groups, output_group_sizes,
                                        sensitivity_multiplier=2.0, extra_links=None,
                                        shared_scale=1.0, background_scale=0.0, scale=0.5, seed=None):
    """Overlapping block-diagonal sensitivity structure (see
    linear.generate_matrix_overlapping_blocks) -- param groups may share
    parameters across output groups, and `extra_links` (optional list of
    (source_group_idx, target_group_idx) pairs) additionally places
    source_group_idx's params into target_group_idx's output block on the plain
    `shared_scale`, instead of the amplified `sensitivity_multiplier`. Same
    `scale` shrinkage as the other exponential.py generators, applied on top of
    every coefficient draw.

    Returns (coefficients, param_group_membership, output_group_id)."""
    if sum(output_group_sizes) != n_outputs:
        raise ValueError(f"output_group_sizes must sum to n_outputs ({n_outputs}), got {sum(output_group_sizes)}")
    if len(param_groups) != len(output_group_sizes):
        raise ValueError("param_groups and output_group_sizes must have the same number of groups")
    for g, idx in enumerate(param_groups):
        if any(i < 0 or i >= n_params for i in idx):
            raise ValueError(f"param_groups[{g}] contains a parameter index outside [0, {n_params})")
    n_groups = len(param_groups)

    normalized_links = []
    for source, target in (extra_links or []):
        if not (-n_groups <= source < n_groups):
            raise ValueError(f"extra_links source={source} out of range for {n_groups} groups")
        if not (-n_groups <= target < n_groups):
            raise ValueError(f"extra_links target={target} out of range for {n_groups} groups")
        normalized_links.append((source % n_groups, target % n_groups))

    rng = np.random.default_rng(seed)
    coefficients = scale * background_scale * rng.normal(size=(n_params, n_outputs))

    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])
    param_group_membership = np.zeros((n_params, n_groups), dtype=bool)

    for g in range(n_groups):
        idx = list(param_groups[g])
        param_group_membership[idx, g] = True
        o0, o1 = output_starts[g], output_starts[g + 1]
        coefficients[idx, o0:o1] = scale * sensitivity_multiplier * rng.normal(size=(len(idx), o1 - o0))

    for source, target in normalized_links:
        idx = list(param_groups[source])
        o0, o1 = output_starts[target], output_starts[target + 1]
        coefficients[idx, o0:o1] = scale * shared_scale * rng.normal(size=(len(idx), o1 - o0))
        param_group_membership[idx, target] = True

    return coefficients, param_group_membership, output_group_id


def compute_Y(X, coefficients):
    """y = exp(X @ coefficients)."""
    return np.exp(X @ coefficients)
