import numpy as np
import random

def sample_X(n_samples, n_params, low=-1.0, high=1.0, seed=None):
    """Draws the input matrix X ~ Uniform(low, high), shape (n_samples, n_params)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n_samples, n_params))


def generate_matrix_full(n_params, n_outputs, n_sensitive_para=0, n_sensitive_output=0, sensitivity_multiplier=2.0, seed = None):
    rng = np.random.default_rng(seed)   
    coefficients = rng.normal(size=(n_params, n_outputs))


    if (n_sensitive_para > 0) & (n_sensitive_output > 0):
        paras_to_vary = random.sample(range(n_params), n_sensitive_para)
        outputs_to_vary = random.sample(range(n_outputs), n_sensitive_output)


        for i in outputs_to_vary:
            coefficients[paras_to_vary, i] = coefficients[paras_to_vary, i] * sensitivity_multiplier

        return coefficients, paras_to_vary, outputs_to_vary
    else:
        return coefficients, [], []


def generate_matrix_diag(n_params, n_outputs, n_sensitive_para=0, window_size = 2, sensitivity_multiplier=2.0, seed = None):
    rng = np.random.default_rng(seed)
    coefficients = np.zeros((n_params, n_outputs))

    max_windows = max(0, n_params - window_size + 1)
    n_windowed = min(n_outputs, max_windows)

    for j in range(n_windowed):
        coefficients[j:j + window_size, j] = rng.normal(size=window_size)
    for j in range(n_windowed, n_outputs):
        coefficients[:, j] = rng.normal(size=n_params)

    if n_sensitive_para > 0:
        paras_to_vary = random.sample(range(n_windowed), n_sensitive_para)
        coefficients[paras_to_vary, :n_windowed] = coefficients[paras_to_vary,:n_windowed] * sensitivity_multiplier
        
        return coefficients, paras_to_vary
    else:
        return coefficients, []



def generate_matrix_triu(n_params, n_outputs, n_sensitive_para, n_sensitive_outputs, seed = None):
    rng = np.random.default_rng(seed)   
    coefficients_I_a = np.zeros((n_params, n_outputs))
    coefficients = rng.normal(size=(n_params, n_outputs))

    coefficients_I_b = np.ones((n_params, n_outputs))
    coefficients_I_b = np.triu(coefficients_I_b, (n_sensitive_outputs - n_sensitive_para ))

    for i in range(n_params):
        if i < n_sensitive_para:
            coefficients_I_a[i, :] = 1

    coefficients_I = ((coefficients_I_a + coefficients_I_b) > 0).astype(int)

    return coefficients_I * coefficients


def generate_matrix_blocks(n_params, n_outputs, param_group_sizes, output_group_sizes,
                            sensitivity_multiplier=2.0, background_scale=0.0, seed=None):
    """Block-diagonal sensitivity structure: partitions the n_params inputs into
    contiguous groups of sizes `param_group_sizes` and the n_outputs outputs into
    contiguous groups of sizes `output_group_sizes` (both must sum to n_params /
    n_outputs and have the same number of groups) -- group g's params are the
    (sole, if background_scale=0) drivers of group g's outputs, e.g.
    param_group_sizes=[2, 2], output_group_sizes=[3, 4] means params 0-1 drive
    outputs 0-2 and params 2-3 drive outputs 3-6, with no other coupling.

    background_scale (default 0.0, i.e. exact block-diagonal sparsity) optionally
    adds small cross-block coefficients on that scale so blocks aren't perfectly
    independent; within-block coefficients are drawn on `sensitivity_multiplier`'s
    scale, same amplification convention as the other structure functions here.

    Returns (coefficients, param_group_id, output_group_id): the last two are
    arrays of shape (n_params,) / (n_outputs,) giving each param's/output's group
    index, for visualization or downstream group-aware masking.
    """
    if sum(param_group_sizes) != n_params:
        raise ValueError(f"param_group_sizes must sum to n_params ({n_params}), got {sum(param_group_sizes)}")
    if sum(output_group_sizes) != n_outputs:
        raise ValueError(f"output_group_sizes must sum to n_outputs ({n_outputs}), got {sum(output_group_sizes)}")
    if len(param_group_sizes) != len(output_group_sizes):
        raise ValueError("param_group_sizes and output_group_sizes must have the same number of groups")

    rng = np.random.default_rng(seed)
    n_groups = len(param_group_sizes)
    coefficients = background_scale * rng.normal(size=(n_params, n_outputs))

    param_group_id = np.repeat(np.arange(n_groups), param_group_sizes)
    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    param_starts = np.concatenate([[0], np.cumsum(param_group_sizes)])
    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])

    for g in range(n_groups):
        p0, p1 = param_starts[g], param_starts[g + 1]
        o0, o1 = output_starts[g], output_starts[g + 1]
        coefficients[p0:p1, o0:o1] = sensitivity_multiplier * rng.normal(size=(p1 - p0, o1 - o0))

    return coefficients, param_group_id, output_group_id


def generate_matrix_overlapping_blocks(n_params, n_outputs, param_groups, output_group_sizes,
                                        sensitivity_multiplier=2.0, extra_links=None,
                                        shared_scale=1.0, background_scale=0.0, seed=None):
    """Like generate_matrix_blocks, but parameter groups may overlap -- a param can
    drive more than one output group. `param_groups` is a list of parameter-index
    lists, one per output group (not necessarily disjoint or contiguous), e.g.
    param_groups=[[0, 1], [1, 2]], output_group_sizes=[4, 6] means params 0-1 drive
    outputs 0-3 and params 1-2 drive outputs 4-9 -- param 1 is shared, driving both
    groups. `output_group_sizes` is still a strict contiguous partition of the
    n_outputs outputs (every output belongs to exactly one group); len(param_groups)
    must equal len(output_group_sizes). Every group's own diagonal block (param_groups[g]
    x its own output block) is always amplified by `sensitivity_multiplier`.

    extra_links (optional): a list of (source_group_idx, target_group_idx) pairs, each
    meaning "source_group_idx's params ALSO get a plain (unamplified, `shared_scale`)
    coefficient block in target_group_idx's output columns", on top of every group's own
    amplified diagonal block. E.g. extra_links=[(0, -1)] with param_groups=[[0,1,2],[1,2],
    [3..9]] means params 0-2 (group 0) additionally drive the LAST group's outputs a
    little (unamplified), in addition to their own dedicated (amplified) block -- negative
    indices work as usual Python indexing into param_groups/output_group_sizes.

    Returns (coefficients, param_group_membership, output_group_id):
    param_group_membership is a (n_params, n_groups) bool array (row i is which groups
    param i has a real, non-background coefficient in -- includes both each group's own
    block and any extra_links insertions); output_group_id is (n_outputs,) int, same as
    generate_matrix_blocks.
    """
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
    coefficients = background_scale * rng.normal(size=(n_params, n_outputs))

    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])
    param_group_membership = np.zeros((n_params, n_groups), dtype=bool)

    for g in range(n_groups):
        idx = list(param_groups[g])
        param_group_membership[idx, g] = True
        o0, o1 = output_starts[g], output_starts[g + 1]
        coefficients[idx, o0:o1] = sensitivity_multiplier * rng.normal(size=(len(idx), o1 - o0))

    for source, target in normalized_links:
        idx = list(param_groups[source])
        o0, o1 = output_starts[target], output_starts[target + 1]
        coefficients[idx, o0:o1] = shared_scale * rng.normal(size=(len(idx), o1 - o0))
        param_group_membership[idx, target] = True

    return coefficients, param_group_membership, output_group_id


def compute_Y(X, coefficients):
    """y = X @ coefficients, shape (n_samples, n_outputs)."""
    return X @ coefficients
