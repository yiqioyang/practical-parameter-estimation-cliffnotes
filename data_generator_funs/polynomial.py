"""Coefficient generator for polynomial models up to degree 2 (quadratic) or 3 (cubic):

    y_k = sum_i b1[i,k]*x_i + sum_i b2[i,k]*x_i**2 + sum_(i,j) c2[(i,j),k]*x_i*x_j
        [+ sum_i b3[i,k]*x_i**3 + sum_(i,j,l) c3[(i,j,l),k]*x_i*x_j*x_l]   -- degree == 3 only

Sensitivity is nested: `n_sensitive_para` individual parameters are marked
sensitive first; `n_sensitive_pairs` pairs and `n_sensitive_triplets` triplets
are then drawn *only* from combinations of those sensitive parameters (so a
sensitive pair is always built from two sensitive params). `n_pairs` /
`n_triplets` are separate, non-sensitive background interaction terms drawn
from the remaining combinations -- the model ends up with
`n_pairs + n_sensitive_pairs` pairs total (and likewise for triplets). Every
sensitive coefficient (individual, pair, or triplet) is multiplied by
`sensitivity_multiplier`.
"""

import itertools

import numpy as np


def _draw_combinations(rng, n_params, order, sensitive_pool, n_background, n_sensitive, sensitivity_multiplier, n_outputs):
    """Pick n_sensitive combinations of size `order` from sensitive_pool, plus
    n_background more from the remaining combinations. Returns (combos, coefficients),
    with the sensitive combos first (and already amplified)."""
    sensitive_combos = list(itertools.combinations(sorted(sensitive_pool), order))
    if n_sensitive > len(sensitive_combos):
        raise ValueError(
            f"n_sensitive={n_sensitive} order-{order} combinations requested but only "
            f"{len(sensitive_combos)} are possible from {len(sensitive_pool)} sensitive parameters"
        )
    chosen_sensitive = (
        [sensitive_combos[i] for i in rng.choice(len(sensitive_combos), size=n_sensitive, replace=False)]
        if n_sensitive > 0 else []
    )

    all_combos = list(itertools.combinations(range(n_params), order))
    chosen_sensitive_set = set(chosen_sensitive)
    remaining = [c for c in all_combos if c not in chosen_sensitive_set]
    if n_background > len(remaining):
        raise ValueError(
            f"n_background={n_background} order-{order} combinations requested but only "
            f"{len(remaining)} remain after reserving {len(chosen_sensitive)} sensitive ones"
        )
    chosen_background = (
        [remaining[i] for i in rng.choice(len(remaining), size=n_background, replace=False)]
        if n_background > 0 else []
    )

    combos = chosen_sensitive + chosen_background
    coefficients = rng.normal(size=(len(combos), n_outputs))
    if chosen_sensitive:
        coefficients[:len(chosen_sensitive)] *= sensitivity_multiplier

    return combos, coefficients


def generate_matrix_full(n_params, n_outputs, degree=2,
                          n_sensitive_para=0, sensitivity_multiplier=2.0,
                          n_pairs=0, n_sensitive_pairs=0,
                          n_triplets=0, n_sensitive_triplets=0,
                          seed=None):
    """Dense polynomial coefficients up to `degree` (2 or 3). See module docstring
    for the model and the nested-sensitivity semantics.

    Returns a dict:
        coefficients_linear, coefficients_quad: (n_params, n_outputs)
        pairs: list of (i, j) tuples -- sensitive ones first
        coefficients_pairs: (len(pairs), n_outputs)
        paras_to_vary: sensitive parameter indices
        [degree == 3 only]
        coefficients_cubic: (n_params, n_outputs)
        triplets: list of (i, j, l) tuples -- sensitive ones first
        coefficients_triplets: (len(triplets), n_outputs)
    """
    if degree not in (2, 3):
        raise ValueError(f"degree must be 2 or 3, got {degree}")
    if degree == 2 and (n_triplets > 0 or n_sensitive_triplets > 0):
        raise ValueError("triplets require degree=3")

    rng = np.random.default_rng(seed)

    coefficients_linear = rng.normal(size=(n_params, n_outputs))
    coefficients_quad = rng.normal(size=(n_params, n_outputs))
    coefficients_cubic = rng.normal(size=(n_params, n_outputs)) if degree == 3 else None

    paras_to_vary = []
    if n_sensitive_para > 0:
        paras_to_vary = rng.choice(n_params, size=n_sensitive_para, replace=False).tolist()
        coefficients_linear[paras_to_vary, :] *= sensitivity_multiplier
        coefficients_quad[paras_to_vary, :] *= sensitivity_multiplier
        if degree == 3:
            coefficients_cubic[paras_to_vary, :] *= sensitivity_multiplier

    pairs, coefficients_pairs = _draw_combinations(
        rng, n_params, 2, paras_to_vary, n_pairs, n_sensitive_pairs, sensitivity_multiplier, n_outputs
    )

    result = {
        "coefficients_linear": coefficients_linear,
        "coefficients_quad": coefficients_quad,
        "pairs": pairs,
        "coefficients_pairs": coefficients_pairs,
        "paras_to_vary": paras_to_vary,
    }

    if degree == 3:
        triplets, coefficients_triplets = _draw_combinations(
            rng, n_params, 3, paras_to_vary, n_triplets, n_sensitive_triplets, sensitivity_multiplier, n_outputs
        )
        result["coefficients_cubic"] = coefficients_cubic
        result["triplets"] = triplets
        result["coefficients_triplets"] = coefficients_triplets

    return result


def compute_Y(X, coefficients_linear, coefficients_quad, pairs, coefficients_pairs,
              coefficients_cubic=None, triplets=None, coefficients_triplets=None):
    """y = linear + quadratic + pairwise terms, plus cubic + triplet terms if given."""
    Y = X @ coefficients_linear + (X ** 2) @ coefficients_quad

    if pairs:
        X_pairs = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
        Y = Y + X_pairs @ coefficients_pairs

    if coefficients_cubic is not None:
        Y = Y + (X ** 3) @ coefficients_cubic

    if triplets:
        X_triplets = np.column_stack([X[:, i] * X[:, j] * X[:, l] for i, j, l in triplets])
        Y = Y + X_triplets @ coefficients_triplets

    return Y
