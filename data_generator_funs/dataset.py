import os

import numpy as np

from . import exponential, linear, lorenz96, ode, polynomial
from .linear import sample_X
from .netcdf_io import default_nc_filename, save_dataset_netcdf  # noqa: F401 (re-exported)

"""Top-level dataset assembly: sample an (n_samples + 1)-row ensemble, generate
coefficients, compute Y, hold out the last row as the single "true" input/
output pair, and package everything that describes one generated dataset into
a single dict (and optionally a single self-documenting NetCDF file -- see
netcdf_io.py).

Every dataset has this shape:
    X, Y          -- the training ensemble: n_samples rows, noise-free (Y is
                     exactly the generating process applied to X -- no
                     observation error).
    x_true        -- one extra sampled row, held out of the ensemble: the
                     input a calibration exercise is trying to recover.
    y_true        -- the generating process applied to x_true, with
                     observation noise/structural bias added -- the single
                     "real" observation a calibration exercise tries to match.
"""


def add_true_observation_error(y_true_row, ensemble_Y, noise_std=0.0, structural_idx=None,
                                structural_error_level=0.0, seed=None):
    """Turns the exact y_true_row (shape (n_outputs,), the noise-free model
    evaluated at x_true) into the single noisy/biased "real" observation a
    calibration exercise tries to match. The training ensemble (X, Y) is
    never touched by this -- it stays exactly noise-free.

    noise_std: std of iid N(0, noise_std) measurement noise added to y_true.
    structural_idx: output indices that additionally get a constant
        structural bias -- these are then "wrong" in a way that a
        correctly-specified model can't average away, unlike observational
        noise.
    structural_error_level: bias size, expressed as a multiple of each
        output's std *within the ensemble* (ensemble_Y) -- a lone point has
        no variance of its own, so the ensemble is the natural reference
        scale for how big a "surprising" bias should be.

    Returns (y_true, structural_bias), both shape (n_outputs,), with
    structural_bias zero outside structural_idx.
    """
    rng = np.random.default_rng(seed)
    y_true = y_true_row.copy()
    n_outputs = y_true.shape[0]

    if noise_std > 0:
        y_true = y_true + rng.normal(scale=noise_std, size=n_outputs)

    structural_bias = np.zeros(n_outputs)
    if structural_idx is not None and len(structural_idx) > 0 and structural_error_level > 0:
        col_std = ensemble_Y[:, list(structural_idx)].std(axis=0)
        bias = structural_error_level * col_std
        structural_bias[list(structural_idx)] = bias
        y_true[list(structural_idx)] += bias

    return y_true, structural_bias


def _package(model, structure, X, Y, x_true, y_true, coefficients, paras_to_vary, outputs_to_vary,
             structural_bias, noise_std, structural_idx, structural_error_level, seed,
             n_samples, n_params, n_outputs, extra_arrays=None, extra_metadata=None):
    dataset = {
        "X": X,
        "Y": Y,
        "x_true": x_true,
        "y_true": y_true,
        "coefficients": coefficients,
        "paras_to_vary": np.array(sorted(paras_to_vary), dtype=int),
        "outputs_to_vary": np.array(sorted(outputs_to_vary), dtype=int),
        "structural_bias": structural_bias,
        "metadata": {
            "model": model,
            "structure": structure,
            "n_samples": n_samples,
            "n_params": n_params,
            "n_outputs": n_outputs,
            "noise_std": noise_std,
            "structural_idx": sorted(structural_idx) if structural_idx else [],
            "structural_error_level": structural_error_level,
            "seed": seed,
        },
    }
    if extra_arrays:
        dataset.update(extra_arrays)
    if extra_metadata:
        dataset["metadata"].update(extra_metadata)
    return dataset


def _abbreviate_kwarg_name(key):
    """n_sensitive_para -> "nsp", window_size -> "ws" -- first letter of each
    underscore-separated word, just enough to tell filenames apart at a glance."""
    return "".join(word[0] for word in key.split("_"))


def _format_kwarg_value(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p").replace("-", "m")


def _structure_kwargs_suffix(structure_kwargs):
    if not structure_kwargs:
        return ""
    parts = [f"{_abbreviate_kwarg_name(k)}{_format_kwarg_value(v)}" for k, v in structure_kwargs.items()]
    return "_" + "_".join(parts)


def generate_linear_dataset(n_samples, n_params, n_outputs, structure="full", structure_kwargs=None,
                             noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                             seed=None, nc_path=None):
    """nc_path, if given, is the *directory* to write into -- the filename is
    derived from the run's shape, e.g. "linear_full_n100_p40_o40.nc"."""
    structure_kwargs = structure_kwargs or {}
    X_all = sample_X(n_samples + 1, n_params, seed=seed)

    if structure == "full":
        coefficients, paras_to_vary, outputs_to_vary = linear.generate_matrix_full(
            n_params, n_outputs, seed=seed, **structure_kwargs)
    elif structure == "diag":
        coefficients, paras_to_vary = linear.generate_matrix_diag(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        outputs_to_vary = []
    elif structure == "triu":
        coefficients = linear.generate_matrix_triu(n_params, n_outputs, seed=seed, **structure_kwargs)
        paras_to_vary, outputs_to_vary = [], []
    else:
        raise ValueError(f"unknown structure: {structure!r}")

    Y_all = linear.compute_Y(X_all, coefficients)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    dataset = _package("linear", structure, X, Y, x_true, y_true, {"coefficients": coefficients},
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_params, n_outputs)
    if nc_path:
        filename = (f"linear_{structure}_n{n_samples}_p{n_params}_o{n_outputs}"
                    f"{_structure_kwargs_suffix(structure_kwargs)}.nc")
        save_dataset_netcdf(dataset, os.path.join(nc_path, filename))
    return dataset


def generate_polynomial_dataset(n_samples, n_params, n_outputs, degree=2,
                                 n_sensitive_para=0, sensitivity_multiplier=2.0,
                                 n_pairs=0, n_sensitive_pairs=0,
                                 n_triplets=0, n_sensitive_triplets=0,
                                 noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                                 seed=None, nc_path=None):
    """degree=2 gives a quadratic model (linear + squared + pairwise terms);
    degree=3 adds cubic + triplet terms. See polynomial.py for the nested
    sensitivity semantics of n_sensitive_para / n_sensitive_pairs / n_sensitive_triplets.
    """
    X_all = sample_X(n_samples + 1, n_params, seed=seed)

    result = polynomial.generate_matrix_full(
        n_params, n_outputs, degree=degree,
        n_sensitive_para=n_sensitive_para, sensitivity_multiplier=sensitivity_multiplier,
        n_pairs=n_pairs, n_sensitive_pairs=n_sensitive_pairs,
        n_triplets=n_triplets, n_sensitive_triplets=n_sensitive_triplets,
        seed=seed,
    )

    Y_all = polynomial.compute_Y(
        X_all, result["coefficients_linear"], result["coefficients_quad"],
        result["pairs"], result["coefficients_pairs"],
        coefficients_cubic=result.get("coefficients_cubic"),
        triplets=result.get("triplets"),
        coefficients_triplets=result.get("coefficients_triplets"),
    )
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    coefficients = {
        "coefficients_linear": result["coefficients_linear"],
        "coefficients_quad": result["coefficients_quad"],
        "coefficients_pairs": result["coefficients_pairs"],
    }
    extra_arrays = {
        "pairs": np.array(result["pairs"], dtype=int) if result["pairs"] else np.zeros((0, 2), dtype=int),
        "triplets": np.zeros((0, 3), dtype=int),
    }
    extra_metadata = {
        "degree": degree, "n_pairs": n_pairs, "n_sensitive_pairs": n_sensitive_pairs,
        "n_triplets": n_triplets, "n_sensitive_triplets": n_sensitive_triplets,
        "sensitivity_multiplier": sensitivity_multiplier,
    }
    if degree == 3:
        coefficients["coefficients_cubic"] = result["coefficients_cubic"]
        coefficients["coefficients_triplets"] = result["coefficients_triplets"]
        extra_arrays["triplets"] = (
            np.array(result["triplets"], dtype=int) if result["triplets"] else np.zeros((0, 3), dtype=int)
        )

    dataset = _package("polynomial", f"degree{degree}", X, Y, x_true, y_true, coefficients,
                        result["paras_to_vary"], [], structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_params, n_outputs,
                        extra_arrays=extra_arrays, extra_metadata=extra_metadata)
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def generate_exponential_dataset(n_samples, n_params, n_outputs, structure="full", structure_kwargs=None,
                                  noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                                  seed=None, nc_path=None):
    structure_kwargs = structure_kwargs or {}
    X_all = sample_X(n_samples + 1, n_params, seed=seed)

    if structure == "full":
        coefficients, paras_to_vary, outputs_to_vary = exponential.generate_matrix_full(
            n_params, n_outputs, seed=seed, **structure_kwargs)
    elif structure == "diag":
        coefficients, paras_to_vary = exponential.generate_matrix_diag(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        outputs_to_vary = []
    elif structure == "triu":
        coefficients = exponential.generate_matrix_triu(n_params, n_outputs, seed=seed, **structure_kwargs)
        paras_to_vary, outputs_to_vary = [], []
    else:
        raise ValueError(f"unknown structure: {structure!r}")

    Y_all = exponential.compute_Y(X_all, coefficients)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    dataset = _package("exponential", structure, X, Y, x_true, y_true, {"coefficients": coefficients},
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_params, n_outputs)
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def _ode_outputs_to_vary(paras_to_vary, n_states, n_times):
    """Sensitive states matter at every observed time -- expand each sensitive
    state index into the flattened (time-major) output columns it appears in."""
    return [t * n_states + s for t in range(n_times) for s in paras_to_vary]


def generate_linear_ode_dataset(n_samples, n_states, t_eval, structure="full", structure_kwargs=None,
                                 n_sensitive_para=0, sensitivity_multiplier=2.0, coefficient_scale=0.5,
                                 ic_scale=1.0, noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                                 seed=None, nc_path=None):
    """dx/dt = A x, integrated from sampled initial conditions and observed at
    t_eval. `structure` means the same thing as in linear.py (full/diag/triu
    coupling between states). n_outputs = n_states * len(t_eval) (one output
    per state per observed time) -- it's a consequence of t_eval, not chosen
    directly the way n_outputs is for the static generators.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    X_all = ode.sample_initial_conditions(n_samples + 1, n_states, scale=ic_scale, seed=seed)

    rhs, coefficients, _extra, paras_to_vary = ode.generate_rhs(
        n_states, degree=1, structure=structure, structure_kwargs=structure_kwargs,
        n_sensitive_para=n_sensitive_para, sensitivity_multiplier=sensitivity_multiplier,
        coefficient_scale=coefficient_scale, seed=seed)

    Y_all = ode.integrate_trajectories(rhs, X_all, t_eval)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    n_times = len(t_eval)
    outputs_to_vary = _ode_outputs_to_vary(paras_to_vary, n_states, n_times)

    dataset = _package("ode_linear", structure, X, Y, x_true, y_true, coefficients,
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_states,
                        n_times * n_states,
                        extra_metadata={"n_states": n_states, "t_eval": t_eval.tolist(),
                                         "coefficient_scale": coefficient_scale, "ic_scale": ic_scale})
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def generate_polynomial_ode_dataset(n_samples, n_states, t_eval, degree=2,
                                     n_sensitive_para=0, sensitivity_multiplier=2.0,
                                     n_pairs=0, n_sensitive_pairs=0,
                                     n_triplets=0, n_sensitive_triplets=0,
                                     coefficient_scale=0.3, ic_scale=1.0,
                                     noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                                     seed=None, nc_path=None):
    """dx/dt = quadratic/cubic polynomial in x (see ode.generate_rhs), integrated
    from sampled initial conditions and observed at t_eval. Same "which
    parameters interact" story as polynomial.py's static generator --
    n_pairs/n_sensitive_pairs choose which state pairs multiply together in
    the dynamics -- except now the interacting quantity feeds back into
    itself over time: with the right pair structure this reproduces
    Lotka-Volterra-style predator/prey dynamics (a state's growth rate
    depends on the product of itself and another state).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    X_all = ode.sample_initial_conditions(n_samples + 1, n_states, scale=ic_scale, seed=seed)

    rhs, coefficients, extra, paras_to_vary = ode.generate_rhs(
        n_states, degree=degree, n_sensitive_para=n_sensitive_para,
        sensitivity_multiplier=sensitivity_multiplier, coefficient_scale=coefficient_scale,
        n_pairs=n_pairs, n_sensitive_pairs=n_sensitive_pairs,
        n_triplets=n_triplets, n_sensitive_triplets=n_sensitive_triplets, seed=seed)

    Y_all = ode.integrate_trajectories(rhs, X_all, t_eval)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    n_times = len(t_eval)
    outputs_to_vary = _ode_outputs_to_vary(paras_to_vary, n_states, n_times)

    extra_arrays = {
        "pairs": np.array(extra["pairs"], dtype=int) if extra["pairs"] else np.zeros((0, 2), dtype=int),
        "triplets": np.array(extra["triplets"], dtype=int) if extra.get("triplets") else np.zeros((0, 3), dtype=int),
    }
    extra_metadata = {
        "n_states": n_states, "t_eval": t_eval.tolist(), "degree": degree,
        "coefficient_scale": coefficient_scale, "ic_scale": ic_scale,
        "n_pairs": n_pairs, "n_sensitive_pairs": n_sensitive_pairs,
        "n_triplets": n_triplets, "n_sensitive_triplets": n_sensitive_triplets,
    }

    dataset = _package(f"ode_polynomial_degree{degree}", "coupled", X, Y, x_true, y_true, coefficients,
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_states,
                        n_times * n_states, extra_arrays=extra_arrays, extra_metadata=extra_metadata)
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def generate_lorenz96_dataset(n_samples, n_states, t_eval, F=8.0,
                               n_sensitive_para=0, sensitivity_multiplier=1.5,
                               ic_perturbation_scale=1.0, noise_std=0.0, structural_idx=None,
                               structural_error_level=0.0, seed=None, nc_path=None):
    """Lorenz-96 chaotic system (see lorenz96.py): dx_i/dt = (x_{i+1} -
    x_{i-2}) x_{i-1} - x_i + F_i, cyclic. The coupling pattern is fixed by
    the model; the "coefficients" here are the per-state forcing F, with
    n_sensitive_para states carrying an amplified forcing.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    forcing, paras_to_vary = lorenz96.generate_forcing(
        n_states, F=F, n_sensitive_para=n_sensitive_para,
        sensitivity_multiplier=sensitivity_multiplier, seed=seed)
    X_all = lorenz96.sample_initial_conditions(
        n_samples + 1, n_states, F=F, perturbation_scale=ic_perturbation_scale, seed=seed)

    Y_all = lorenz96.integrate_trajectories(forcing, X_all, t_eval)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    n_times = len(t_eval)
    outputs_to_vary = _ode_outputs_to_vary(paras_to_vary, n_states, n_times)
    coefficients = {"forcing": forcing}

    dataset = _package("lorenz96", "cyclic", X, Y, x_true, y_true, coefficients,
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_states,
                        n_times * n_states,
                        extra_metadata={"n_states": n_states, "t_eval": t_eval.tolist(), "F": F})
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def generate_structural_regime_dataset(generate_fn, n_samples, regime_fraction=0.5,
                                        kwargs_1=None, kwargs_2=None, seed=None, nc_path=None):
    """Structural/model-form error, the way it actually shows up in practice:
    not a random per-column bias, but data secretly generated by *two
    different* parameter sets (regime 1 and regime 2) and pooled into one
    ensemble. `generate_fn` is any generate_*_dataset function above;
    kwargs_1/kwargs_2 are its keyword arguments for each regime (they must
    produce matching X/Y dimensions, since rows get stacked). `regime_fraction`
    of the rows come from regime 2, the rest from regime 1, and rows are
    shuffled together so a regime isn't just "the second half".

    `x_true`/`y_true` are kept per regime (there's no single held-out point
    that represents "the" truth here -- that's the point). `regime_label`
    (0/1 per row) is included for diagnostic use, but a real calibration
    exercise would not have access to it.
    """
    kwargs_1 = dict(kwargs_1 or {})
    kwargs_2 = dict(kwargs_2 or {})
    rng = np.random.default_rng(seed)

    n_2 = int(round(n_samples * regime_fraction))
    n_1 = n_samples - n_2
    seed_1 = int(rng.integers(2**32))
    seed_2 = int(rng.integers(2**32))

    ds1 = generate_fn(n_samples=n_1, seed=seed_1, **kwargs_1)
    ds2 = generate_fn(n_samples=n_2, seed=seed_2, **kwargs_2)

    if ds1["X"].shape[1] != ds2["X"].shape[1] or ds1["Y"].shape[1] != ds2["Y"].shape[1]:
        raise ValueError("regime_1 and regime_2 datasets must have matching X/Y dimensions to be pooled")

    X = np.concatenate([ds1["X"], ds2["X"]], axis=0)
    Y = np.concatenate([ds1["Y"], ds2["Y"]], axis=0)
    regime_label = np.concatenate([np.zeros(n_1, dtype=int), np.ones(n_2, dtype=int)])

    perm = rng.permutation(n_samples)
    X, Y, regime_label = X[perm], Y[perm], regime_label[perm]

    paras_to_vary = sorted(set(ds1["paras_to_vary"].tolist()) | set(ds2["paras_to_vary"].tolist()))
    outputs_to_vary = sorted(set(ds1["outputs_to_vary"].tolist()) | set(ds2["outputs_to_vary"].tolist()))

    dataset = {
        "X": X,
        "Y": Y,
        "x_true": {"regime_1": ds1["x_true"], "regime_2": ds2["x_true"]},
        "y_true": {"regime_1": ds1["y_true"], "regime_2": ds2["y_true"]},
        "regime_label": regime_label,
        "coefficients": {"regime_1": ds1["coefficients"], "regime_2": ds2["coefficients"]},
        "paras_to_vary": np.array(paras_to_vary, dtype=int),
        "outputs_to_vary": np.array(outputs_to_vary, dtype=int),
        "metadata": {
            "generator": generate_fn.__name__,
            "structural_regime_error": True,
            "regime_fraction": regime_fraction,
            "n_samples": n_samples,
            "n_regime_1": n_1,
            "n_regime_2": n_2,
            "seed": seed,
            "regime_1_metadata": ds1["metadata"],
            "regime_2_metadata": ds2["metadata"],
        },
    }
    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset
