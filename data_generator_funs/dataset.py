import numpy as np

from . import exponential, linear, lorenz96, ode, polynomial
from .linear import sample_X
from .netcdf_io import (  # noqa: F401 (re-exported)
    default_nc_filename, load_linear_dataset, save_dataset_folder, save_dataset_netcdf,
)

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
    if isinstance(value, (list, tuple)):
        return "-".join(_format_kwarg_value(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p").replace("-", "m")


def _structure_kwargs_suffix(structure_kwargs):
    if not structure_kwargs:
        return ""
    parts = [f"{_abbreviate_kwarg_name(k)}{_format_kwarg_value(v)}" for k, v in structure_kwargs.items()]
    return "_" + "_".join(parts)


def _generate_structured_coefficients(module, n_params, n_outputs, structure, structure_kwargs, seed):
    """Shared structure= dispatch for linear.py/exponential.py, which expose the
    same four coefficient generators (full/diag/triu/blocks) under the same
    names. Returns (coefficients, paras_to_vary, outputs_to_vary, extra_arrays)
    -- extra_arrays is {} except for structure="blocks", where it carries
    param_group_id/output_group_id (see linear.generate_matrix_blocks)."""
    structure_kwargs = structure_kwargs or {}
    if structure == "full":
        coefficients, paras_to_vary, outputs_to_vary = module.generate_matrix_full(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        return coefficients, paras_to_vary, outputs_to_vary, {}
    if structure == "diag":
        coefficients, paras_to_vary = module.generate_matrix_diag(n_params, n_outputs, seed=seed, **structure_kwargs)
        return coefficients, paras_to_vary, [], {}
    if structure == "triu":
        coefficients = module.generate_matrix_triu(n_params, n_outputs, seed=seed, **structure_kwargs)
        return coefficients, [], [], {}
    if structure == "blocks":
        coefficients, param_group_id, output_group_id = module.generate_matrix_blocks(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        return coefficients, [], [], {"param_group_id": param_group_id, "output_group_id": output_group_id}
    if structure == "overlapping_blocks":
        coefficients, param_group_membership, output_group_id = module.generate_matrix_overlapping_blocks(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        return coefficients, [], [], {
            "param_group_membership": param_group_membership.astype(int), "output_group_id": output_group_id,
        }
    raise ValueError(f"unknown structure: {structure!r}")


def generate_linear_dataset(n_samples, n_params, n_outputs, structure="full", structure_kwargs=None,
                             noise_std=0.0, structural_idx=None, structural_error_level=0.0,
                             seed=None, nc_path=None):
    """nc_path, if given, is the *directory* to write into -- the filename is
    derived from the run's shape, e.g. "linear_full_n100_p40_o40.nc"."""
    structure_kwargs = structure_kwargs or {}
    X_all = sample_X(n_samples + 1, n_params, seed=seed)

    coefficients, paras_to_vary, outputs_to_vary, extra_arrays = _generate_structured_coefficients(
        linear, n_params, n_outputs, structure, structure_kwargs, seed=seed)

    Y_all = linear.compute_Y(X_all, coefficients)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    dataset = _package("linear", structure, X, Y, x_true, y_true, {"coefficients": coefficients},
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_params, n_outputs,
                        extra_arrays=extra_arrays or None)
    if nc_path:
        filename_base = (f"linear_{structure}_n{n_samples}_p{n_params}_o{n_outputs}"
                          f"{_structure_kwargs_suffix(structure_kwargs)}")
        save_dataset_folder(dataset, nc_path, filename_base)
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

    coefficients, paras_to_vary, outputs_to_vary, extra_arrays = _generate_structured_coefficients(
        exponential, n_params, n_outputs, structure, structure_kwargs, seed=seed)

    Y_all = exponential.compute_Y(X_all, coefficients)
    X, Y, x_true, y_true_clean = X_all[:-1], Y_all[:-1], X_all[-1], Y_all[-1]
    y_true, structural_bias = add_true_observation_error(
        y_true_clean, Y, noise_std=noise_std, structural_idx=structural_idx,
        structural_error_level=structural_error_level, seed=seed)

    dataset = _package("exponential", structure, X, Y, x_true, y_true, {"coefficients": coefficients},
                        paras_to_vary, outputs_to_vary, structural_bias, noise_std,
                        structural_idx, structural_error_level, seed, n_samples, n_params, n_outputs,
                        extra_arrays=extra_arrays or None)
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


_CONFLICTING_TRUTH_FAMILIES = ("linear", "exponential", "polynomial")


def _family_generate_coefficients(family, n_params, n_outputs, structure, structure_kwargs, seed):
    """Coefficient generation for one model block of generate_conflicting_truth_dataset,
    dispatched by family name. linear/exponential go through the same structure=
    full/diag/triu/blocks dispatch as their generate_*_dataset functions
    (_generate_structured_coefficients); polynomial always uses its own
    generate_matrix_full (degree/pairs/triplets are its own "structure").
    Returns (coefficients, paras_to_vary) -- coefficients is a plain array for
    linear/exponential, a dict of arrays for polynomial (same shape compute_Y expects).

    Adding a new family here is what makes this generator model-agnostic: any
    module exposing a compatible coefficient-generator + compute_Y can be
    wired in the same way. ODE/Lorenz96 aren't included since their n_outputs
    is a consequence of t_eval rather than chosen directly -- combining those
    would need the shared ensemble to carry a time axis too.
    """
    if family in ("linear", "exponential"):
        module = linear if family == "linear" else exponential
        coefficients, paras_to_vary, _outputs_to_vary, _extra = _generate_structured_coefficients(
            module, n_params, n_outputs, structure, structure_kwargs, seed=seed)
        return coefficients, paras_to_vary
    if family == "polynomial":
        result = polynomial.generate_matrix_full(n_params, n_outputs, seed=seed, **(structure_kwargs or {}))
        coefficients = {k: v for k, v in result.items() if k != "paras_to_vary"}
        return coefficients, result["paras_to_vary"]
    raise ValueError(f"unknown family: {family!r} (expected one of {_CONFLICTING_TRUTH_FAMILIES})")


def _family_compute_Y(family, coefficients, X):
    if family == "linear":
        return linear.compute_Y(X, coefficients)
    if family == "exponential":
        return exponential.compute_Y(X, coefficients)
    if family == "polynomial":
        return polynomial.compute_Y(
            X, coefficients["coefficients_linear"], coefficients["coefficients_quad"],
            coefficients["pairs"], coefficients["coefficients_pairs"],
            coefficients_cubic=coefficients.get("coefficients_cubic"),
            triplets=coefficients.get("triplets"), coefficients_triplets=coefficients.get("coefficients_triplets"))
    raise ValueError(f"unknown family: {family!r} (expected one of {_CONFLICTING_TRUTH_FAMILIES})")


def generate_conflicting_truth_dataset(n_samples, n_params, model_specs, noise_std=0.0, seed=None, nc_path=None):
    """The multi-model structural-error scenario: a set of x values maps to y_1
    given model m_1, and the *same* x values also map to y_2 given a
    different model m_2 -- m_1 and m_2 share the same parameter space (same
    n_params), so the training ensemble (X, Y) is a single valid table with
    Y = [Y_1 | Y_2 | ...] as column-blocks, one per model. The conflict is in
    the truth: each model's y_true is generated from its *own*, independently
    sampled x_true_i rather than one shared x_true -- so a calibration
    exercise handed the joint y_true has no single x that can reproduce every
    block, which is the point (this is what "no parameter values satisfy y_1
    and y_2 given m_1 and m_2" looks like as data).

    model_specs: list of dicts, one per model/output-block:
        {"name": "m1" (optional, defaults to "m1", "m2", ...),
         "family": "linear" | "exponential" | "polynomial",
         "n_outputs": int,
         "structure": structure kwarg for that family's generator (default "full";
             ignored -- always "full" -- for polynomial, whose own knobs are
             degree/n_pairs/etc.),
         "structure_kwargs": dict of kwargs for that family's coefficient generator}
    See _family_generate_coefficients for which families are supported and how
    to add more.

    Returns a dataset dict shaped like generate_structural_regime_dataset's output
    for the parts that don't have a single answer -- x_true/y_true/coefficients
    are dicts keyed by model name -- plus:
        output_model_label: (n_outputs,) int, which model block each output column
            belongs to.
        cross_model_mismatch: (n_models, n_models) array; entry [i, j] is the
            normalized RMS gap between model i's actual y_true and what model i
            would predict at model j's x_true (each output normalized by its
            ensemble std, same convention as structural_error_level elsewhere).
            The diagonal is ~0 (noise_std only) by construction; large
            off-diagonal entries are the visible signature of the conflict.
    """
    rng = np.random.default_rng(seed)
    X = sample_X(n_samples, n_params, seed=seed)

    model_names, coefficients_by_model, compute_fns = [], {}, {}
    Y_blocks, x_true_by_model, y_true_clean_by_model, y_true_by_model = [], {}, {}, {}
    output_model_label_parts, per_model_metadata = [], {}

    for model_idx, spec in enumerate(model_specs):
        name = spec.get("name", f"m{model_idx + 1}")
        family = spec["family"]
        n_outputs_i = spec["n_outputs"]
        structure = spec.get("structure", "full")
        structure_kwargs = spec.get("structure_kwargs", {})
        model_seed = int(rng.integers(2**32))
        true_seed = int(rng.integers(2**32))

        coefficients_i, paras_to_vary_i = _family_generate_coefficients(
            family, n_params, n_outputs_i, structure, structure_kwargs, seed=model_seed)
        Y_i = _family_compute_Y(family, coefficients_i, X)

        x_true_i = sample_X(1, n_params, seed=true_seed)[0]
        y_true_i_clean = _family_compute_Y(family, coefficients_i, x_true_i[None, :])[0]
        y_true_i, _structural_bias_i = add_true_observation_error(
            y_true_i_clean, Y_i, noise_std=noise_std, seed=true_seed)

        model_names.append(name)
        coefficients_by_model[name] = coefficients_i
        compute_fns[name] = lambda Xin, c=coefficients_i, f=family: _family_compute_Y(f, c, Xin)
        Y_blocks.append(Y_i)
        x_true_by_model[name] = x_true_i
        y_true_clean_by_model[name] = y_true_i_clean
        y_true_by_model[name] = y_true_i
        output_model_label_parts.append(np.full(n_outputs_i, model_idx))
        per_model_metadata[name] = {
            "family": family, "structure": structure, "n_outputs": n_outputs_i,
            "paras_to_vary": list(paras_to_vary_i), "seed": model_seed,
        }

    Y = np.hstack(Y_blocks)
    output_model_label = np.concatenate(output_model_label_parts)

    n_models = len(model_names)
    cross_model_mismatch = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        ref_std = Y_blocks[i].std(axis=0)
        ref_std = np.where(ref_std == 0, 1.0, ref_std)
        for j, name_j in enumerate(model_names):
            y_pred = compute_fns[name_i](x_true_by_model[name_j][None, :])[0]
            gap = (y_true_clean_by_model[name_i] - y_pred) / ref_std
            cross_model_mismatch[i, j] = np.sqrt(np.mean(gap ** 2))

    dataset = {
        "X": X,
        "Y": Y,
        "x_true": x_true_by_model,
        "y_true": y_true_by_model,
        "output_model_label": output_model_label,
        "cross_model_mismatch": cross_model_mismatch,
        "coefficients": coefficients_by_model,
        "metadata": {
            "model": "conflicting_truth",
            "structural_conflicting_models": True,
            "model_names": model_names,
            "n_samples": n_samples,
            "n_params": n_params,
            "n_outputs": int(Y.shape[1]),
            "noise_std": noise_std,
            "seed": seed,
            "per_model": per_model_metadata,
        },
    }
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
