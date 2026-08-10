import numpy as np
from scipy.integrate import solve_ivp

from . import linear, polynomial

"""ODE data generators: a state trajectory x(t) obtained by integrating
dx/dt = f(x) from a sampled initial condition, observed at a fixed set of
times. degree=1/2/3 means f is linear/quadratic/cubic in x, built by
reusing linear.py's and polynomial.py's own coefficient generators and
compute_Y with n_params = n_outputs = n_states -- "state k's derivative
depends on state i" is the same (row, col) relationship as "output k
depends on param i" in the static generators, so f(x) is literally
linear.compute_Y / polynomial.compute_Y evaluated at a single state vector
instead of a batch of inputs.

structure= (full/diag/triu, degree=1 only) means exactly what it does in
linear.py: diag = each state evolves independently (no interaction
between states), full = every state's derivative depends on every state,
triu = one-directional coupling. For degree 2/3 the interaction knobs are
n_pairs/n_sensitive_pairs (and triplets), exactly as in polynomial.py --
e.g. with the right pair structure this reproduces Lotka-Volterra-style
predator/prey dynamics (a state's growth rate depends on the product of
itself and another state).
"""


def sample_initial_conditions(n_samples, n_states, scale=1.0, seed=None):
    """Draws initial conditions x0 ~ N(0, scale^2), shape (n_samples, n_states)
    -- the ODE analogue of linear.sample_X."""
    rng = np.random.default_rng(seed)
    return scale * rng.normal(size=(n_samples, n_states))


def generate_rhs(n_states, degree=1, structure="full", structure_kwargs=None,
                  n_sensitive_para=0, sensitivity_multiplier=2.0, coefficient_scale=1.0,
                  n_pairs=0, n_sensitive_pairs=0, n_triplets=0, n_sensitive_triplets=0,
                  seed=None):
    """Builds dx/dt = f(x). Returns (rhs, coefficients, extra, paras_to_vary):

    rhs(t, x): evaluates f for a single state vector x, shape (n_states,).
    coefficients: dict of raw coefficient arrays (same keys/shapes as the
        matching static generator), already multiplied by coefficient_scale.
    extra: {} for degree=1; {"pairs": [...], "triplets": [...]} for degree 2/3.
    paras_to_vary: indices of states whose coupling was amplified.

    coefficient_scale shrinks the raw N(0,1) coefficients before use --
    quadratic/cubic feedback terms blow up in finite time if left at scale
    1, so a smaller scale (e.g. 0.2-0.5) keeps trajectories well-behaved
    over a reasonable integration horizon.
    """
    structure_kwargs = structure_kwargs or {}

    if degree == 1:
        if structure == "full":
            A, paras_to_vary, _ = linear.generate_matrix_full(
                n_states, n_states, n_sensitive_para=n_sensitive_para,
                n_sensitive_output=n_sensitive_para, sensitivity_multiplier=sensitivity_multiplier,
                seed=seed, **structure_kwargs)
        elif structure == "diag":
            A, paras_to_vary = linear.generate_matrix_diag(
                n_states, n_states, n_sensitive_para=n_sensitive_para,
                sensitivity_multiplier=sensitivity_multiplier, seed=seed, **structure_kwargs)
        elif structure == "triu":
            A = linear.generate_matrix_triu(
                n_states, n_states, n_sensitive_para, n_sensitive_para, seed=seed, **structure_kwargs)
            paras_to_vary = []
        else:
            raise ValueError(f"unknown structure: {structure!r}")

        A = A * coefficient_scale
        coefficients = {"A": A}

        def rhs(t, x):
            return linear.compute_Y(x[None, :], A)[0]

        return rhs, coefficients, {}, paras_to_vary

    if degree in (2, 3):
        result = polynomial.generate_matrix_full(
            n_states, n_states, degree=degree,
            n_sensitive_para=n_sensitive_para, sensitivity_multiplier=sensitivity_multiplier,
            n_pairs=n_pairs, n_sensitive_pairs=n_sensitive_pairs,
            n_triplets=n_triplets, n_sensitive_triplets=n_sensitive_triplets,
            seed=seed)

        coefficients = {}
        for key in ("coefficients_linear", "coefficients_quad", "coefficients_pairs",
                    "coefficients_cubic", "coefficients_triplets"):
            if key in result:
                coefficients[key] = result[key] * coefficient_scale

        def rhs(t, x):
            return polynomial.compute_Y(
                x[None, :], coefficients["coefficients_linear"], coefficients["coefficients_quad"],
                result["pairs"], coefficients["coefficients_pairs"],
                coefficients_cubic=coefficients.get("coefficients_cubic"),
                triplets=result.get("triplets"),
                coefficients_triplets=coefficients.get("coefficients_triplets"),
            )[0]

        extra = {"pairs": result["pairs"], "triplets": result.get("triplets", [])}
        return rhs, coefficients, extra, result["paras_to_vary"]

    raise ValueError(f"degree must be 1, 2, or 3, got {degree}")


def integrate_trajectories(rhs, X0, t_eval, method="RK45", rtol=1e-6, atol=1e-9):
    """Integrates dx/dt = rhs(t, x) from each row of X0 (n_samples, n_states),
    treating t_eval[0] as the initial time and evaluating the solution at
    every point in t_eval. Returns Y of shape (n_samples, len(t_eval) *
    n_states), each row's trajectory flattened time-major (all states at
    t_eval[0], then all states at t_eval[1], ...).

    Raises RuntimeError if any trajectory's integration fails (e.g. blew up
    in finite time) -- typically fixed by lowering coefficient_scale,
    shortening t_eval, or shrinking the initial-condition scale.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    n_samples, n_states = X0.shape
    n_times = len(t_eval)
    t_span = (t_eval[0], t_eval[-1])

    Y = np.empty((n_samples, n_times, n_states))
    for s in range(n_samples):
        sol = solve_ivp(rhs, t_span, X0[s], t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(
                f"ODE integration failed for sample {s}: {sol.message}. "
                "Try a smaller coefficient_scale, a shorter t_eval, or smaller initial conditions."
            )
        Y[s] = sol.y.T

    return Y.reshape(n_samples, n_times * n_states)
