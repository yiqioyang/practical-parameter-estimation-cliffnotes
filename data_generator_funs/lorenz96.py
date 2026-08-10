import numpy as np
from scipy.integrate import solve_ivp

"""Lorenz-96 model -- a classic chaotic toy-climate system:

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F_i,   i = 0..n_states-1 (cyclic)

Unlike the linear/polynomial ODE families, there's no structure= choice
here: every state interacts with its two neighbors and its "predecessor's
predecessor" in a fixed cyclic pattern -- the coupling pattern *is* the
model, not something we generate. The free parameters are the forcing F
(scalar, or a per-node vector so n_sensitive_para nodes can carry an
amplified forcing) and the initial conditions. F=8 with n_states>=4 sits in
the standard chaotic regime.
"""


def generate_forcing(n_states, F=8.0, n_sensitive_para=0, sensitivity_multiplier=1.5, seed=None):
    """Per-node forcing vector, shape (n_states,): constant at F everywhere
    except n_sensitive_para randomly chosen nodes, which get F * sensitivity_multiplier.
    Returns (forcing, paras_to_vary)."""
    rng = np.random.default_rng(seed)
    forcing = np.full(n_states, float(F))
    paras_to_vary = []
    if n_sensitive_para > 0:
        paras_to_vary = rng.choice(n_states, size=n_sensitive_para, replace=False).tolist()
        forcing[paras_to_vary] *= sensitivity_multiplier
    return forcing, paras_to_vary


def rhs(t, x, forcing):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def sample_initial_conditions(n_samples, n_states, F=8.0, perturbation_scale=1.0, seed=None):
    """Initial conditions near the forcing's fixed point (x_i = F) plus a
    small perturbation -- starting exactly at x_i=0 with F=8 sits in an
    unusually low-energy transient; nudging around the F fixed point
    reaches the chaotic attractor faster."""
    rng = np.random.default_rng(seed)
    return F + perturbation_scale * rng.normal(size=(n_samples, n_states))


def integrate_trajectories(forcing, X0, t_eval, method="RK45", rtol=1e-6, atol=1e-9):
    """Same contract as ode.integrate_trajectories, with the fixed Lorenz-96 rhs."""
    t_eval = np.asarray(t_eval, dtype=float)
    n_samples, n_states = X0.shape
    n_times = len(t_eval)
    t_span = (t_eval[0], t_eval[-1])

    Y = np.empty((n_samples, n_times, n_states))
    for s in range(n_samples):
        sol = solve_ivp(rhs, t_span, X0[s], t_eval=t_eval, args=(forcing,), method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Lorenz-96 integration failed for sample {s}: {sol.message}")
        Y[s] = sol.y.T

    return Y.reshape(n_samples, n_times * n_states)
