import numpy as np
import random

def sample_X(n_samples, n_params, seed=None):
    """Draws the input matrix X ~ N(0, 1), shape (n_samples, n_params)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_params))


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


def compute_Y(X, coefficients):
    """y = X @ coefficients, shape (n_samples, n_outputs)."""
    return X @ coefficients
