"""Fit simple empirical model/data scaling curves from AethyxLM runs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalingFit:
    irreducible_loss: float
    parameter_exponent: float
    token_exponent: float
    coefficient: float
    log_mse: float


def fit_scaling_law(parameters, tokens, losses, floor_candidates=200):
    """Fit L = E + A * N^-alpha * D^-beta by a floor grid search."""
    parameters = np.asarray(parameters, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if not (parameters.shape == tokens.shape == losses.shape) or parameters.size < 4:
        raise ValueError("at least four aligned scaling observations are required")
    if np.any(parameters <= 0) or np.any(tokens <= 0) or np.any(losses <= 0):
        raise ValueError("scaling observations must be positive")
    design = np.column_stack(
        (np.ones(parameters.size), -np.log(parameters), -np.log(tokens))
    )
    best = None
    for floor in np.linspace(0.0, float(losses.min()) * 0.99, floor_candidates):
        target = np.log(losses - floor)
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        mse = float(np.mean(residual**2))
        if best is None or mse < best.log_mse:
            best = ScalingFit(
                irreducible_loss=float(floor),
                coefficient=float(np.exp(coefficients[0])),
                parameter_exponent=float(coefficients[1]),
                token_exponent=float(coefficients[2]),
                log_mse=mse,
            )
    return best
