import numpy as np
import pytest

from evaluation.scaling import fit_scaling_law


def test_scaling_fit_recovers_synthetic_exponents():
    parameters = np.array([10, 20, 40, 80, 20, 40, 80, 160], dtype=float) * 1e6
    tokens = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=float) * 1e9
    losses = 1.2 + 80 * parameters ** -0.2 * tokens ** -0.1
    fit = fit_scaling_law(parameters, tokens, losses, floor_candidates=1000)
    assert fit.irreducible_loss == pytest.approx(1.2, abs=0.02)
    assert fit.parameter_exponent == pytest.approx(0.2, abs=0.02)
    assert fit.token_exponent == pytest.approx(0.1, abs=0.02)
