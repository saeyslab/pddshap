"""Tests for symmetric Shapley values: ground-truth correlation and the efficiency axiom."""

import numpy as np
from scipy.stats import pearsonr


def test_shapley_values_correlate_with_ground_truth(
    quadratic_model_and_data, fitted_decomposition, estimator_config
):
    model, _, X_test = quadratic_model_and_data

    true_values = model.shapley_values(X_test.to_numpy())
    pdd_values = fitted_decomposition.shapley_values(X_test, project=True)[:, :, 0]

    pearsons = [
        pearsonr(pdd_values[i], true_values[i])[0]
        for i in range(len(X_test))
        if not np.allclose(true_values[i], true_values[i][0])
    ]
    assert np.mean(pearsons) > estimator_config["pearson_threshold"]


def test_shapley_values_satisfy_efficiency_axiom(
    quadratic_model_and_data, fitted_decomposition
):
    model, _, X_test = quadratic_model_and_data

    pdd_values = fitted_decomposition.shapley_values(X_test, project=True)
    pred_diff = model(X_test.to_numpy()) - fitted_decomposition.bg_avg

    np.testing.assert_allclose(
        pdd_values.sum(axis=1).ravel(), pred_diff.ravel(), atol=1e-5
    )
