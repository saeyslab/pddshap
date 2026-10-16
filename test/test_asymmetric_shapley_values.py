"""Tests for asymmetric Shapley values computed under a partial feature ordering."""

import numpy as np

from .conftest import NUM_FEATURES


def test_asymmetric_shapley_values_with_partial_ordering(
    quadratic_model_and_data, fitted_decomposition
):
    _, _, X_test = quadratic_model_and_data
    X_test_np = X_test.to_numpy()
    partial_ordering = [[0], [1, 2]]

    pdd_values = fitted_decomposition.shapley_values(
        X_test, partial_ordering=partial_ordering
    )

    assert pdd_values.shape == (len(X_test), NUM_FEATURES, 1)
    assert np.all(np.isfinite(pdd_values))

    # Feature 0 strictly precedes 1 and 2. Per `_asv_coefficients`, a feature
    # gets zero credit from any subset that also contains one of its
    # successors - the successor absorbs the interaction instead. So feature
    # 0's value is just its own main effect, while features 1 and 2 (mutually
    # incomparable) each pick up half of their shared pairwise component on
    # top of their own main effect and interaction with feature 0.
    dataset = fitted_decomposition.dataset
    pdp_values = fitted_decomposition.evaluate(X_test_np)

    def evaluate(*features):
        return pdp_values[dataset.get_subset(*features)]

    c0 = evaluate(0)
    c1 = evaluate(1)
    c2 = evaluate(2)
    c01 = evaluate(0, 1)
    c02 = evaluate(0, 2)
    c12 = evaluate(1, 2)

    expected_feature_0 = c0
    expected_feature_1 = c1 + c01 + c12 / 2
    expected_feature_2 = c2 + c02 + c12 / 2

    np.testing.assert_allclose(
        pdd_values[:, 0, 0], expected_feature_0.ravel(), atol=1e-6
    )
    np.testing.assert_allclose(
        pdd_values[:, 1, 0], expected_feature_1.ravel(), atol=1e-6
    )
    np.testing.assert_allclose(
        pdd_values[:, 2, 0], expected_feature_2.ravel(), atol=1e-6
    )
