"""Tests for asymmetric Shapley values computed under a partial feature ordering."""

import numpy as np

from pddshap.signature import FeatureSubset

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
    components = fitted_decomposition.components
    c0 = components[FeatureSubset(0)](X_test_np)
    c1 = components[FeatureSubset(1)](X_test_np)
    c2 = components[FeatureSubset(2)](X_test_np)
    c01 = components[FeatureSubset(0, 1)](X_test_np)
    c02 = components[FeatureSubset(0, 2)](X_test_np)
    c12 = components[FeatureSubset(1, 2)](X_test_np)

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
