"""
End-to-end regression tests for PartialDependenceDecomposition.

The model under test is a multilinear polynomial (test/multilinear_polynomial.py),
which has closed-form Shapley values and a known variance decomposition. This lets
us check PDD-SHAP's output against ground truth instead of just checking it runs.
"""

import numpy as np
import pandas as pd
import pytest
from .multilinear_polynomial import RandomMultilinearPolynomial
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.model_selection import train_test_split

from pddshap import PartialDependenceDecomposition
from pddshap.sampling import IndependentConditioningMethod, RandomSubsampleCollocation
from pddshap.signature import FeatureSubset

NUM_FEATURES = 3
SEED = 0

# Background/collocation size (X_train below) and the held-out test size.
# A larger background reduces Monte-Carlo noise in the conditional-expectation
# estimates that each PDDComponent is fitted on, which matters most for the
# less smooth estimators (tree, forest).
TOTAL_SIZE = 3000
TEST_SIZE = 0.85

# Per-estimator quality thresholds. These differ because the estimators have
# different inductive biases (e.g. a single unconstrained DecisionTreeRegressor
# fits a step function to a smooth surface, so it recovers raw output less
# accurately than knn/forest even though its Shapley value ranking is fine).
ESTIMATOR_CONFIGS = [
    pytest.param(
        {
            "estimator_type": "knn",
            "est_kwargs": {"k": 3},
            "r2_threshold": 0.9,
            "pearson_threshold": 0.9,
        },
        id="knn",
    ),
    pytest.param(
        {
            "estimator_type": "forest",
            "est_kwargs": {},
            "r2_threshold": 0.85,
            "pearson_threshold": 0.9,
        },
        id="forest",
    ),
    pytest.param(
        {
            "estimator_type": "tree",
            "est_kwargs": {},
            "r2_threshold": 0.75,
            "pearson_threshold": 0.9,
        },
        id="tree",
    ),
]


@pytest.fixture(scope="module")
def quadratic_model_and_data():
    # Bias + linear + pairwise terms only (no 3rd-order interaction), so a
    # decomposition fitted with max_size=2 has no structural approximation error.
    np.random.seed(SEED)
    model = RandomMultilinearPolynomial(NUM_FEATURES, [-1, -1, -1])

    mean = np.zeros(NUM_FEATURES)
    cov = np.eye(NUM_FEATURES)
    X = np.random.multivariate_normal(mean, cov, size=TOTAL_SIZE).astype(np.float32)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])

    X_train, X_test = train_test_split(X_df, test_size=TEST_SIZE, random_state=SEED)
    return model, X_train, X_test


@pytest.fixture(scope="module", params=ESTIMATOR_CONFIGS)
def estimator_config(request):
    return request.param


@pytest.fixture(scope="module")
def fitted_decomposition(quadratic_model_and_data, estimator_config):
    model, X_train, _ = quadratic_model_and_data
    decomposition = PartialDependenceDecomposition(
        model,
        collocation_method=RandomSubsampleCollocation(),
        conditioning_method=IndependentConditioningMethod(X_train.to_numpy()),
        estimator_type=estimator_config["estimator_type"],
        est_kwargs=estimator_config["est_kwargs"],
    )
    decomposition.fit(X_train, X_train, max_size=2)
    return decomposition


def test_fit_creates_expected_feature_subset_components(fitted_decomposition):
    subset_sizes = sorted(len(fs) for fs in fitted_decomposition.components)

    assert subset_sizes.count(0) == 1
    assert subset_sizes.count(1) == NUM_FEATURES
    assert subset_sizes.count(2) == NUM_FEATURES * (NUM_FEATURES - 1) // 2
    assert max(subset_sizes) == 2, "fit(max_size=2) must not fit larger subsets"
    assert FeatureSubset() in fitted_decomposition.components


def test_decomposition_reconstructs_model_output(
    quadratic_model_and_data, fitted_decomposition, estimator_config
):
    model, _, X_test = quadratic_model_and_data

    predicted = fitted_decomposition(X_test).ravel()
    true = model(X_test.to_numpy())

    r2 = metrics.r2_score(true, predicted)
    threshold = estimator_config["r2_threshold"]
    assert r2 > threshold, (
        f"Decomposition should recover the full model output, got R2={r2}"
    )


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
