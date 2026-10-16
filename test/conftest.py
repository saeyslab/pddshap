"""
Shared fixtures for the PartialDependenceDecomposition test suite.

The model under test is a multilinear polynomial (pddshap.synthetic.multilinear),
which has closed-form Shapley values and no 3rd-order interaction. This lets
tests check PDD-SHAP's output against ground truth (with max_size=2 having no
structural approximation error) instead of just checking it runs.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from pddshap import PartialDependenceDecomposition
from pddshap.sampling import IndependentConditioningMethod, RandomSubsampleCollocation
from pddshap.experiments import RandomMultilinearPolynomial

NUM_FEATURES = 3
SEED = 0

# Background/collocation size (X_train below) and the held-out test size.
# A larger background reduces Monte-Carlo noise in the conditional-expectation
# estimates that each feature subset's estimator is fitted on, which matters
# most for the less smooth estimators (tree, forest).
TOTAL_SIZE = 1000
TEST_SIZE = 0.1

# Per-estimator quality thresholds. These differ because the estimators have
# different inductive biases (e.g. a single unconstrained DecisionTreeRegressor
# fits a step function to a smooth surface, so it recovers raw output less
# accurately than knn/forest even though its Shapley value ranking is fine).
ESTIMATOR_CONFIGS = [
    pytest.param(
        {
            "estimator_type": KNeighborsRegressor,
            "est_kwargs": {"n_neighbors": 3},
            "r2_threshold": 0.9,
            "pearson_threshold": 0.9,
        },
        id="knn",
    ),
    pytest.param(
        {
            "estimator_type": GradientBoostingRegressor,
            "est_kwargs": {},
            "r2_threshold": 0.9,
            "pearson_threshold": 0.9,
        },
        id="forest",
    ),
    pytest.param(
        {
            "estimator_type": DecisionTreeRegressor,
            "est_kwargs": {},
            "r2_threshold": 0.9,
            "pearson_threshold": 0.9,
        },
        id="tree",
    ),
]


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session", params=ESTIMATOR_CONFIGS)
def estimator_config(request):
    return request.param


@pytest.fixture(scope="session")
def fitted_decomposition(quadratic_model_and_data, estimator_config):
    model, X_train, _ = quadratic_model_and_data
    decomposition = PartialDependenceDecomposition(
        model,
        collocation_method=RandomSubsampleCollocation(),
        conditioning_method=IndependentConditioningMethod(X_train.to_numpy()),
        estimator_type=estimator_config["estimator_type"],
        estimator_kwargs=estimator_config["est_kwargs"],
    )
    decomposition.fit(X_train, max_size=2)
    return decomposition
