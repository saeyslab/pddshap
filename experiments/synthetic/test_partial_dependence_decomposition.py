from typing import cast

import numpy as np
import pandas as pd
from multilinear_polynomial import RandomMultilinearPolynomial
from numpy import typing as npt
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.model_selection import train_test_split

from pddshap import PartialDependenceDecomposition
from pddshap.sampling import IndependentConditioningMethod, RandomSubsampleCollocation


def r2_score(values: npt.NDArray, true_values: npt.NDArray):
    # Shape: [num_samples, num_features, num_outputs]
    if values.shape != true_values.shape:
        raise ValueError(f"Shapes don't match: {values.shape}, {true_values.shape}")

    r2_values = []
    for i in range(true_values.shape[2]):
        r2_values.append(metrics.r2_score(true_values[:, :, i], values[:, :, i]))
    return np.array(r2_values)


def correlations(values1: npt.NDArray, values2: npt.NDArray):
    # Shape: [num_samples, num_features, num_outputs]
    if values1.shape != values2.shape:
        raise ValueError(f"Shapes don't match: {values1.shape}, {values2.shape}")
    pearsons = []
    spearmans = []
    for i in range(values1.shape[0]):
        for j in range(values2.shape[2]):
            pearson = pearsonr(values1[i, :, j], values2[i, :, j])[0]
            if not np.isnan(pearson):
                pearsons.append(pearson)
            spearman = spearmanr(values1[i, :, j], values2[i, :, j])[0]
            if not np.isnan(spearman):
                spearmans.append(spearman)
    return np.array(pearsons), np.array(spearmans)


if __name__ == "__main__":
    # Generate input data
    num_features = 3
    np.random.seed(0)
    # num_features = 2
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000).astype(np.float32)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(num_features)])

    # Create model and compute ground truth Shapley values
    # model = RandomMultilinearPolynomial(num_features, [-1, -1, 5])
    model = RandomMultilinearPolynomial(num_features, [-1, -1, -1])
    print(model)
    y = model(X)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.9)
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)

    true_values = np.expand_dims(model.shapley_values(X_test.to_numpy()), -1)

    # Compute Shapley values using PDD-SHAP
    decomposition = PartialDependenceDecomposition(
        model,
        collocation_method=RandomSubsampleCollocation(),
        conditioning_method=IndependentConditioningMethod(X_train),
        # conditioning_method=GaussianConditioningMethod(X_train),
        # conditioning_method=KernelConditioningMethod(X_train.values, sigma_sq=0.1),
        estimator_type="knn",
        #est_kwargs={"k": 3},
    )
    print(X_train)
    decomposition.fit(X_train, max_size=2)
    pdd_values = decomposition.shapley_values(X_test, project=True)

    print(f"Comparing PDD-SHAP vs Ground truth:")
    pearson, spearman = correlations(pdd_values, true_values)
    r2 = r2_score(pdd_values, true_values)
    print(f"\tPearson correlation: {np.average(pearson):.2f}")
    print(f"\tSpearman correlation: {np.average(spearman):.2f}")
    print(f"\tR2: {np.average(r2):.2f}")

    print()
    for key in decomposition.components.keys():
        print(key)
