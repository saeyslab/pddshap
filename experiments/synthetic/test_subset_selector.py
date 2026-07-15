from pddshap.variance import FeatureSubsetSelector
from pddshap.signature import FeatureSubset
from experiments.synthetic.multilinear_polynomial import MultilinearPolynomial
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    num_features = 2
    model = MultilinearPolynomial(num_features, coefficients={
        FeatureSubset(): 3.,
        FeatureSubset(0): 2.,
        FeatureSubset(1): .5,
        FeatureSubset(0, 1): .1
    })
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=10000).astype(np.float32)

    subset_selector = FeatureSubsetSelector(X, model)
    variance_estimator = subset_selector.variance_estimator
    subsets = subset_selector.get_significant_feature_sets(
            desired_variance_explained=0.99, max_cardinality=X.shape[1])

    print(model)
    print("Subset: CoE Sobol Var")
    print("=================")
    for card in subsets.keys():
        for subset, variance in subsets[card]:
            print(f"{subset}: "
                  f"{variance_estimator.cost_of_exclusion(subset)}"
                  f"\t{variance_estimator.lower_sobol_index(subset)}"
                  f"\t{variance}")
