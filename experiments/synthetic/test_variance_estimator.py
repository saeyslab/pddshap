from pddshap.variance import VarianceEstimator
from pddshap.signature import FeatureSubset
import numpy as np
from experiments.synthetic.multilinear_polynomial import MultilinearPolynomial


if __name__ == "__main__":
    np.random.seed(0)
    num_features = 2
    """
    Ground truth:
        - Model variance: 4.26
        - 0:
            - Lower sobol: 4 (0.939)
            - COE: 4.01 (0.941)
            - Variance: 4 (0.939)
        - 1:
            - Lower sobol: .25 (0.059)
            - COE: .26 (0.061)
            - Variance: .25 (0.059)
        - 0, 1:
            - Lower sobol: 4.26 (1)
            - COE: .01 (0.002)
            - Variance: .01 (0.002)
    """
    model = MultilinearPolynomial(num_features, coefficients={
        FeatureSubset(): 3.,
        FeatureSubset(0): 2.,
        FeatureSubset(1): .5,
        FeatureSubset(0, 1): .1
    })
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    feature_sets = (FeatureSubset(*s) for s in ((), (0,), (1,), (0,1)))
    X = np.random.multivariate_normal(mean, cov, size=10000).astype(np.float32)

    est = VarianceEstimator(X, model, lower_sobol_strategy="lower_bound")
    print(f"Estimated variance: {est.model_variance}")
    
    for fs in feature_sets:
        print(f"Lower Sobol index {fs}: {est.lower_sobol_index(fs)}")
        print(f"COE {fs}: {est.cost_of_exclusion(fs)}")
        print(f"Component variance {fs}: {est.component_variance(fs)}")
        print()
