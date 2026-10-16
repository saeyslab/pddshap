from pddshap.estimator.pdd_estimator import PDDEstimator
from pddshap.signature.feature_subset import FeatureSubset
from typing import Dict, List
from numpy import typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# Based on https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
class GaussianProcessPDDEstimator(PDDEstimator):
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    def fit(self, collocation_points: npt.NDArray,
            partial_dependence: npt.NDArray):
        self.regressor = GaussianProcessRegressor(
                kernel=self.kernel, n_restarts_optimizer=9, alpha=1)
        self.regressor.fit(collocation_points, partial_dependence)
