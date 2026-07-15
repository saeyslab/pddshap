from pddshap.estimator import EstimatorNotFittedException
from pddshap.estimator.pdd_estimator import PDDEstimator
from pddshap.signature import FeatureSubset
from typing import Dict, List, Optional
from numpy import typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# Based on https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
class GaussianProcessPDDEstimator(PDDEstimator):
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gaussian_process: Optional[GaussianProcessRegressor] = None

    def fit(self, collocation_points: npt.NDArray,
            partial_dependence: npt.NDArray):
        self.gaussian_process = GaussianProcessRegressor(
                kernel=self.kernel, n_restarts_optimizer=9, alpha=1)
        self.gaussian_process.fit(collocation_points, partial_dependence)

    # TODO return the std as well, create a class UncertaintyPDDEstimator
    def __call__(self, data: npt.NDArray):
        if self.gaussian_process is None:
            raise EstimatorNotFittedException()
        result = self.gaussian_process.predict(data, return_std=False)
        if len(result.shape) == 1:
            return result.reshape(-1, 1)
        return result
