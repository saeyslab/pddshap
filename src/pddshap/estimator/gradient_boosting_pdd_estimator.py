from pddshap.estimator import PDDEstimator
from numpy import typing as npt
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingPDDEstimator(PDDEstimator):
    def fit(self, collocation_points: npt.NDArray,
            partial_dependence: npt.NDArray):
        self.regressor = GradientBoostingRegressor()
        if partial_dependence.shape[1] == 1:
            partial_dependence = partial_dependence.ravel()
        self.regressor.fit(collocation_points, partial_dependence)
