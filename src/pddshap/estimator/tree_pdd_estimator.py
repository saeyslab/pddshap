from pddshap.estimator import PDDEstimator
from numpy import typing as npt
from sklearn.tree import DecisionTreeRegressor


class TreePDDEstimator(PDDEstimator):
    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        self.regressor = DecisionTreeRegressor()
        self.regressor.fit(coords, partial_dependence)
