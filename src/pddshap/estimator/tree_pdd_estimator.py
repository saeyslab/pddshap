from pddshap.estimator import PDDEstimator, EstimatorNotFittedException
from pddshap.signature import FeatureSubset
from numpy import typing as npt
from typing import Dict, List, Optional
from sklearn.tree import DecisionTreeRegressor


class TreePDDEstimator(PDDEstimator):
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.tree: Optional[DecisionTreeRegressor] = None

    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        self.tree = DecisionTreeRegressor()
        self.tree.fit(coords, partial_dependence)

    def __call__(self, data: npt.NDArray):
        if self.tree is None:
            raise EstimatorNotFittedException()
        result = self.tree.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
