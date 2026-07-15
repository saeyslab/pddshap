from pddshap.estimator import PDDEstimator, EstimatorNotFittedException
from pddshap.signature import FeatureSubset
from numpy import typing as npt
from typing import Optional, Dict, List
from sklearn.ensemble import RandomForestRegressor


class ForestPDDEstimator(PDDEstimator):
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.forest: Optional[RandomForestRegressor] = None

    def fit(self, collocation_points: npt.NDArray,
            partial_dependence: npt.NDArray):
        self.forest = RandomForestRegressor()
        self.forest.fit(collocation_points, partial_dependence)

    def __call__(self, data: npt.NDArray):
        if self.forest is None:
            raise EstimatorNotFittedException()
        result = self.forest.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
