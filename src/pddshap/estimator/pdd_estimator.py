from abc import abstractmethod
from numpy import typing as npt

from pddshap.signature.feature_subset import FeatureSubset
from typing import Dict, List, Self
from pddshap.estimator import EstimatorNotFittedException
from sklearn.base import RegressorMixin

class PDDEstimator:
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        self.feature_subset = feature_subset
        self.categories = categories
        self.regressor: RegressorMixin | None = None

    @abstractmethod
    def fit(self, collocation_points: npt.NDArray, partial_dependence: npt.NDArray) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: npt.NDArray) -> npt.NDArray:
        if self.regressor is None:
            raise EstimatorNotFittedException()
        result = self.regressor.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
