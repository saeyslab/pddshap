from abc import abstractmethod
from numpy import typing as npt

from pddshap.signature import FeatureSubset
from typing import Dict, List


class PDDEstimator:
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset):
        self.feature_subset = feature_subset
        self.categories = categories

    @abstractmethod
    def fit(self, collocation_points: npt.NDArray, partial_dependence: npt.NDArray):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

