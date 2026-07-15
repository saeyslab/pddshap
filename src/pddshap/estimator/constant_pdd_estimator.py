from pddshap.estimator import PDDEstimator
from numpy import typing as npt
import numpy as np
from typing import Dict, List

from pddshap.signature.feature_subset import FeatureSubset


class ConstantPDDEstimator(PDDEstimator):
    def __init__(self, categories: Dict[int, List[int]],
                 feature_subset: FeatureSubset,
                 output: npt.NDArray):
        super().__init__(categories, feature_subset)
        self.output = output

    def __call__(self, data: npt.NDArray):
        return np.tile(self.output, (data.shape[0], 1))

