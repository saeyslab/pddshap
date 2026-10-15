from pddshap.estimator import PDDEstimator
from numpy import typing as npt
import numpy as np

from pddshap.signature.feature_subset import FeatureSubset


class ConstantPDDEstimator(PDDEstimator):
    def __init__(self, output: npt.NDArray):
        super().__init__({}, FeatureSubset())
        self.output = output

    def __call__(self, data: npt.NDArray):
        return np.tile(self.output, (data.shape[0], 1))

