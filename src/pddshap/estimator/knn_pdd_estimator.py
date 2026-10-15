import numpy as np
from numpy import typing as npt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

from pddshap.estimator import PDDEstimator, EstimatorNotFittedException
from pddshap.signature import FeatureSubset


class KNNPDDEstimator(PDDEstimator):
    def __init__(
        self, categories: dict[int, list[int]], feature_subset: FeatureSubset, k=3
    ):
        super().__init__(categories, feature_subset)
        self.k = k

    def _preprocess(self, data: npt.NDArray):
        columns = []
        for i, feat in enumerate(self.feature_subset):
            if feat not in self.categories:
                columns.append(data[:, i].reshape(-1, 1))
            else:
                ohe = OneHotEncoder(
                    categories=[self.categories[feat]],
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                columns.append(ohe.fit_transform(data[:, i].reshape(-1, 1)))
        return np.concatenate(columns, axis=1)


    def __call__(self, data: npt.NDArray):
        if self.regressor is None:
            raise EstimatorNotFittedException()
        result = self.regressor.predict(self._preprocess(data))
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result

    def fit(self, collocation_points: np.ndarray, partial_dependence: np.ndarray):
        collocation_points = self._preprocess(collocation_points)
        self.regressor = KNeighborsRegressor(n_neighbors=self.k)
        self.regressor.fit(collocation_points, partial_dependence)
