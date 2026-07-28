import numpy as np
from numpy import typing as npt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

from pddshap.estimator import EstimatorNotFittedException, PDDEstimator
from pddshap.signature import FeatureSubset


class KNNPDDEstimator(PDDEstimator):
    def __init__(
        self, categories: dict[int, list[int]], feature_subset: FeatureSubset, k=3
    ):
        super().__init__(categories, feature_subset)
        self.k = k
        self.knn: KNeighborsRegressor | None = None
        self.categories = categories

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

    def fit(self, collocation_points: np.ndarray, partial_dependence: np.ndarray):
        collocation_points = self._preprocess(collocation_points)
        self.knn = KNeighborsRegressor(n_neighbors=self.k)
        self.knn.fit(collocation_points, partial_dependence)

    def __call__(self, data):
        if self.knn is None:
            raise EstimatorNotFittedException()
        result = self.knn.predict(self._preprocess(data))
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
