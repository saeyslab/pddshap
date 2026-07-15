"""
The collocation method is the method used to choose coordinates at which
to approximate the integral. Note that at each of these collocation points,
an expected value of the function over some distribution must be computed.
Basically, the collocation points are the input datapoints for the PDDComponent.
"""
from abc import abstractmethod
import numpy as np
from numpy import typing as npt
from sklearn import cluster


class CollocationMethod:
    @abstractmethod
    def get_collocation_points(self, data: npt.NDArray):
        raise NotImplementedError


class RandomSubsampleCollocation(CollocationMethod):
    def __init__(self, frac=1.0):
        self.frac = frac

    def get_collocation_points(self, data: npt.NDArray):
        if self.frac < 1.0:
            indices = np.random.choice(np.arange(data.shape[0]), replace=False,
                                       size=int(data.shape[0] * self.frac))
            result = data[indices, ...]
            if len(result.shape) == 1:
                result = np.expand_dims(result, axis=-1)
            return result
        return data


class KMeansCollocation(CollocationMethod):
    def __init__(self, k: int):
        self.k = k

    def get_collocation_points(self, data: npt.NDArray):
        return cluster.KMeans(n_clusters=self.k).fit(data).cluster_centers_
