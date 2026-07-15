"""
This module implements different methods for conditioning on a subset of the
input features. This includes using the marginal distribution (as in KernelSHAP),
or using the conditional distribution using various models of conditional
probability. See the following paper for more details:

Aas, K., Jullum, M., & Løland, A. (2019).  Explaining individual predictions
 when features are dependent: More accurate approximations to Shapley values.
"""

from abc import abstractmethod
import numpy as np
from numpy import typing as npt
from ..signature import FeatureSubset
from typing import Callable


class ConditioningMethod:
    def __init__(self, train_data: npt.NDArray, **kwargs) -> None:
        self.train_data = train_data
    
    @abstractmethod
    def _conditional_expectation(self,
                                 feature_subset: FeatureSubset, value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray],
                                 **kwargs) -> npt.NDArray:
        raise NotImplementedError

    def conditional_expectation(self,
                                 feature_subset: FeatureSubset,
                                 value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray],
                                 **kwargs) -> npt.NDArray:
        result = self._conditional_expectation(feature_subset, value, model, **kwargs)
        if len(result.shape) == 1:
            # If there is only 1 output, the dimension must be added
            result = np.expand_dims(result, axis=-1)
        return result


class IndependentConditioningMethod(ConditioningMethod):
    """
    This conditioning method assumes independence between the features.
    In this case, conditioning on a feature is equivalent to sampling from
    the marginal distribution of the other features.
    This corresponds to off-manifold Shapley values.
    """
    def _conditional_expectation(self, 
                                 feature_subset: FeatureSubset,
                                 value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray], 
                                 **kwargs) -> npt.NDArray:
        return model(feature_subset.project(self.train_data, value))


class GaussianConditioningMethod(ConditioningMethod):
    def __init__(self, train_data: npt.NDArray, **kwargs) -> None:
        super().__init__(train_data, **kwargs)
        self.cov = np.cov(train_data, rowvar=False)
        self.mean = np.mean(train_data, axis=0)
        self.num_features = train_data.shape[1]
    
    def _conditional_expectation(self, feature_subset: FeatureSubset,
                                 value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray],
                                 num_samples=100, **kwargs) -> npt.NDArray:
        if len(feature_subset) < self.num_features:
            x = feature_subset.features
            y = np.setdiff1d(np.arange(self.num_features), x)
            # Compute the conditional covariance matrix (y given x)
            cov_xx = self.cov[x, :][:, x]
            cov_yx = self.cov[y, :][:, x]
            cov_xy = self.cov[x, :][:, y]
            cov_yy = self.cov[y, :][:, y]
            cov_cond = cov_yy - cov_yx @ np.linalg.inv(cov_xx) @ cov_xy

            # Compute the conditional mean
            mu_y = self.mean[y]
            mu_x = self.mean[x]
            mean_cond = mu_y + cov_yx @ np.linalg.inv(cov_xx) @ (value - mu_x)

            # Sample from the conditional distribution
            result = np.zeros((num_samples, self.num_features))
            result[:, y] = np.random.multivariate_normal(mean_cond, cov_cond, num_samples)
            result[:, x] = value
            return model(result)
        return model(value.reshape(1, -1))


class KernelConditioningMethod(ConditioningMethod):
    def __init__(self, train_data: npt.NDArray, sigma_sq=0.1, **kwargs) -> None:
        super().__init__(train_data, **kwargs)
        self.sigma_sq = sigma_sq
        self.inv_cov = np.linalg.inv(np.cov(train_data, rowvar=False))
    
    def _conditional_expectation(self, feature_subset: FeatureSubset,
                                    value: npt.NDArray,
                                    model: Callable[[npt.NDArray], npt.NDArray],
                                    **kwargs) -> npt.NDArray:
            # TODO if necessary, add a K parameter to decide on the number of
            # TODO samples (for now just using full background distribution)

            # TODO computing mahalanobis distances is biggest bottleneck,
            # TODO this can probably be improved by better vectorizing
            # https://stackoverflow.com/questions/55094599/speeding-up-mahalanobis-distance-calculation
            fs = feature_subset.features
            if len(feature_subset) < self.train_data.shape[1]:
                distances = self.train_data[:, fs] - value
                sq_mahalanobis = distances @ self.inv_cov[fs, :][:, fs] @ distances.T / len(fs)
                weights = np.exp(-sq_mahalanobis.diagonal() / (2 * self.sigma_sq))
                return np.average(model(self.train_data), axis=0, 
                                           weights=weights, keepdims=True)
            return model(value.reshape(1, -1))


class KernelGaussianConditioningMethod(ConditioningMethod):
    def __init__(self, train_data: npt.NDArray, threshold=3, **kwargs) -> None:
        super().__init__(train_data, **kwargs)
        self.kernel_method = KernelConditioningMethod(train_data, **kwargs)
        self.gaussian_method = GaussianConditioningMethod(train_data, **kwargs)
        self.threshold = threshold
    
    def _conditional_expectation(self, feature_subset: FeatureSubset,
                                 value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray],
                                 **kwargs) -> npt.NDArray:
        if len(feature_subset) < self.threshold:
            return self.kernel_method._conditional_expectation(feature_subset, value, model, **kwargs)
        return self.gaussian_method._conditional_expectation(feature_subset, value, model, **kwargs)


class GaussianCopulaConditioningMethod(ConditioningMethod):
    ...