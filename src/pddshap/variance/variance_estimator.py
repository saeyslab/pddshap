from typing import Dict, Tuple
from itertools import combinations
import numpy as np
from numpy import typing as npt

from pddshap.util.model import Model
from pddshap.signature import FeatureSubset


class SobolNotAvailableException(Exception):
    def __init__(self, feature_subset: FeatureSubset) -> None:
        super().__init__(
                f"Sobol' index not available for subset {feature_subset}")


class VarianceEstimator:
    """
    This class is used to estimate variance-based
    properties of feature subsets given a model and a dataset:
        - Lower Sobol' index [REF]
        - Cost of exclusion [REF]
        - Component variance 
        (using lower Sobol' and inclusion-exclusion principle)
    """
    def __init__(self, data: npt.NDArray, model: Model,
                 lower_sobol_strategy="lower_bound",
                 lower_sobol_threshold=0.1) -> None:
        r"""
        :param data: an NDArray containing the data to be used to estimate
            properties. Shape: (num_instances, num_features).
        :param model: the model to estimate variance properties of. Can be
            an instance of pddshapley.util.model.Model, or a Callable that
            takes 1 NDArray as argument and produces an NDArray as result.
        :param lower_sobol_strategy: the strategy to use when estimating
            lower Sobol' indices.
            - If "correlation", the Correlation estimator is always used
            - If "oracle", the Oracle estimator is always used
            - If "lower_bound" (default), an estimate of the lower bound of
              the Sobol' index is computed from Sobol' indices of strict subsets
              which are assumed to be computed earlier. If this lower bound is
              smaller than lower_sobol_threshold for any of the outputs,
              Correlation is used, otherwise Oracle is used.
              If any of the strict subsets' Sobol' indices is
              not available, a SobolNotAvailableException is raised.
            - If "rough_estimate", a first estimate is computed using the Oracle
              estimator. If this estimate is lower than lower_sobol_threshold
              for anuy of the outputs, Correlation is used to find a better 
              estimate.
        """
        self.data = data
        self.model = model
        if lower_sobol_strategy in ("correlation", "oracle", 
                                    "lower_bound", "rough_estimate"):
            self.lower_sobol_strategy = lower_sobol_strategy
        else:
            raise ValueError("Invalid value for lower_sobol_strategy")
        self.lower_sobol_threshold = lower_sobol_threshold
        self.all_features = FeatureSubset(*np.arange(data.shape[1]))

        # Need 2 series of shuffled indices because some Sobol' estimators
        # require 2 random samples instead of 1
        self.shuffled_indices = (np.arange(data.shape[0]), 
                                 np.arange(data.shape[0]))
        np.random.shuffle(self.shuffled_indices[0])
        np.random.shuffle(self.shuffled_indices[1])

        # Contains outputs of the model on the dataset where different subsets
        # of features (keys in dict) are kept while the others are randomized.
        self._model_evaluations: Tuple[
                Dict[FeatureSubset, npt.NDArray], 
                Dict[FeatureSubset, npt.NDArray]] = ({}, {})

        # Compute output on original non-shuffled data and center
        self.orig_output = self.model(data)
        if len(self.orig_output.shape) == 1:
            self.orig_output = np.expand_dims(self.orig_output, axis=1)
        self.avg_output = np.average(self.orig_output, axis=0)
        self.orig_output -= self.avg_output
        self._model_evaluations[0][self.all_features] = self.orig_output
        self._model_evaluations[1][self.all_features] = self.orig_output

        if len(self.orig_output.shape) == 1:
            self.model_variance = np.var(self.orig_output)
            self.num_outputs = 1
        else:
            self.model_variance = np.var(self.orig_output, axis=0)
            self.num_outputs = self.orig_output.shape[1]

        self.lower_sobol_index_memo: Dict[FeatureSubset, npt.NDArray] = {}

    def lower_sobol_index(self, feature_set: FeatureSubset) -> npt.NDArray:
        r"""
        Estimates the lower Sobol' index :math:`\underline{\tau}` for a given
        feature subset using the strategy given in the constructor.
        """
        if feature_set in self.lower_sobol_index_memo:
            return self.lower_sobol_index_memo[feature_set]
        if self.lower_sobol_strategy == "correlation":
            result = self._lower_sobol_index_correlation(feature_set)
        elif self.lower_sobol_strategy == "oracle":
            result = self._lower_sobol_index_oracle(feature_set)
        elif self.lower_sobol_strategy == "lower_bound":
            # Estimate a lower bound using the inclusion-exclusion principle
            lower_bound = np.zeros(self.num_outputs)
            for i in range(1, len(feature_set)):
                for subset in combinations(feature_set, i):
                    try:
                        if (len(feature_set) - len(subset)) % 2 == 0:
                            lower_bound += self.lower_sobol_index_memo[
                                    FeatureSubset(*subset)]
                        else:
                            lower_bound -= self.lower_sobol_index_memo[
                                    FeatureSubset(*subset)]
                    except KeyError:
                        raise SobolNotAvailableException(
                                FeatureSubset(*subset))
            if np.any(lower_bound < self.lower_sobol_threshold):
                result = self._lower_sobol_index_correlation(feature_set)
            else:
                result = self._lower_sobol_index_oracle(feature_set)
        else:
            # Strategy is rough_estimate
            rough_estimate = self._lower_sobol_index_oracle(feature_set)
            if np.any(rough_estimate < self.lower_sobol_threshold):
                result = self._lower_sobol_index_correlation(feature_set)
            else:
                result = rough_estimate
        self.lower_sobol_index_memo[feature_set] = result
        return np.maximum(result, 0)

    def cost_of_exclusion(self, feature_set: FeatureSubset) -> npt.NDArray:
        """
        Estimates cost of exclusion for a given feature subset.
        If the model has multiple outputs, the resulting array contains an
        estimate for each output.
        Based on eq. (11) in Liu et al. 2006: Estimating Mean Dimensionality 
        of Analysis of Variance Decompositions

        :param feature_set: The feature subset to compute CoE for.
        :return: Estimated CoE.
        """
        inner_sum = np.zeros(shape=(self.data.shape[0], self.num_outputs))
        for i in range(len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                multiplier = 1 if (len(feature_set) - len(subset)) % 2 == 0\
                        else -1
                inner_sum += multiplier * self._get_model_evaluation(
                        FeatureSubset(*subset))
        coe = np.sum(
                np.power(inner_sum, 2),
                axis=0) / (self.data.shape[0] * 2**len(feature_set))
        return np.maximum(coe/self.model_variance, 0)

    def component_variance(self, feature_set: FeatureSubset) -> npt.NDArray:
        r"""
        Uses the lower Sobol' index and the inclusion-exclusion principle to
        estimate the variance of a given component.
        If the model has multiple outputs, the resulting array contains an
        estimate for each output.

        :param feature_set: The feature set corresponding to the component to
            be measured.
        :return: An estimate of the variance of the component.
        """
        result = np.zeros(self.num_outputs)
        for i in range(1, len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                if (len(feature_set) - len(subset)) % 2 == 0:
                    result += self.lower_sobol_index(FeatureSubset(*subset))
                else:
                    result -= self.lower_sobol_index(FeatureSubset(*subset))
        return np.maximum(result, 0)

    def _lower_sobol_index_correlation(self, feature_set: FeatureSubset):
        r"""
        Estimates the lower Sobol' index :math:`\underline{\tau}` for a given 
        feature subset using the Correlation 2 method from Owen, 2013.
        This method is especially useful if the index is assumed to be small.
        If the model has multiple outputs, the resulting array contains an
        estimated Sobol' index for each output.
        See: Owen, 2013: Better Estimation of Small Sobol' Sensitivity Indices

        :param feature_set: The feature subset :math:`u` in
            :math:`\underline{\tau}^2_u`
        """
        # Variable names correspond to the notation in Owen, 2013
        f_x = self.orig_output
        fs_complement = set(feature_set.features) - set(
                self.all_features.features)
        f_z_x = self._get_model_evaluation(FeatureSubset(*fs_complement), 
                                           shuffle_idx=1)
        f_x_y = self._get_model_evaluation(feature_set, shuffle_idx=0)
        f_y = self._get_model_evaluation(FeatureSubset(), shuffle_idx=0)

        result = np.average((f_x - f_z_x)*(f_x_y - f_y), axis=0)
        return result/self.model_variance

    def _lower_sobol_index_oracle(self, feature_set: FeatureSubset):
        r"""
        Estimates the lower Sobol' index :math:`\underline{\tau}` for a given 
        feature subset using the Oracle 2 method from Owen, 2013.
        This method is especially useful if the index is assumed to be large.
        If the model has multiple outputs, the resulting array contains an
        estimated Sobol' index for each output.
        See: Owen, 2013: Better Estimation of Small Sobol' Sensitivity Indices

        :param feature_set: The feature subset :math:`u` in
            :math:`\underline{\tau}^2_u`
        """
        shuffled_output = self._get_model_evaluation(feature_set)
        # Note that the outputs given by _get_model_evaluation are already
        # centered, so we don't need to subtract \mu again
        result = np.average((self.orig_output) * 
                            (shuffled_output),
                            axis=0)
        return result/self.model_variance

    def _get_model_evaluation(self, feature_set: FeatureSubset,
                              shuffle_idx=0) -> npt.NDArray:
        """
        Returns the model evaluated on the full dataset, where the features in 
        feature_set are kept and the others are randomized.
        :param feature_set: The features to be kept
        :return: Model output on partially shuffled data.
            Shape: (self.data.shape[0], self.num_outputs)
        """
        if feature_set not in self._model_evaluations[shuffle_idx].keys():
            # Shuffle the data for all indices not in feature_set
            data = self.data[self.shuffled_indices[shuffle_idx], :]
            keep_idx = list(feature_set)
            data[:, keep_idx] = self.data[:, keep_idx]
            # Centering the data results in better numerical performance
            result = self.model(data) - self.avg_output
            if len(result.shape) == 1:
                result = np.expand_dims(result, axis=1)
            self._model_evaluations[shuffle_idx][feature_set] = result
        return self._model_evaluations[shuffle_idx][feature_set]
