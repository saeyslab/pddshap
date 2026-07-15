"""
TODO add module docstring
"""

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import cluster
from tqdm import tqdm

from . import ConstantPDDComponent, PDDComponent
from .sampling import CollocationMethod, ConditioningMethod
from .signature import DataSignature, FeatureSubset
from .util import Model, SimplePartialOrdering
from .variance import COETracker, VarianceEstimator


class PartialDependenceDecomposition:
    """
    TODO add class docstring
    """

    def __init__(
        self,
        model: Model,
        collocation_method: CollocationMethod,
        conditioning_method: ConditioningMethod,
        estimator_type: str,
        est_kwargs=None,
    ) -> None:
        self.model = model
        self.components: Dict[
            FeatureSubset, Union[ConstantPDDComponent, PDDComponent]
        ] = {}
        self.collocation_method = collocation_method
        self.conditioning_method = conditioning_method
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}
        self.data_signature: Optional[DataSignature] = None

        self.bg_avg = None
        self.num_outputs = None

    def _get_significant_feature_sets(
        self, data: npt.NDArray, model: Model, variance_explained: float, max_size: int
    ) -> List[FeatureSubset]:
        """
        Computes all subsets (up to a given cardinality) that should be
        incorporated in an ANOVA decomposition model in order to explain a given
        fraction of the variance.

        :param variance_explained: Desired fraction of variance modeled
            by the components
        :param max_cardinality: Maximal cardinality of subsets.
        :return: Dictionary containing significant subsets for each
            cardinality: {int: List[Tuple]}
        """

        # Maps cardinality to included feature sets of that cardinality
        # and their estimated component variance for each output
        result: List[FeatureSubset] = []
        variance_estimator = VarianceEstimator(
            data, model, lower_sobol_strategy="lower_bound", lower_sobol_threshold=0.1
        )
        # Current fraction of variance explained for each output
        cur_var_explained = np.zeros(variance_estimator.num_outputs)
        num_columns = data.shape[1]

        # Keep track of CoE values for candidate components, while taking into
        # account possible multiple outputs
        tracker = COETracker(num_columns, variance_estimator.num_outputs)

        # We start with all singleton subsets
        for i in range(num_columns):
            # coe contains CoE for each output
            coe = variance_estimator.cost_of_exclusion(FeatureSubset(i))
            tracker.push(FeatureSubset(i), coe)

        # subset_counts contains the number of immediate subsets for each
        # feature set that have been included.
        # If all immediate subsets of a feature set are included, then that
        # feature set should be added to the queue.
        subset_counts: Dict[FeatureSubset, int] = defaultdict(lambda: 0)

        # Add subsets in order of decreasing CoE until the desired fraction of
        # variance has been included
        while np.any(tracker.active_outputs) and not tracker.empty():
            # Pop the next feature subset to be modeled, which is the one
            # having the largest CoE over all active outputs
            feature_subset = tracker.pop()
            print(feature_subset)
            if len(feature_subset) <= max_size:
                # Compute component variance and add it to variance explained
                component_variance = variance_estimator.component_variance(
                    feature_subset
                )
                component_variance = np.maximum(component_variance, 0)
                cur_var_explained += component_variance
                # Add feature subset + its variance to the result
                result.append(feature_subset)
                # Increment subset_counts for each immediate superset of
                # feature_subset
                for i in range(num_columns):
                    if i not in feature_subset:
                        superset = FeatureSubset(i, *feature_subset)
                        subset_counts[superset] += 1
                        # If all immediate subsets of superset have been
                        # included, add superset to queue
                        if subset_counts[superset] == len(superset):
                            coe = variance_estimator.cost_of_exclusion(superset)
                            tracker.push(superset, coe)
            # Refresh active outputs
            tracker.active_outputs = cur_var_explained < variance_explained
        # Sort subsets in result by size
        result.sort(key=len)
        return result

    def fit(
        self,
        background_data: pd.DataFrame,
        feature_set_selection: str = "max_size",
        variance_explained: Optional[float] = None,
        coe_threshold: Optional[float] = None,
        max_size: Optional[int] = None,
        feature_sets: Optional[List[FeatureSubset]] = None,
        kmeans: Optional[int] = None,
        n_jobs: Optional[int] = 1,
    ) -> None:
        """
        Fit the partial dependence decomposition using a given
        background dataset.

        :param background_data: Background dataset
        :param max_size: Maximal size of subsets to be modeled.
            If None, max_size will be set to the number of features.
        :param feature_set_selection: Method for selecting subsets to be
            modeled. Options:
            - "max_size": All subsets of size up to max_size will be modeled.
            - "coe_threshold": All subsets up to max_size will be modeled,
                but only if their cost of exclusion is above a given threshold.
                Requires coe_threshold to be set.
            - "coe_var_explained": All subsets up to max_size will be modeled,
                until a given fraction of explained variance is reached.
                Requires variance_explained to be set.
        :param variance_explained: Fraction of variance to be explained by
            the model. Only used if feature_set_selection is "coe_frac_var".
        :param coe_threshold: Threshold for cost of exclusion. Only used if
            feature_set_selection is "coe_threshold".
        :param kmeans: If not None, the background data will be clustered
            using k-means with the given number of clusters before fitting.
        :param feature_sets: If not None, all subsets in this dictionary will
            be modeled. The keys of the dictionary should be the sizes of
            the subsets, and the values should be lists of FeatureSubset objects.
        """

        # Argument checking and preprocessing
        assert feature_set_selection in [
            "max_size",
            "coe_threshold",
            "coe_var_explained",
        ], f"Invalid feature_set_selection: {feature_set_selection}"
        if feature_set_selection == "coe_threshold":
            assert coe_threshold is not None, "coe_threshold must be set"
        if feature_set_selection == "coe_var_explained":
            assert variance_explained is not None, "variance_explained must be set"
        self.data_signature = DataSignature(background_data)
        if isinstance(background_data, pd.DataFrame):
            background_data = background_data.to_numpy()
        if max_size is None:
            max_size = background_data.shape[1]
        if kmeans is not None:
            background_data = cluster.KMeans(n_clusters=kmeans).fit(background_data).cluster_centers_

        # Select subsets to be modeled
        # If feature_sets is not None, we use the given feature sets
        # Otherwise, we use the feature_set_selection method
        if feature_sets is None:
            if feature_set_selection == "max_size":
                feature_sets = []
                for i in range(1, max_size + 1):
                    feature_sets += list(
                        FeatureSubset(*comb)
                        for comb in combinations(range(background_data.shape[1]), i)
                    )
            elif feature_set_selection == "coe_threshold" and coe_threshold is not None:
                # TODO use coe_threshold
                feature_sets = self._get_significant_feature_sets(
                    background_data, self.model, variance_explained, max_size
                )
            elif (
                feature_set_selection == "coe_var_explained"
                and variance_explained is not None
            ):
                # TODO use component_variance to see if we need to model
                # each component
                feature_sets = self._get_significant_feature_sets(
                    background_data, self.model, variance_explained, max_size
                )

        # First, model the empty component
        empty_component = ConstantPDDComponent(FeatureSubset(), self.data_signature)
        self.bg_avg = empty_component.fit(background_data, self.model)
        self.components[FeatureSubset()] = empty_component
        self.num_outputs = empty_component.num_outputs

        # Model the selected subsets in parallel by cardinality
        fs_size = 1
        cur_subsets = [fs for fs in feature_sets if len(fs) == fs_size]
        while len(cur_subsets) > 0:
            print("Fitting components of size", fs_size, "...")
            Parallel(n_jobs=n_jobs)(
                delayed(self._fit_component)(
                    background_data, fs) for fs in tqdm(cur_subsets))
            fs_size += 1
            cur_subsets = [fs for fs in feature_sets if len(fs) == fs_size]
    
    def _fit_component(self, background_data: npt.NDArray, feature_set: FeatureSubset):
        # All subcomponents are necessary to compute the values
        # for this component
        subcomponents = {
            k: v
            for k, v in self.components.items()
            if all(feat in feature_set for feat in k)
        }
        component = PDDComponent(
            feature_set,
            self.data_signature,
            self.collocation_method,
            self.conditioning_method,
            self.estimator_type,
            self.est_kwargs,
        )
        component.fit(background_data, self.model, subcomponents)
        self.components[feature_set] = component


    def __call__(self, data: pd.DataFrame | npt.NDArray):
        """
        Compute the output of the decomposition at the given coordinates
        :param data: The coordinates where we evaluate the model
        :return: The output of the decomposition at each coordinate.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)
        result = np.zeros(shape=(data.shape[0], self.num_outputs))
        for _, values in pdp_values.items():
            result += values
        return result

    def evaluate(self, data: npt.NDArray) -> Dict[FeatureSubset, npt.NDArray]:
        """
        Evaluate PDP decomposition at all rows in data
        :param data: [num_rows, num_features]
        :return: Dictionary containing each component function value:
            {FeatureSubset, npt.NDArray[num_rows, num_outputs]}
        """
        return {
            subset: component(data) for subset, component in self.components.items()
        }

    def _asv_coefficients(self, feature_subset: FeatureSubset,
                          partial_ordering: SimplePartialOrdering):
        result = []
        for feature in feature_subset:
            if partial_ordering.contains_successor(feature, feature_subset):
                result.append(0)
            else:
                incomparables = FeatureSubset(*partial_ordering.get_incomparables(feature))
                result.append(len(incomparables.intersection(feature_subset)))
        return np.array(result)


    def shapley_values(
        self, data: pd.DataFrame | npt.NDArray, project=False,
        partial_ordering: Optional[List[List[Union[int, str]]]] = None
    ) -> npt.NDArray:
        """
        Compute Shapley values for each row in data.
        :param data: DataFrame or NDArray, shape: (num_rows, self.num_features)
        :param project: Boolean value indicating if the results should be
            orthogonally projected to the hyperplane satisfying the
            Efficiency axiom.
        :return: NDArray containing Shapley values for each row and each output.
            Shape: (num_rows, self.num_features, num_outputs)
        """
        assert self.data_signature is not None, "Must fit model before computing Shapley values"
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)

        if partial_ordering is not None:
            simple_partial_ordering = SimplePartialOrdering(
                partial_ordering, self.data_signature)

        result = np.zeros(shape=(data.shape[0], data.shape[1], self.num_outputs))
        for feature_subset, output_vector in pdp_values.items():
            if len(feature_subset) > 0:
                component_effect = np.expand_dims(output_vector, axis=1)
                features = feature_subset.features
                if partial_ordering is not None:
                    coef = self._asv_coefficients(feature_subset, simple_partial_ordering)
                    nonzero_features = np.array(features)[coef != 0]
                    component_effect = np.tile(component_effect, (1, len(nonzero_features), 1))
                    result[:, nonzero_features, :] += component_effect / coef[coef != 0].reshape(1, -1, 1)
                else:
                    result[:, features, :] += component_effect / len(feature_subset)

        if project:
            # Orthogonal projection of Shapley values onto hyperplane
            # x_1 + ... + x_d = c where c is the prediction difference
            pred_diff = self.model(data) - self.bg_avg
            pred_diff = pred_diff.reshape(-1, 1, result.shape[-1])
            adjustment = np.sum(result, axis=1, keepdims=True) - pred_diff
            adjustment /= data.shape[1]
            return result - adjustment
        return result
