from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import typing as npt
from sklearn import cluster
from tqdm import tqdm

from . import ConstantPDDComponent, PDDComponent
from .sampling import CollocationMethod, ConditioningMethod
from .signature import DataSignature, FeatureSubset
from .util import Model, SimplePartialOrdering


class PartialDependenceDecomposition:
    def __init__(
        self,
        model: Model,
        collocation_method: CollocationMethod,
        conditioning_method: ConditioningMethod,
        estimator_type: str,
        est_kwargs=None,
    ) -> None:
        self.model = model
        self.components: dict[FeatureSubset, ConstantPDDComponent | PDDComponent] = {}
        self.collocation_method = collocation_method
        self.conditioning_method = conditioning_method
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}
        self.data_signature: DataSignature | None = None

        self.bg_avg = None
        self.num_outputs = None

    def fit(
        self,
        background_data: pd.DataFrame | npt.NDArray,
        max_size: int | None = None,
        feature_sets: list[FeatureSubset] | None = None,
        kmeans: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Fit the partial dependence decomposition using a given background dataset.

        Parameters
        ----------
        background_data : pandas.DataFrame or numpy.ndarray
            Background dataset.
        max_size : int, optional
            Maximal size of subsets to be modeled. If None, `max_size` will be
            set to the number of features.
        feature_sets : list of FeatureSubset, optional
            If not None, all subsets in this list will be modeled instead of
            enumerating subsets up to `max_size`.
        kmeans : int, optional
            If not None, the background data will be clustered using k-means
            with the given number of clusters before fitting.
        n_jobs : int, default 1
            Number of parallel jobs used to fit components of the same
            cardinality.
        """

        # Argument checking and preprocessing
        self.data_signature = DataSignature(background_data)
        if isinstance(background_data, pd.DataFrame):
            background_data = background_data.to_numpy()
        if max_size is None:
            max_size = background_data.shape[1]
        if kmeans is not None:
            background_data = (
                cluster.KMeans(n_clusters=kmeans).fit(background_data).cluster_centers_
            )

        # Select subsets to be modeled if not provided explicitly
        if feature_sets is None:
            feature_sets = []
            for i in range(1, max_size + 1):
                feature_sets += [
                    FeatureSubset(*comb)
                    for comb in combinations(range(background_data.shape[1]), i)
                ]

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
                delayed(self._fit_component)(background_data, fs)
                for fs in tqdm(cur_subsets)
            )
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
        Compute the output of the decomposition at the given coordinates.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            The coordinates where we evaluate the model.

        Returns
        -------
        numpy.ndarray
            The output of the decomposition at each coordinate.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)
        result = np.zeros(shape=(data.shape[0], self.num_outputs))
        for values in pdp_values.values():
            result += values
        return result

    def evaluate(self, data: npt.NDArray) -> dict[FeatureSubset, npt.NDArray]:
        """
        Evaluate the PDP decomposition at all rows in `data`.

        Parameters
        ----------
        data : numpy.ndarray of shape (num_rows, num_features)
            Input data.

        Returns
        -------
        dict of FeatureSubset to numpy.ndarray
            Each component function's value, as an array of shape
            (num_rows, num_outputs).
        """
        return {
            subset: component(data) for subset, component in self.components.items()
        }

    def _asv_coefficients(
        self, feature_subset: FeatureSubset, partial_ordering: SimplePartialOrdering
    ):
        result = []
        for feature in feature_subset:
            if partial_ordering.contains_successor(feature, feature_subset):
                result.append(0)
            else:
                incomparables = FeatureSubset(
                    *partial_ordering.get_incomparables(feature)
                )
                result.append(len(incomparables.intersection(feature_subset)))
        return np.array(result)

    def shapley_values(
        self,
        data: pd.DataFrame | npt.NDArray,
        project=False,
        partial_ordering: list[list[int | str]] | None = None,
    ) -> npt.NDArray:
        """
        Compute Shapley values for each row in `data`.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray of shape (num_rows, num_features)
            Input data.
        project : bool, default False
            Whether the results should be orthogonally projected onto the
            hyperplane satisfying the efficiency axiom.
        partial_ordering : list of list of int or str, optional
            Groups of features ranked by precedence, used to compute
            asymmetric Shapley values. If None, symmetric Shapley values are
            computed.

        Returns
        -------
        numpy.ndarray of shape (num_rows, num_features, num_outputs)
            Shapley values for each row and each output.
        """
        assert self.data_signature is not None, (
            "Must fit model before computing Shapley values"
        )
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)

        if partial_ordering is not None:
            simple_partial_ordering = SimplePartialOrdering(
                partial_ordering, self.data_signature
            )

        result = np.zeros(shape=(data.shape[0], data.shape[1], self.num_outputs))
        for feature_subset, output_vector in pdp_values.items():
            if len(feature_subset) > 0:
                component_effect = np.expand_dims(output_vector, axis=1)
                features = feature_subset.features
                if partial_ordering is not None:
                    coef = self._asv_coefficients(
                        feature_subset, simple_partial_ordering
                    )
                    nonzero_features = np.array(features)[coef != 0]
                    component_effect = np.tile(
                        component_effect, (1, len(nonzero_features), 1)
                    )
                    result[:, nonzero_features, :] += component_effect / coef[
                        coef != 0
                    ].reshape(1, -1, 1)
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
