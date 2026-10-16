from collections.abc import Collection
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import typing as npt
from tqdm import tqdm

from .estimator import (
    GaussianProcessPDDEstimator,
    GradientBoostingPDDEstimator,
    KNNPDDEstimator,
    PDDEstimator,
    TreePDDEstimator,
)
from .sampling import CollocationMethod, ConditioningMethod
from .signature import Dataset
from .signature.feature_subset import FeatureSubset
from .util import Model, SimplePartialOrdering

ESTIMATOR_REGISTRY: dict[str, type[PDDEstimator]] = {
    "tree": TreePDDEstimator,
    "forest": GradientBoostingPDDEstimator,
    "knn": KNNPDDEstimator,
    "gp": GaussianProcessPDDEstimator,
}


class PartialDependenceDecomposition:
    def __init__(
        self,
        model: Model,
        collocation_method: CollocationMethod,
        conditioning_method: ConditioningMethod,
        regressor_type: str | type[PDDEstimator],
        regressor_kwargs: dict or None = None,
        use_onehot: bool = False
    ) -> None:
        self.model = model
        self.collocation_method = collocation_method
        self.conditioning_method = conditioning_method
        self.regressor_type = (
            ESTIMATOR_REGISTRY[regressor_type]
            if isinstance(regressor_type, str)
            else regressor_type
        )
        self.regressor_kwargs = regressor_kwargs if regressor_kwargs is not None else {}
        self.use_onehot = use_onehot

        self.dataset: Dataset | None = None
        self.components: dict[FeatureSubset, PDDEstimator] = {}
        self.bg_avg = None
        self.num_outputs = None

    def fit(
        self,
        X_train: pd.DataFrame | npt.NDArray,
        max_size: int | None = None,
        feature_sets: list[Collection[int | str]] | None = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Fit the partial dependence decomposition using a given background dataset.

        Parameters
        ----------
        X_train : pandas.DataFrame or numpy.ndarray
            Dataset to train the PDD surrogate model on.
        max_size : int, optional
            Maximal size of subsets to be modeled. If None, `max_size` will be
            set to the number of features.
        feature_sets : list of collection of int or str, optional
            If not None, all subsets in this list will be modeled instead of
            enumerating subsets up to `max_size`. Each subset is a collection
            of feature indices and/or names (e.g. `[(0, 1), ("a",)]`).
        n_jobs : int, default 1
            Number of parallel jobs used to fit components of the same
            cardinality.
        """

        # Argument checking and preprocessing
        self.dataset = Dataset(X_train, onehot_enable=self.use_onehot)
        if max_size is None:
            max_size = self.dataset.num_features

        # Select subsets to be modeled if not provided explicitly
        if feature_sets is None:
            subsets = []
            for i in range(1, max_size + 1):
                subsets += [
                    FeatureSubset(*comb)
                    for comb in combinations(range(self.dataset.num_features), i)
                ]
        else:
            subsets = [self.dataset.get_subset(*fs) for fs in feature_sets]

        # The constant term of the decomposition: the model's average output
        # over the background data. Every component is fitted against the
        # partial dependence relative to this baseline (see `_fit_component`).
        output = self.model(self.dataset.data)
        self.num_outputs = 1 if len(output.shape) == 1 else output.shape[1]
        self.bg_avg = np.average(output, axis=0)

        # Model the selected subsets in parallel by cardinality
        fs_size = 1
        cur_subsets = [fs for fs in subsets if len(fs) == fs_size]
        while len(cur_subsets) > 0:
            print("Fitting components of size", fs_size, "...")
            fitted_estimators = Parallel(n_jobs=n_jobs)(
                delayed(self._fit_component)(fs)
                for fs in tqdm(cur_subsets)
            )
            for fs, estimator in zip(cur_subsets, fitted_estimators):
                self.components[fs] = estimator
            fs_size += 1
            cur_subsets = [fs for fs in subsets if len(fs) == fs_size]

    def _fit_component(self, feature_set: FeatureSubset) -> PDDEstimator:
        """
        Fit an estimator for `feature_set` against the residual partial
        dependence left after subtracting every already-fitted strict-subset
        component. Kept as a separate method (rather than inlined in `fit()`)
        so it can be passed to `joblib.Parallel` for parallel fitting of
        same-cardinality subsets.
        """
        # All subcomponents are necessary to compute the values
        # for this component
        subcomponents = {
            k: v
            for k, v in self.components.items()
            if all(feat in feature_set for feat in k)
        }

        # Collocation points are the points where we compute partial dependence
        collocation_points = self.collocation_method.get_collocation_points(
            self.dataset.extract_columns(feature_set)
        )

        # For each collocation point, get partial dependence at this point
        # Shape: (num_rows, num_outputs)
        partial_dependence = np.average(
            np.array(
                [
                    self.conditioning_method.conditional_expectation(
                        feature_set, collocation_point, self.model
                    )
                    for collocation_point in collocation_points
                ]
            ),
            axis=1,
        )

        # Subtract the constant term and every strict subcomponent's output
        # from the partial dependence. The result is the estimated output for
        # the functional component.
        partial_dependence -= self.bg_avg
        full_width_collocation_points = self.dataset.expand_columns(
            feature_set, collocation_points
        )
        for subset, regressor in subcomponents.items():
            partial_dependence -= regressor(
                self.dataset.extract_columns(subset, full_width_collocation_points)
            )

        # Fit an estimator on the resulting values
        categories = self.dataset.get_categories(feature_set)
        regressor = self.regressor_type(categories, feature_set, **self.regressor_kwargs)
        regressor.fit(collocation_points, partial_dependence)
        return regressor

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
            The output of the decomposition at each coordinate: the constant
            term plus every fitted component's contribution.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)
        result = np.tile(self.bg_avg, (data.shape[0], 1))
        for values in pdp_values.values():
            result += values
        return result

    def evaluate(self, data: npt.NDArray) -> dict[FeatureSubset, npt.NDArray]:
        """
        Evaluate the PDP decomposition's fitted components at all rows in
        `data`. Does not include the constant term (`self.bg_avg`), since
        that isn't attributable to any feature subset; see `__call__` to
        reconstruct the full model output including it.

        Parameters
        ----------
        data : numpy.ndarray of shape (num_rows, num_features)
            Input data.

        Returns
        -------
        dict of FeatureSubset to numpy.ndarray
            Each non-constant component function's value, as an array of
            shape (num_rows, num_outputs).
        """
        return {
            subset: estimator(self.dataset.extract_columns(subset, data))
            for subset, estimator in self.components.items()
        }

    def component_summary(self, data: pd.DataFrame | npt.NDArray) -> pd.DataFrame:
        """
        Compute summary statistics of each fitted component's output over
        `data`, as a quick way to see which feature subsets contribute most
        to the model's output.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Input data to evaluate the components on.

        Returns
        -------
        pandas.DataFrame
            One row per fitted component, indexed by the names of the
            features in that component's subset, with columns `mean`,
            `std`, `min`, `max`, `q90`, `q10`.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)

        index = [
            tuple(self.dataset.feature_names[i] for i in feature_subset.features)
            for feature_subset in pdp_values
        ]
        rows = [
            {
                "mean": np.mean(output),
                "std": np.std(output),
                "min": np.min(output),
                "max": np.max(output),
                "q90": np.quantile(output, 0.9),
                "q10": np.quantile(output, 0.1),
            }
            for output in pdp_values.values()
        ]
        return pd.DataFrame(rows, index=index)

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
        assert self.dataset is not None, (
            "Must fit model before computing Shapley values"
        )
        if isinstance(data, pd.DataFrame):
            data = data.values
        pdp_values = self.evaluate(data)

        if partial_ordering is not None:
            simple_partial_ordering = SimplePartialOrdering(
                partial_ordering, self.dataset
            )

        result = np.zeros(
            shape=(data.shape[0], self.dataset.num_features, self.num_outputs)
        )
        for feature_subset, output_vector in pdp_values.items():
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
            adjustment /= self.dataset.num_features
            return result - adjustment
        return result
