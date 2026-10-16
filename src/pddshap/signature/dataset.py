from collections.abc import Collection

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.preprocessing import OneHotEncoder

from .feature_subset import FeatureSubset


class Dataset:
    """
    A light wrapper around a numpy array, used internally by
    `PartialDependenceDecomposition` instead of a pandas DataFrame for
    performance. Keeps track of feature names and which columns are
    categorical vs. continuous, and exposes the column operations needed to
    fit and evaluate a decomposition: selecting a feature subset's columns
    (`extract_columns`), zero-padding them back out to full width
    (`expand_columns`), and overwriting them with fixed values to compute a
    partial dependence (`overwrite`).

    If `onehot_enable` is True, categorical features are one-hot encoded
    internally, so a single categorical feature can then occupy multiple
    columns of `self.data` (one per category) instead of a single
    integer-coded column. All column operations account for this
    automatically, so callers can keep addressing features by their
    original index without caring how many raw columns back them.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The data to wrap. If a DataFrame, feature names are taken from the
        columns and categorical features are inferred from column dtypes
        (int8/int64 columns are categorical, float32/float64 are
        continuous). If a numpy array, `categorical_features` must be used
        to indicate which columns are categorical; all other columns are
        treated as continuous.
    categorical_features : collection of int, optional
        Indices of categorical columns (ignored if `data` is a DataFrame).
    onehot_enable : bool, default False
        Whether to one-hot encode categorical features internally.
    """

    def __init__(
        self,
        data: pd.DataFrame | npt.NDArray,
        categorical_features: Collection[int] | None = None,
        onehot_enable: bool = False,
    ) -> None:
        self.onehot_enable = onehot_enable
        self.categories: dict[int, list[int]] = {}

        if isinstance(data, pd.DataFrame):
            self.feature_names = tuple(data.columns)
            if any(
                dt not in ["int8", "int64", "float32", "float64"]
                for dt in data.dtypes
            ):
                raise ValueError(
                    "Encode categorical values as int8 and numerical as float32 or float64"
                )
            for i, feat_name in enumerate(self.feature_names):
                if data.dtypes[feat_name] in ["int8", "int64"]:
                    self.categories[i] = list(range(int(data[feat_name].max()) + 1))
            raw = data.to_numpy()
        else:
            self.feature_names = tuple(str(i) for i in range(data.shape[1]))
            for feat in categorical_features or []:
                self.categories[feat] = sorted(
                    int(v) for v in np.unique(data[:, feat])
                )
            raw = data

        self.num_features = len(self.feature_names)
        self.data, self._raw_columns = self._encode(raw)

    def _encode(
        self, raw: npt.NDArray
    ) -> tuple[npt.NDArray, dict[int, list[int]]]:
        """
        Build `self.data` and the column map `{feature: raw columns}`
        (identity, one raw column per feature, unless one-hot encoding
        expands a categorical feature into one raw column per category).
        The `OneHotEncoder` fitted for each categorical feature is kept
        around in `self._encoders` so later raw-value encoding (`overwrite`)
        reuses it instead of re-deriving the category layout.
        """
        self._encoders: dict[int, OneHotEncoder] = {}
        if not self.onehot_enable or not self.categories:
            raw_columns = {i: [i] for i in range(self.num_features)}
            return raw.astype(float, copy=True), raw_columns

        blocks = []
        raw_columns: dict[int, list[int]] = {}
        cursor = 0
        for feat in range(self.num_features):
            if feat in self.categories:
                encoder = OneHotEncoder(
                    categories=[self.categories[feat]],
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                block = encoder.fit_transform(raw[:, feat].reshape(-1, 1))
                self._encoders[feat] = encoder
            else:
                block = raw[:, feat].reshape(-1, 1).astype(float)
            blocks.append(block)
            raw_columns[feat] = list(range(cursor, cursor + block.shape[1]))
            cursor += block.shape[1]
        return np.concatenate(blocks, axis=1), raw_columns

    def _columns(self, feature_subset: FeatureSubset) -> list[int]:
        """Raw columns of `self.data` backing every feature in `feature_subset`."""
        columns = []
        for feature in feature_subset:
            columns.extend(self._raw_columns[feature])
        return columns

    def get_subset(self, *features: int | str) -> FeatureSubset:
        """
        Build a `FeatureSubset` from feature names and/or indices.

        Parameters
        ----------
        *features : int or str
            Feature indices, feature names, or a mix of both.

        Returns
        -------
        FeatureSubset
        """
        return FeatureSubset(
            *(
                self.feature_names.index(feature)
                if isinstance(feature, str)
                else feature
                for feature in features
            )
        )

    def get_categories(
        self, feature_subset: FeatureSubset | None = None
    ) -> dict[int, list[int]]:
        """
        Get the categories of categorical features, optionally restricted to
        a feature subset.

        Parameters
        ----------
        feature_subset : FeatureSubset, optional
            If given, only categories of features in this subset are kept.

        Returns
        -------
        dict of int to list of int
            Maps each categorical feature's index to its sorted list of
            possible values.
        """
        if feature_subset is None:
            return self.categories
        return {
            key: value
            for key, value in self.categories.items()
            if key in feature_subset
        }

    def extract_columns(
        self, feature_subset: FeatureSubset, data: npt.NDArray | None = None
    ) -> npt.NDArray:
        """
        Extract the columns that correspond to `feature_subset`.

        Parameters
        ----------
        feature_subset : FeatureSubset
            Features to extract.
        data : numpy.ndarray of shape (-1, self.data.shape[1]), optional
            Data laid out like `self.data` to extract columns from (e.g. a
            full-width array produced by `expand_columns`). Defaults to
            `self.data`.

        Returns
        -------
        numpy.ndarray of shape (-1, n)
            The extracted columns, where `n` is the number of raw columns
            backing `feature_subset` (equal to `len(feature_subset)` unless
            one-hot encoding expands some of them).
        """
        if data is None:
            data = self.data
        return data[:, self._columns(feature_subset)]

    def expand_columns(
        self,
        feature_subset: FeatureSubset,
        data: npt.NDArray,
        num_cols: int | None = None,
    ) -> npt.NDArray:
        """
        Expand `data`, which holds only `feature_subset`'s columns, back out
        to `num_cols` columns by zero-padding every other feature.

        Parameters
        ----------
        feature_subset : FeatureSubset
            Features that `data`'s columns correspond to.
        data : numpy.ndarray of shape (-1, n)
            Data restricted to `feature_subset`'s columns, as returned by
            `extract_columns`.
        num_cols : int, optional
            Total number of columns in the expanded output; must be at least
            the number of raw columns backing `feature_subset`. Defaults to
            `self.data.shape[1]`.

        Returns
        -------
        numpy.ndarray of shape (-1, num_cols)
            Zero-padded data.
        """
        columns = self._columns(feature_subset)
        if num_cols is None:
            num_cols = self.data.shape[1]
        if num_cols < len(columns):
            raise ValueError(f"num_cols must be at least {len(columns)}")
        result = np.zeros(shape=(data.shape[0], num_cols))
        result[:, columns] = data
        return result

    def _encode_value(self, feature: int, value: float) -> npt.NDArray:
        """
        Encode a single raw feature value (a category value for a
        categorical feature, or the value itself for a continuous one) into
        the raw column(s) backing `feature` in `self.data`.
        """
        columns = self._raw_columns[feature]
        if len(columns) == 1:
            return np.array([value])
        return self._encoders[feature].transform([[value]])[0]

    def overwrite(self, feature_subset: FeatureSubset, values: npt.NDArray) -> npt.NDArray:
        """
        Return a copy of `self.data` with `feature_subset`'s columns
        overwritten by `values`. Used to compute the partial dependence at a
        fixed point in `feature_subset`'s coordinates, by overwriting those
        coordinates across the rest of the dataset.

        `values` are given in raw, pre-encoding form: one entry per feature
        in `feature_subset`, with categorical features given as their
        category value rather than a one-hot vector. One-hot encoding (if
        enabled) is handled internally, so callers never need to know how
        many raw columns back a categorical feature.

        Parameters
        ----------
        feature_subset : FeatureSubset
            Features to overwrite.
        values : numpy.ndarray of shape (len(feature_subset),)
            Raw value to overwrite each feature with, one per feature in
            `feature_subset` (in sorted feature order).

        Returns
        -------
        numpy.ndarray of shape (-1, self.data.shape[1])
            `self.data` with the given columns overwritten.
        """
        result = np.copy(self.data)
        for feature, value in zip(feature_subset, values):
            result[:, self._raw_columns[feature]] = self._encode_value(feature, value)
        return result
