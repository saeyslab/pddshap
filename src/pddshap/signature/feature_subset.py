from numpy import typing as npt
import numpy as np


class FeatureSubset:
    def __init__(self, *features: int):
        self._features = frozenset(features) if features is not None else frozenset()

    def project(self, data: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
        """
        Project the rows in `data` to the hyperplane defined by `values`.

        In practice, this just overwrites the columns corresponding to this
        feature subset using the given values. Can be used to compute the
        partial dependence at a given point for this feature subset.

        Parameters
        ----------
        data : numpy.ndarray of shape (-1, total_num_features)
            Data to be projected.
        values : numpy.ndarray of shape (len(self.features),)
            Values to project the data to.

        Returns
        -------
        numpy.ndarray of shape (-1, total_num_features)
            Projected data.
        """
        # Check shapes
        if len(data.shape) != 2:
            raise ValueError("Invalid shape: data:", data.shape)
        if len(values.shape) != 1:
            raise ValueError("Invalid shape: values:", values.shape)
        data_copy = np.copy(data)
        data_copy[:, sorted(tuple(self._features))] = values
        return data_copy

    def get_columns(self, data: npt.NDArray) -> npt.NDArray:
        """
        Extract the columns in `data` that correspond to this feature subset.

        Parameters
        ----------
        data : numpy.ndarray of shape (-1, total_num_features)
            Data from which to extract columns.

        Returns
        -------
        numpy.ndarray of shape (-1, len(self.features))
            Extracted columns.
        """
        if len(data.shape) != 2:
            raise ValueError("Invalid shape: data:", data.shape)
        return data[:, sorted(tuple(self._features))]

    def expand_columns(self, data: npt.NDArray, num_cols: int) -> npt.NDArray:
        """
        Expand the columns in `data` to `num_cols` by zero-padding all the
        features not in this subset.

        Parameters
        ----------
        data : numpy.ndarray of shape (-1, len(self.features))
            Data restricted to this feature subset's columns.
        num_cols : int
            Total number of columns in the expanded output; must be at least
            `len(self.features)`.

        Returns
        -------
        numpy.ndarray of shape (-1, num_cols)
            Zero-padded data.
        """
        if num_cols < len(self.features):
            raise ValueError(f"num_cols must be at least {len(self.features)}")
        result = np.zeros(shape=(data.shape[0], num_cols))
        result[:, self.features] = data
        return result

    def union(self, other: "FeatureSubset") -> "FeatureSubset":
        return FeatureSubset(*self._features.union(other._features))

    def intersection(self, other: "FeatureSubset") -> "FeatureSubset":
        return FeatureSubset(*self._features.intersection(other._features))

    def __contains__(self, item):
        return item in self._features

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(sorted(tuple(self._features)))

    def __repr__(self):
        return "FeatureSubset(" + ", ".join(map(str, self.features)) + ")"

    def __hash__(self):
        return hash(self._features)

    def __eq__(self, other):
        return type(other) is FeatureSubset and other.features == self.features

    def __ne__(self, other):
        return self != other

    def __lt__(self, other):
        return self.features < other.features

    def __gt__(self, other):
        return self.features > other.features

    def __le__(self, other):
        return self.features <= other.features

    def __ge__(self, other):
        return self.features >= other.features

    @property
    def features(self):
        return sorted(tuple(self._features))
