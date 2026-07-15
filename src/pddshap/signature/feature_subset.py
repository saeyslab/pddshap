from typing import Collection
from numpy import typing as npt
import numpy as np


class FeatureSubset:
    def __init__(self, *features: int):
        self._features = frozenset(features) if features is not None else frozenset()

    def project(self, data: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
        """
        Projects the rows in data to the hyperplane defined by the values.
        In practice, this just overwrites the columns corresponding to this feature subset
        using the values given by values.
        Can be used to compute the partial dependence at a given point for this feature subset.
        :param data: Data to be projected. Shape: (-1, total_num_features)
        :param values: Values to project the data to. Shape: (len(self.features),)
        :return: Projected data. Shape: (-1, total_num_features)
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
        Extracts the columns in data that correspond to this feature subset.
        :param data: Data from which to extract columns. Shape: (-1, total_num_features)
        :return: Extracted columns. Shape: (-1, len(self.features))
        """
        if len(data.shape) != 2:
            raise ValueError("Invalid shape: data:", data.shape)
        return data[:, sorted(tuple(self._features))]

    def expand_columns(self, data: npt.NDArray, num_cols: int) -> npt.NDArray:
        """
        Expands the columns in data to the specified number by adding 0 columns for all the features not in this subset
        :param num_cols: must be at least len(self.features)
        :param data: shape: (-1, len(self.features))
        :return: shape: (-1, total_num_features)
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
        return type(other) == FeatureSubset and other.features == self.features

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
