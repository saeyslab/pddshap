from typing import Collection, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt

from .feature_subset import FeatureSubset


class DataSignature:
    def __init__(
        self,
        dataframe: pd.DataFrame | npt.NDArray,
        categorical_features: Collection[int] | None = None,
    ):
        """
        The DataSignature class keeps track of feature names, which features are numerical or categorical,
        and for categorical features, the possible values they can take.

        If the passed dataframe is a numpy array, an extra collection of integers is required to specify
        which columns should be treated as categorical.
        If the dataframe is a Pandas DataFrame, the feature names are taken from the column names and the categorical features are inferred
        from the data types of the columns.

        :param dataframe: the data to extract the signature from
        :param categorical_features: a collection of integers indicating the categorical variables
            (ignored if dataframe is a pandas DataFrame).
        """
        self.feature_names: Tuple[str, ...]
        self.num_features: int
        self.categories: Dict[int, List[int]] = {}

        if isinstance(dataframe, pd.DataFrame):
            self.feature_names = tuple(dataframe.columns)
            if any(dt not in ["int8", "int64", "float32", "float64"] for dt in dataframe.dtypes):
                raise ValueError(
                    "Encode categorical values as int8 and numerical as float32 or float64"
                )
            for i, feat_name in enumerate(self.feature_names):
                if dataframe.dtypes[feat_name] in ["int8", "int64"]:
                    self.categories[i] = list(range(dataframe[feat_name].max() + 1))
        else:
            assert categorical_features is not None, "categorical_features must be provided for numpy arrays"
            self.feature_names = tuple(str(i) for i in range(dataframe.shape[1]))
            for cat_feat in categorical_features:
                self.categories[cat_feat] = list(np.unique(dataframe[:, cat_feat]))
        self.num_features = len(self.feature_names)

    def get_categories(
        self, feature_subset: FeatureSubset | None = None
    ) -> Dict[int, List[int]]:
        if feature_subset is None:
            return self.categories
        return {
            key: value
            for key, value in self.categories.items()
            if key in feature_subset
        }
