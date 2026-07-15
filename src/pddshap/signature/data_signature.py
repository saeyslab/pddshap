from typing import Tuple, Dict, List, Collection
import pandas as pd
from numpy import typing as npt
import numpy as np
from .feature_subset import FeatureSubset


class DataSignature:
    def __init__(self, dataframe: pd.DataFrame | npt.NDArray, categorical_features: Collection[int] = None):
        """
        Initializes the data signature from a pandas DataFrame or numpy NDArray.
        If the passed data is a numpy array, an extra collection of integers is required to specify
        which columns should be treated as categorical.
        This argument can be ignored if the data is a pandas DataFrame.
        :param dataframe: the data to extract the signature from
        :param categorical_features: a collection of integers indicating the categorical variables
            (ignored if dataframe is a pandas DataFrame).
        """
        self.feature_names: Tuple[str, ...]
        self.num_features: int
        self.categories: Dict[int, List[int]] = {}

        if type(dataframe) == pd.DataFrame:
            self.feature_names = tuple(dataframe.columns)
            illegal_dtypes = [dt not in ["int8", "int64", "float32", "float64"] for col, dt in dataframe.dtypes.items()]
            if any(illegal_dtypes) > 0:
                raise ValueError("Encode categorical values as int8 and numerical as float32 or float64")
            for i, feat_name in enumerate(self.feature_names):
                if dataframe.dtypes[i] in ["int8", "int64"]:
                    self.categories[i] = list(range(dataframe[feat_name].max() + 1))
        else:
            self.feature_names = tuple(str(i) for i in range(dataframe.shape[1]))
            for cat_feat in categorical_features:
                self.categories[cat_feat] = list(np.unique(dataframe[:, cat_feat]))
        self.num_features = len(self.feature_names)

    def get_categories(self, feature_subset: FeatureSubset = None) -> Dict[int, List[int]]:
        if feature_subset is None:
            return self.categories
        return {key: value for key, value in self.categories.items() if key in feature_subset}
