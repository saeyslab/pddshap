import numpy as np
from typing import Dict, Optional
from numpy import typing as npt

from pddshap.signature import FeatureSubset


class COETracker:
    """
    Keeps track of CoE values for multiple outputs and which outputs are active
    TODO Might still be able to use heaps here: re-heapify after an output has 
    been set to inactive. Only useful if linear search operations in this class 
    turn out to be a bottleneck, which is very unlikely
    """
    def __init__(self, num_columns, num_outputs):
        self.num_columns = num_columns
        self.num_outputs = num_outputs
        # Maps feature subsets to CoE values for each output
        self.subset_coe: Dict[FeatureSubset, npt.NDArray] = {}
        # True if corresponding output is active, False otherwise
        self.active_outputs = np.array([True for _ in range(num_outputs)])

    def push(self, subset: FeatureSubset, coe_values: npt.NDArray):
        """
        Add a feature subset to the data structure
        """
        self.subset_coe[subset] = coe_values

    def pop(self) -> FeatureSubset:
        """
        Retrieve and remove the feature subset with the largest CoE value among
        the active outputs
        """
        max_subset: Optional[FeatureSubset] = None
        max_coe = None
        for subset in self.subset_coe.keys():
            coe = np.max(self.subset_coe[subset][self.active_outputs])
            if max_subset is None or coe > max_coe:
                max_subset = subset
                max_coe = coe
        if max_subset is not None:
            del self.subset_coe[max_subset]
            return max_subset
        else:
            raise ValueError("No more feature subsets are left")

    def empty(self) -> bool:
        return len(self.subset_coe.keys()) == 0
