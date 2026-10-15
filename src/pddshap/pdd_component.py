from typing import Callable, Dict, Optional, Union
from .estimator import KNNPDDEstimator, PDDEstimator, ConstantPDDEstimator, \
        TreePDDEstimator, GradientBoostingPDDEstimator, GaussianProcessPDDEstimator
from .sampling import CollocationMethod, ConditioningMethod
from .signature import FeatureSubset, DataSignature
import numpy as np
from numpy import typing as npt
import pandas as pd


class PDDComponent:
    def __init__(self, feature_subset: FeatureSubset, data_signature: DataSignature,
                 collocation_method: CollocationMethod,
                 conditioning_method: ConditioningMethod,
                 estimator_type: str,
                 est_kwargs: Optional[Dict]) -> None:
        self.data_signature = data_signature
        self.feature_subset = feature_subset
        self.estimator: Optional[PDDEstimator] = None
        self.fitted = False
        est_constructors = {
            "tree": TreePDDEstimator,
            "forest": GradientBoostingPDDEstimator,
            "knn": KNNPDDEstimator,
            "gp": GaussianProcessPDDEstimator
        }
        self.est_constructor = est_constructors[estimator_type]
        self.collocation_method = collocation_method
        self.conditioning_method = conditioning_method
        self.est_kwargs = est_kwargs
        self.num_outputs = None


    def fit(self, data: npt.NDArray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[FeatureSubset, Union["PDDComponent", "ConstantPDDComponent"]]):
        """
        Fit this component against the residual partial dependence left
        after subtracting every already-fitted strict-subset component.

        Parameters
        ----------
        data : numpy.ndarray of shape (n, num_features)
            Background data, in full (not subset-restricted) column space.
        model : callable
            The model of which we want to compute the partial dependence.
        subcomponents : dict of FeatureSubset to PDDComponent or ConstantPDDComponent
            Every already-fitted component whose feature subset is a strict
            subset of `self.feature_subset`.
        """

        # Collocation points are the points where we compute partial dependence
        collocation_points = self.collocation_method.get_collocation_points(
            self.feature_subset.extract_columns(data)
        )

        # For each collocation point, get partial dependence at this point
        # Shape: (num_rows, num_outputs)
        partial_dependence = np.average(
            np.array([
                self.conditioning_method.conditional_expectation(self.feature_subset, collocation_point, model)
                for collocation_point in collocation_points
            ]
        ), axis=1)

        # Subtract strict subcomponents outputs from partial dependence
        # The result is the estimated output for the functional component
        self.num_outputs = partial_dependence.shape[1]
        for subset, subcomponent in subcomponents.items():
            # Subtract subcomponent from partial dependence
            full_width_collocation_points = self.feature_subset.expand_columns(collocation_points, data.shape[1])
            partial_dependence -= subcomponent(full_width_collocation_points)

        # Fit a model on resulting values
        categories = self.data_signature.get_categories(self.feature_subset)
        if np.max(partial_dependence) - np.min(partial_dependence) < 1e-3:
            # If the partial dependence is constant, don't fit an estimator
            self.estimator = ConstantPDDEstimator(categories, self.feature_subset, partial_dependence[0])
        else:
            # Otherwise, fit a model
            self.estimator = self.est_constructor(categories, self.feature_subset, **self.est_kwargs)
            self.estimator.fit(collocation_points, partial_dependence)
        self.fitted = True

    def __call__(self, data: npt.NDArray) -> npt.NDArray:
        """
        Compute this component's output on the given data.

        Note: all columns must be passed in `data`. 
        The method extracts the relevant columns itself.

        Parameters
        ----------
        data : numpy.ndarray of shape (num_rows, len(self.data_signature.feature_names))
            Full-width input data.

        Returns
        -------
        numpy.ndarray of shape (num_rows, self.num_outputs)
            This component's output at each row.
        """
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(self.feature_subset.extract_columns(data))

# TODO this should be a subclass of PDDComponent, might use this for other
# components than just the empty set
class ConstantPDDComponent:
    def __init__(self, feature_subset: FeatureSubset,
                 data_signature: DataSignature):
        self.estimator = None
        self.num_outputs = None
        self.feature_subset = feature_subset
        self.data_signature = data_signature

    def fit(self, data: npt.NDArray, model: Callable[[npt.NDArray], npt.NDArray]):
        output = model(data)
        self.num_outputs = 1 if len(output.shape) == 1 else output.shape[1]
        avg_output = np.average(output, axis=0)
        self.estimator = ConstantPDDEstimator(
            self.data_signature.get_categories(self.feature_subset),
            self.feature_subset, avg_output)
        return avg_output

    def __call__(self, data: npt.NDArray):
        if self.estimator is not None:
            return self.estimator(data)
        raise ValueError("Evaluating component before fit()")
