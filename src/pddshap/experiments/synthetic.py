import itertools
from typing import Dict, Tuple, List, Union, Callable

import numpy as np
from numpy import typing as npt
from scipy import special

from ..util import Model


class MultilinearPolynomial(Model):
    def __init__(self, num_features: int, coefficients: Dict[Tuple[int, ...], float]):
        """
        Initialize the multilinear polynomial with a given number of features
        and coefficients.

        Parameters
        ----------
        num_features : int
            Total number of input features.
        coefficients : dict of tuple of int to float
            Maps a subset of features (as a tuple of feature indices) to the
            coefficient of the corresponding monomial.
        """
        self.num_features = num_features
        self.coefficients = coefficients

    def shapley_values(self, data: npt.NDArray):
        """
        Compute closed-form Shapley values for inputs given by `data`.

        This assumes that the input features are centered around 0 and
        independently distributed.

        Parameters
        ----------
        data : numpy.ndarray of shape (num_rows, self.num_features)
            Input data.

        Returns
        -------
        numpy.ndarray of shape (num_rows, self.num_features)
            Shapley values for each row and feature.
        """
        result = np.zeros_like(data)
        for term in self.coefficients:
            if len(term) > 0:
                result[:, tuple(term)] += (self.coefficients[term] * np.prod(data[:, tuple(term)], axis=1) / len(term)).reshape(-1, 1)
        return result

    def variance(self):
        """
        Compute the theoretical variance of the polynomial, under the
        assumption that all input features are zero-centered with variance 1.

        Returns
        -------
        float
            The theoretical variance.
        """
        result = 0.
        for term in self.coefficients:
            if len(term) > 0:
                result += self.coefficients[term]**2
        return result

    def mean(self):
        """
        Compute the theoretical mean of the polynomial, under the assumption
        that all input features are zero-centered. This is just the bias.

        Returns
        -------
        float
            The theoretical mean.
        """
        if () in self.coefficients:
            return self.coefficients[()]
        return 0.

    def __call__(self, data: npt.NDArray):
        """
        Compute the output of the multilinear polynomial for the given input data.

        Parameters
        ----------
        data : numpy.ndarray of shape (num_rows, self.num_features)
            Input data.

        Returns
        -------
        numpy.ndarray of shape (num_rows,)
            Polynomial output for each row.
        """
        assert(data.shape[1] == self.num_features)
        output = np.zeros(shape=(data.shape[0]))
        for term in self.coefficients:
            if len(term) == 0:
                output += self.coefficients[term]  # bias term
            else:
                output += self.coefficients[term] * np.prod(data[:, tuple(term)], axis=1)
        return output

    def __repr__(self):
        s = ""
        for term in self.coefficients:
            coef = self.coefficients[term]
            if len(s) > 0:
                s += "+ " if coef > 0 else "- "
                s += f"{coef if coef > 0 else -coef:.3f}"
            else:
                s += f"{coef:.3f}"
            for feat in term:
                s += f" * x{feat}"
            s += "\n"
        return s


class RandomMultilinearPolynomial(MultilinearPolynomial):
    def __init__(self, num_features: int, num_terms: List[Union[int, float]],
                 coefficient_generator: Callable[[], float] | None = None):
        """
        Parameters
        ----------
        num_features : int
            Total number of input features.
        num_terms : list of int or float
            Number/fraction of terms of each order. `num_terms[k]` is the
            number of desired order-`k` terms. If `num_terms[k]` is -1, all
            order-`k` terms are included. If `len(num_terms) < num_features`,
            all entries between `len(num_terms)` and `num_features` are
            assumed to be 0 (e.g. `len(num_terms) == 3` means only a bias,
            univariate, and bivariate terms are generated). If `num_terms[k]`
            is a float between 0 and 1, it is interpreted as the fraction of
            all possible order-`k` interactions.
        coefficient_generator : callable, optional
            Zero-argument callable returning a random coefficient. Defaults
            to `numpy.random.normal`.
        """
        coefficient_generator = coefficient_generator if coefficient_generator is not None else np.random.normal
        coefficients: Dict[Tuple, float] = {}
        for k in range(len(num_terms)):
            total_num_interactions = special.comb(num_features, k)
            if type(num_terms[k]) is int:
                num_interactions = num_terms[k]
            else:
                num_interactions = int(num_terms[k] * total_num_interactions)
            if num_interactions > total_num_interactions:
                raise ValueError(f"Cannot have more than {total_num_interactions} terms of order {k}")
            elif num_interactions == -1:
                # Add all interactions of order k
                for comb in itertools.combinations(list(range(num_features)), k):
                    coefficients[comb] = coefficient_generator()
            else:
                # Add the required amount of interactions of order k
                # If the number of required terms is more than 90% of the total number of terms,
                # we do this by generating all of them and sampling without replacement.
                # Otherwise, we sample randomly, throwing out duplicates
                # (avoids generating ${n \choose order_terms[k]}$ subsets).
                if num_interactions / total_num_interactions > 0.9:
                    all_subsets = list(itertools.combinations(list(range(num_features)), k))
                    indices = np.random.choice(len(all_subsets), size=num_interactions, replace=False)
                    for i in indices:
                        subset = all_subsets[i]
                        coefficients[subset] = coefficient_generator()
                else:
                    count = 0
                    while count < num_interactions:
                        subset = tuple(np.sort(np.random.choice(num_features, size=k, replace=False)))
                        if subset not in coefficients.keys():
                            coefficients[subset] = coefficient_generator()
                            count += 1
        super().__init__(num_features, coefficients)
