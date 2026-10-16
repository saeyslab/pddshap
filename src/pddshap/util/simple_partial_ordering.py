import numpy as np

from ..signature import Dataset
from ..signature.feature_subset import FeatureSubset


def random_partial_ordering(
    num_elements: int, rng: np.random.Generator
) -> list[list[int]]:
    """
    Create a random partial ordering as a list of variable-length feature groups.

    Parameters
    ----------
    num_elements : int
        Number of features to order.
    rng : numpy.random.Generator
        Random number generator used to shuffle features and group sizes.

    Returns
    -------
    list of list of int
        Feature groups ranked by precedence.
    """
    shuffled = list(rng.permutation(range(num_elements)))
    groups: list[list[int]] = []
    current: list[int] = []
    for feature in shuffled:
        current.append(int(feature))
        # End the current rank with moderate probability to get variable group sizes.
        if len(current) > 1 and rng.random() < 0.35:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    # Avoid degenerate single-group ordering for multi-feature datasets.
    if len(groups) == 1 and len(shuffled) > 1:
        split = int(rng.integers(1, len(shuffled)))
        groups = [
            [int(f) for f in shuffled[:split]],
            [int(f) for f in shuffled[split:]],
        ]

    return groups


class SimplePartialOrdering:
    def __init__(
        self, structure: list[list[int]], dataset: Dataset
    ) -> None:
        self.dataset = dataset
        # Maps every feature to its corresponding group in the partial ordering
        self.feature_to_group: dict[int, list[int]] = {}
        # Maps every feature to its rank in the partial ordering
        self.ranks = {}
        for rank, group in enumerate(structure):
            for feature in group:
                if not isinstance(feature, int):
                    raise TypeError(
                        f"Invalid feature type: {feature} ({type(feature)})"
                    )
                self.ranks[feature] = rank
                self.feature_to_group[feature] = group

        # Contains all features that are generally incomparable
        self.incomparable_features = [
            feature
            for feature in range(dataset.num_features)
            if feature not in self.feature_to_group
        ]

    def get_incomparables(self, feature: int) -> list[int]:
        return self.feature_to_group.get(feature, []) + self.incomparable_features

    def contains_successor(self, feature: int, feature_subset: FeatureSubset) -> bool:
        if feature in self.incomparable_features:
            return False
        return any(self.ranks.get(f, -1) > self.ranks[feature] for f in feature_subset)
