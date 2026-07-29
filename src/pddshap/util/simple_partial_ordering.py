from ..signature import DataSignature
from ..signature.feature_subset import FeatureSubset


class SimplePartialOrdering:
    def __init__(
        self, structure: list[list[int]], data_signature: DataSignature
    ) -> None:
        self.data_signature = data_signature
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
            for feature in range(data_signature.num_features)
            if feature not in self.feature_to_group
        ]

    def get_incomparables(self, feature: int) -> list[int]:
        return self.feature_to_group.get(feature, []) + self.incomparable_features

    def contains_successor(self, feature: int, feature_subset: FeatureSubset) -> bool:
        if feature in self.incomparable_features:
            return False
        return any(self.ranks.get(f, -1) > self.ranks[feature] for f in feature_subset)
