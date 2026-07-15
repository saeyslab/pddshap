from typing import List, Dict
from ..signature import DataSignature
from ..signature.feature_subset import FeatureSubset


class SimplePartialOrdering:
    def __init__(self, structure: List[List],
                 data_signature: DataSignature) -> None:
        self.data_signature = data_signature
        # Maps every feature to its corresponding group in the partial ordering
        self.incomparables: Dict[int, List[int]] = {}
        # Maps every feature to its rank in the partial ordering
        self.ranks = {}
        for rank, group in enumerate(structure):
            for feature in group:
                self.ranks[feature] = rank
                if isinstance(feature, str):
                    feature = data_signature.feature_names.index(feature)
                    self.incomparables[feature] = [data_signature.feature_names.index(f) for f in group]
                else:
                    self.incomparables[feature] = group

        # Contains all features that are generally incomparable
        self.general_incomparables = [
            feature for feature in range(data_signature.num_features) 
            if feature not in self.incomparables.keys()
        ]
    
    def get_incomparables(self, feature: int) -> List[int]:
        return self.incomparables.get(feature, []) + self.general_incomparables
    
    def contains_successor(self, feature: int, feature_subset: FeatureSubset) -> bool:
        if feature in self.general_incomparables:
            return False
        return any([self.ranks.get(f, -1) > self.ranks[feature] for f in feature_subset])