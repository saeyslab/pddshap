class FeatureSubset:
    """
    An immutable, hashable set of feature indices. Iteration and `features`
    always yield indices in sorted order - `Dataset` and
    `PartialDependenceDecomposition` rely on this to keep column/value
    arrays aligned across calls.
    """

    def __init__(self, *features: int):
        self._features = frozenset(features) if features is not None else frozenset()

    def intersection(self, other: "FeatureSubset") -> "FeatureSubset":
        return FeatureSubset(*self._features.intersection(other._features))

    def __contains__(self, item):
        return item in self._features

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(self.features)

    def __repr__(self):
        return "FeatureSubset(" + ", ".join(map(str, self.features)) + ")"

    def __hash__(self):
        return hash(self._features)

    def __eq__(self, other):
        return type(other) is FeatureSubset and other._features == self._features

    @property
    def features(self):
        return sorted(tuple(self._features))
