"""Tests that `fit()` builds the expected set of feature-subset components."""

from pddshap.signature import FeatureSubset

from .conftest import NUM_FEATURES


def test_fit_creates_expected_feature_subset_components(fitted_decomposition):
    subset_sizes = sorted(len(fs) for fs in fitted_decomposition.components)

    assert subset_sizes.count(0) == 1
    assert subset_sizes.count(1) == NUM_FEATURES
    assert subset_sizes.count(2) == NUM_FEATURES * (NUM_FEATURES - 1) // 2
    assert max(subset_sizes) == 2, "fit(max_size=2) must not fit larger subsets"
    assert FeatureSubset() in fitted_decomposition.components
