# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PDD-SHAP: a Python package that approximates Shapley-value explanations by decomposing a
model's prediction function into an ANOVA-style **partial dependence decomposition** (a sum of
component functions, one per feature subset), then deriving Shapley values from those
components directly instead of resampling the model at explanation time. The expensive part
(fitting the decomposition against a background dataset) is done once; cheap aggregation
afterwards can produce Shapley values under different removal/conditioning schemes or partial
orderings (see `experiments/reuse_decomposition/`) without refitting.

## Commands

This project uses `uv` (package manager + build backend, see `uv.lock`), Python >= 3.13.

```bash
# install (editable, including the dev group: h5py, matplotlib, shap, seaborn, numba, ipython, pyyaml)
uv sync
# or: pip install -e .[dev]
```

There is no test runner/CI configured (no pytest config, no `.github/workflows`). Files under
`test/` are standalone scripts with a `if __name__ == "__main__":` block that print
correlation/R² comparisons against ground truth rather than asserting; run them directly:

```bash
uv run python test/test_partial_dependence_decomposition.py
```

Ground truth in tests comes from `test/multilinear_polynomial.py`, which implements an exact
multilinear polynomial model with a closed-form Shapley value solution — this is the only
model type simple enough to have known-correct component functions and Shapley values, so it's
the thing PDD-SHAP's output is checked against.

### Experiments

Experiments live under `experiments/` and read YAML config (`experiments/config/config.yaml`
for shared data/output dirs, `experiments/config/datasets.yaml` for per-dataset OpenML IDs and
problem type, plus a per-experiment `config.yaml`). Datasets are pre-downloaded CSVs under
`data/<dataset>/`. Run an experiment script directly, e.g.:

```bash
uv run python experiments/reuse_decomposition/run_experiment.py
```

See `experiments/README.md` and `experiments/reuse_decomposition/README.md` for the
prerequisites → experiment → evaluate workflow and what each experiment demonstrates.

## Architecture

### The decomposition pipeline

`PartialDependenceDecomposition` (`src/pddshap/partial_dependence_decomposition.py`) is the
top-level object. Its `fit()`:

1. Builds a `DataSignature` from the training data (feature names, and for integer-typed
   columns, their category sets — used for one-hot encoding categorical features downstream).
2. Fits a `ConstantPDDComponent` for the empty feature subset — the model's average output over
   the background data.
3. Enumerates feature subsets up to `max_size` by cardinality (or accepts an explicit
   `feature_sets` list) and fits a `PDDComponent` per subset, **in increasing order of subset
   size**, in parallel within each size tier (`joblib.Parallel`). Each component's fit needs
   every strict-subset component already fitted, because it models the *residual* partial
   dependence after subtracting all lower-order components (the ANOVA decomposition property).

`evaluate()` / `__call__` reconstruct the model's output as the sum over all fitted components.
`shapley_values()` redistributes each component's output equally among its member features
(or, when a `partial_ordering` is given, using asymmetric Shapley value coefficients computed
by `_asv_coefficients`, based on `SimplePartialOrdering` — see
Aas/Jullum/Løland on asymmetric SV with dependent features). `project=True` orthogonally
projects the result onto the hyperplane satisfying the efficiency axiom (values sum to the
actual prediction difference from the background average).

### `PDDComponent` (`src/pddshap/pdd_component.py`)

Represents one feature-subset term in the decomposition. Fitting a component:
- picks **collocation points** (`CollocationMethod`, `src/pddshap/sampling/collocation_method.py`)
  — the coordinates in the component's own feature subset at which the partial dependence is
  evaluated (full background data, a random subsample, or k-means centroids),
- computes partial dependence at each point via a **conditioning method**
  (`ConditioningMethod`, `src/pddshap/sampling/conditioning_method.py`) — this is what encodes
  the assumption about dependence between features (independent/marginal "off-manifold" SHAP,
  Gaussian conditional sampling, kernel-weighted conditioning by Mahalanobis distance, or a
  threshold-based hybrid of the two),
- subtracts out every already-fitted strict-subset component's contribution at those points
  (`FeatureSubset.expand_columns` pads a subset-local point back to full dimensionality with
  zeros so sub-components can be evaluated),
- and fits a regression estimator (`PDDEstimator` subclasses in `src/pddshap/estimator/`:
  `tree`, `forest` (gradient boosting), `knn`, `gp` (Gaussian process)) on
  (collocation point → residual partial dependence), unless the residual is ~constant, in
  which case it falls back to a `ConstantPDDEstimator`. The estimator used is selected by the
  `estimator_type` string passed into `PartialDependenceDecomposition.__init__`.

`ConstantPDDComponent` is the always-present empty-subset term; a `# TODO` in the source notes
it should really be a `PDDComponent` subclass.

### `FeatureSubset` (`src/pddshap/signature/feature_subset.py`)

An immutable, hashable, sorted set of feature-column indices. This is the key/identity used
throughout (`dict[FeatureSubset, PDDComponent]` in the decomposition, subset containment checks
for "is this a sub-component", etc.). Core operations: `get_columns` (slice out this subset's
columns), `expand_columns` (inverse — zero-pad back to full width), `project` (overwrite a
subset's columns in a full-width array with fixed values — used to compute partial dependence
at a fixed point in that subset's coordinates).

### Categorical features

`DataSignature` requires pandas input to use `int8`/`int64` for categorical columns and
`float32`/`float64` for numerical ones (anything else raises); categorical columns get their
category range inferred automatically; numpy-array input requires passing
`categorical_features` indices explicitly. `KNNPDDEstimator` one-hot encodes categorical
columns in a subset before fitting/predicting; other estimators (tree/forest/GP) take the raw
integer-encoded columns as-is.

### Asymmetric Shapley values / partial orderings

`SimplePartialOrdering` (`src/pddshap/util/simple_partial_ordering.py`) encodes a user-supplied
partial order over features as groups ranked by precedence (features not mentioned are treated
as mutually incomparable with everything). `PartialDependenceDecomposition.shapley_values()`
uses it to zero out contributions from features that have a strict successor in their own
feature subset, and to reweight remaining contributions by the count of "incomparable" features
in that subset — this is what lets one fitted decomposition serve both symmetric Shapley values
and asymmetric ones under a causal/precedence ordering, without refitting (the point of the
`reuse_decomposition` experiment).
