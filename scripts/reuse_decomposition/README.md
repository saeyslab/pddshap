# Re-using the decomposition for multiple explanations
The goal of this experiment is to show how a single partial dependence decomposition can be used to generate multiple explanations. If a set of explanation methods only differs in their aggregation coefficients, i.e. they have identical explanation targets (e.g. the model output) and removal methods (e.g. randomization), then their corresponding decompositions are identical. This means that training the decomposition, which is the computationally most expensive step, only needs to happen once. Then, the same decomposition can be used to generate explanations using all of the methods, greatly reducing computational cost.

The script now also:
- Samples a configurable number of explanation points from the test set (`pdd.num_explanations` in `config.yaml`).
- Computes and saves PDD-SHAP values without a partial ordering (`pdd_shap.h5`).
- Creates a random partial ordering over all features and saves it as JSON (`partial_ordering.json`).
- Computes and saves PDD-SHAP values with that partial ordering (`pdd_shap_partial_order.h5`).