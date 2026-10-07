"""Tests that the fitted decomposition reconstructs the underlying model's output."""

from sklearn import metrics


def test_decomposition_reconstructs_model_output(
    quadratic_model_and_data, fitted_decomposition, estimator_config
):
    model, _, X_test = quadratic_model_and_data

    predicted = fitted_decomposition(X_test).ravel()
    true = model(X_test.to_numpy())

    r2 = metrics.r2_score(true, predicted)
    threshold = estimator_config["r2_threshold"]
    assert r2 > threshold, (
        f"Decomposition should recover the full model output, got R2={r2}"
    )
