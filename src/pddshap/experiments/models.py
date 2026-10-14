from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

MODEL_REGISTRY: dict[str, dict[str, type]] = {
    "classification": {
        "knn": KNeighborsClassifier,
        "gradientboosting": GradientBoostingClassifier,
    },
    "regression": {
        "knn": KNeighborsRegressor,
        "gradientboosting": GradientBoostingRegressor,
    },
}


def model_names(pred_type: str) -> list[str]:
    return list(MODEL_REGISTRY[pred_type])


def make_model(pred_type: str, name: str, **kwargs):
    return MODEL_REGISTRY[pred_type][name](**kwargs)


def predict_fn(model, pred_type: str) -> Callable:
    """The right prediction method to explain for a given problem type."""
    return model.predict_proba if pred_type == "classification" else model.predict
