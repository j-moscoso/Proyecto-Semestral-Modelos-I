"""Construcción de modelos y métricas de regresión."""

from math import sqrt

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_baseline(strategy: str) -> DummyRegressor:
    """Construye un regresor ingenuo con la estrategia solicitada."""
    return DummyRegressor(strategy=strategy)


def compute_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Calcula RMSE, MAE y R2 como ``float`` nativos de Python."""
    return {
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
