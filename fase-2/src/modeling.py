"""Construcción de modelos y métricas de regresión."""

from math import sqrt

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.dummy import DummyRegressor

from src.features import (
    build_preprocessor,
    extract_engine_features,
    extract_transmission_features,
    normalize_missing,
)


def build_pipeline(
    seed: int,
    n_estimators: int = 40,
    max_depth: int = 20,
    min_samples_leaf: int = 3,
    max_features: float = 0.7,
) -> Pipeline:
    """Construye el pipeline completo de normalización, extracción y regresión."""
    return Pipeline(
        steps=[
            ("normalize", FunctionTransformer(normalize_missing, validate=False)),
            ("engine_features", FunctionTransformer(extract_engine_features, validate=False)),
            (
                "transmission_features",
                FunctionTransformer(extract_transmission_features, validate=False),
            ),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


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
