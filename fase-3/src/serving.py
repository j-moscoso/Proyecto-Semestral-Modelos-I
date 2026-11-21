"""Carga de modelos, predicción y entrenamiento para la API."""

import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import (
    AppError,
    INPUT_COLUMNS,
    NUMERIC_INPUT,
    ModelNotFoundError,
)
from src.validation import validate_and_prepare_input
from src.versioning import load_model


class TrainingInProgressError(AppError):
    """Indica que ya existe un entrenamiento en ejecución."""


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[Path, tuple[Any, dict[str, Any], str]] = {}
_TRAINING_LOCK = threading.Lock()


def get_models_dir() -> Path:
    """Obtiene el directorio de modelos configurado en el entorno."""
    return Path(os.environ.get("MODELS_DIR", "models"))


def get_active_version() -> str:
    """Obtiene la versión activa configurada en el entorno."""
    return os.environ.get("MODEL_VERSION", "v1")


def get_train_data_path() -> Path:
    """Obtiene la ruta de datos de entrenamiento configurada."""
    return Path(os.environ.get("TRAIN_DATA", "data/train.csv"))


def list_versions(models_dir: str | Path | None = None) -> list[str]:
    """Lista las versiones que contienen un archivo modelo.joblib."""
    directory = get_models_dir() if models_dir is None else Path(models_dir)
    if not directory.is_dir():
        return []
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir()
        and not entry.name.endswith(".tmp")
        and (entry / "modelo.joblib").is_file()
    )


def load_active_model(
    version: str | None = None,
) -> tuple[Any, dict[str, Any], str]:
    """Carga el modelo activo y reutiliza la instancia almacenada en caché."""
    selected_version = get_active_version() if version is None else version
    if not re.fullmatch(r"[A-Za-z0-9._-]+", selected_version):
        raise ModelNotFoundError(
            f"Versión de modelo no válida: '{selected_version}'. "
            f"Versiones disponibles: {', '.join(list_versions()) or 'ninguna'}."
        )

    directory = get_models_dir()
    model_dir = directory / selected_version

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_dir)
        if cached is not None:
            return cached

    try:
        pipeline, metadata = load_model(model_dir)
    except ModelNotFoundError as error:
        available = list_versions(directory)
        raise ModelNotFoundError(
            f"{error} Versiones disponibles: "
            f"{', '.join(available) if available else 'ninguna'}."
        ) from error

    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        pipeline.named_steps["model"].n_jobs = 1

    result = (pipeline, metadata, selected_version)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.setdefault(model_dir, result)
    return result


def clear_model_cache() -> None:
    """Vacía la caché de pipelines cargados."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def predict_record(record: dict) -> tuple[float, str]:
    """Valida un registro, ejecuta el pipeline y devuelve predicción y versión."""
    pipeline, metadata, selected_version = load_active_model()
    frame = pd.DataFrame([record])
    input_schema = metadata.get("input_schema")
    if isinstance(input_schema, dict):
        prepared, _ = validate_and_prepare_input(
            frame,
            expected_columns=input_schema.get("columns", INPUT_COLUMNS),
            numeric_columns=input_schema.get("numeric", NUMERIC_INPUT),
        )
    else:
        prepared, _ = validate_and_prepare_input(frame)
    prediction = float(pipeline.predict(prepared)[0])
    return prediction, selected_version


def next_version_id(models_dir: str | Path | None = None) -> str:
    """Devuelve el siguiente identificador de versión disponible."""
    directory = get_models_dir() if models_dir is None else Path(models_dir)
    maximum = 0
    if directory.is_dir():
        for entry in directory.iterdir():
            if not entry.is_dir():
                continue
            match = re.fullmatch(r"v([0-9]+)", entry.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return f"v{maximum + 1}"


def train_new_version(
    version: str | None = None,
    n_estimators: int = 40,
    seed: int = 42,
) -> dict:
    """Entrena y guarda una nueva versión sin sobrescribir una existente."""
    if not _TRAINING_LOCK.acquire(blocking=False):
        raise TrainingInProgressError("Ya hay un entrenamiento en curso.")

    try:
        models_dir = get_models_dir()
        selected_version = next_version_id(models_dir) if version is None else version
        import train

        args = argparse.Namespace(
            data=str(get_train_data_path()),
            target="price",
            output_dir=str(models_dir),
            version=selected_version,
            seed=seed,
            test_size=0.2,
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.7,
            cv_folds=0,
        )
        train.run_training(args)

        version_dir = models_dir / selected_version
        metrics_path = version_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = metrics["test"]["random_forest"]
        return {
            "version": selected_version,
            "metricas_test": test_metrics,
            "ruta": str(version_dir),
        }
    finally:
        _TRAINING_LOCK.release()
