"""Persistencia atómica, carga y metadatos de versiones de modelos."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.schema import (
    CATEGORICAL_INPUT,
    INPUT_COLUMNS,
    NUMERIC_INPUT,
    ModelNotFoundError,
    InputValidationError,
    VersionExistsError,
)


def resolve_version_dir(output_dir: str | Path, version: str) -> Path:
    """Valida y resuelve una ruta de versión sin crearla."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise InputValidationError(
            f"Nombre de versión inválido: '{version}'."
        )
    version_dir = Path(output_dir) / version
    if version_dir.exists():
        raise VersionExistsError(f"La versión ya existe: '{version_dir}'.")
    return version_dir


def save_version(
    version_dir: str | Path,
    pipeline: Any,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Guarda una versión completa mediante un directorio temporal atómico."""
    destination = Path(version_dir)
    temporary = Path(f"{destination}.tmp")
    if temporary.exists():
        _remove_path(temporary)
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        joblib.dump(pipeline, temporary / "modelo.joblib", compress=3)
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if destination.exists():
            raise VersionExistsError(f"La versión ya existe: '{destination}'.")
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            _remove_path(temporary)
        raise


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Carga un pipeline desde una carpeta de versión o desde su archivo."""
    requested = Path(path)
    model_path = requested / "modelo.joblib" if requested.is_dir() else requested
    if not model_path.exists() or not model_path.is_file():
        raise ModelNotFoundError(f"No existe el modelo: '{model_path}'.")
    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ModelNotFoundError(
            f"No se pudo cargar el modelo '{model_path}': {error}"
        ) from error
    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelNotFoundError(
                f"No se pudo cargar la metadata '{metadata_path}': {error}"
            ) from error
    else:
        metadata = {}
    return pipeline, metadata


def build_metadata(
    *,
    version: str,
    target: str,
    params: dict[str, Any],
    seed: int,
    data_path: str | Path,
    n_rows: int,
    n_train: int,
    n_test: int,
    test_size: float,
    notes: str,
) -> dict[str, Any]:
    """Construye los metadatos reproducibles de una versión entrenada."""
    source = Path(data_path)
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_type": "regresion",
        "target": target,
        "algorithm": "RandomForestRegressor",
        "params": params,
        "seed": seed,
        "input_schema": {
            "columns": INPUT_COLUMNS,
            "numeric": NUMERIC_INPUT,
            "categorical": CATEGORICAL_INPUT,
            "dtypes": {column: "object" for column in CATEGORICAL_INPUT}
            | {column: "float64" for column in NUMERIC_INPUT},
        },
        "data": {
            "fuente": "CSV",
            "archivo": str(source),
            "sha256": _sha256_file(source),
            "n_filas": n_rows,
        },
        "split": {
            "estrategia": "train_test_split",
            "test_size": test_size,
            "n_train": n_train,
            "n_test": n_test,
        },
        "metric": "RMSE (USD)",
        "python_version": platform.python_version(),
        "requirements_file": "requirements.txt",
        "package_versions": _package_versions(),
        "git_commit": get_git_commit(),
        "notes": notes,
    }


def get_git_commit() -> str:
    """Obtiene el commit desde entorno o Git, con un valor explícito de reserva."""
    commit = os.environ.get("GIT_COMMIT")
    if commit:
        return commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "desconocido"
    except (OSError, subprocess.SubprocessError):
        return "desconocido"


def _remove_path(path: Path) -> None:
    """Elimina un temporal que puede ser archivo o directorio."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques para no cargar el CSV completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    """Obtiene versiones de las dependencias principales disponibles."""
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "desconocido"
    return versions
