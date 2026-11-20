"""Fixtures sintéticas y aisladas para las pruebas de la aplicación."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Genera un conjunto reproducible sin leer datos del proyecto."""
    rng = np.random.default_rng(20261103)
    n_rows = 300
    model_year = rng.integers(2000, 2025, size=n_rows)
    milage = rng.integers(1_000, 220_000, size=n_rows)
    brands = np.array(["Toyota", "Ford", "Honda", "BMW", "Chevrolet"])
    models = np.array(["Sedan Base", "SUV Sport", "Pickup XLT", "Coupe S"])
    fuel_types = np.array(["Gasoline", "Diesel", "Hybrid", "E85 Flex Fuel"])
    engines = np.array(
        [
            "300.0HP 3.0L 6 Cylinder Engine Gasoline Fuel",
            "250.0HP 2.5L 4 Cylinder Engine Gasoline Fuel",
            "400.0HP 5.0L 8 Cylinder Engine Gasoline Fuel",
        ]
    )
    transmissions = np.array(
        ["8-Speed A/T", "6-Speed M/T", "CVT Transmission"]
    )
    colors = np.array(["Black", "White", "Blue", "Silver"])

    data = pd.DataFrame(
        {
            "id": np.arange(10_000, 10_000 + n_rows),
            "brand": rng.choice(brands, size=n_rows),
            "model": rng.choice(models, size=n_rows),
            "model_year": model_year,
            "milage": milage,
            "fuel_type": rng.choice(fuel_types, size=n_rows).astype(object),
            "engine": rng.choice(engines, size=n_rows).astype(object),
            "transmission": rng.choice(transmissions, size=n_rows).astype(object),
            "ext_col": rng.choice(colors, size=n_rows),
            "int_col": rng.choice(colors, size=n_rows),
            "accident": rng.choice(
                ["None reported", "At least 1 accident or damage reported"],
                size=n_rows,
            ).astype(object),
            "clean_title": rng.choice(["Yes", "No"], size=n_rows).astype(object),
        }
    )
    data["price"] = (
        12_000
        + (model_year - 2000) * 1_100
        - milage * 0.075
        + rng.normal(0, 2_500, size=n_rows)
    ).round().clip(min=1_000).astype(int)

    for column in ("fuel_type", "accident", "clean_title", "transmission", "engine"):
        missing_indices = rng.choice(n_rows, size=6, replace=False)
        data.loc[missing_indices, column] = None

    return data


@pytest.fixture
def training_csv(tmp_path, synthetic_data) -> Path:
    """Guarda el conjunto sintético dentro del directorio temporal de pytest."""
    path = tmp_path / "synthetic_cars.csv"
    synthetic_data.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_version(tmp_path, training_csv, monkeypatch):
    """Entrena una versión pequeña y conserva la instancia antes de guardarla."""
    import train
    from src.versioning import save_version as persist_version

    captured = {}

    def capture_and_save(version_dir, pipeline, metrics, metadata):
        captured["pipeline"] = pipeline
        persist_version(version_dir, pipeline, metrics, metadata)

    monkeypatch.setattr(train, "save_version", capture_and_save)
    monkeypatch.setenv("GIT_COMMIT", "pytest-synthetic-data")
    output_dir = tmp_path / "models"
    result = train.main(
        [
            "--data",
            str(training_csv),
            "--version",
            "v1",
            "--output-dir",
            str(output_dir),
            "--n-estimators",
            "5",
            "--test-size",
            "0.2",
        ]
    )

    assert result == 0
    return {
        "output_dir": output_dir,
        "version_dir": output_dir / "v1",
        "pipeline_before_save": captured["pipeline"],
    }
