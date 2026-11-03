"""Pruebas de entrenamiento, versionamiento y predicción de la aplicación."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import predict
import train
from src.schema import INPUT_COLUMNS, InputValidationError
from src.validation import validate_and_prepare_input
from src.versioning import load_model


def test_entrenamiento_y_almacenamiento(trained_version):
    version_dir = trained_version["version_dir"]

    assert (version_dir / "modelo.joblib").is_file()
    assert (version_dir / "metrics.json").is_file()
    assert (version_dir / "metadata.json").is_file()

    pipeline, metadata = load_model(version_dir)
    assert pipeline is not None
    assert metadata["version"] == "v1"

    metrics = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"random_forest", "dummy_mean", "dummy_median"} <= set(metrics["test"])
    for model_metrics in metrics["test"].values():
        assert {"rmse", "mae", "r2"} <= set(model_metrics)


def test_carga_y_prediccion(trained_version, synthetic_data, tmp_path):
    version_dir = trained_version["version_dir"]
    pipeline_before_save = trained_version["pipeline_before_save"]
    loaded_pipeline, _ = load_model(version_dir)

    sample = synthetic_data.iloc[:12].copy()
    expected = pipeline_before_save.predict(sample.loc[:, INPUT_COLUMNS])
    actual = loaded_pipeline.predict(sample.loc[:, INPUT_COLUMNS])
    assert np.allclose(actual, expected)

    input_path = tmp_path / "prediction_input.csv"
    output_path = tmp_path / "predictions.csv"
    sample.to_csv(input_path, index=False)
    assert (
        predict.main(
            [
                "--model",
                str(version_dir),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    result = pd.read_csv(output_path)
    assert result["id"].tolist() == sample["id"].tolist()
    assert np.allclose(result["prediccion"].to_numpy(), expected)


def test_valores_faltantes(trained_version, synthetic_data):
    pipeline, _ = load_model(trained_version["version_dir"])
    sample = synthetic_data.iloc[:3].copy()
    for column in ("engine", "fuel_type", "accident", "clean_title"):
        sample[column] = sample[column].astype(object)
        sample.loc[sample.index[0], column] = None
        sample.loc[sample.index[1], column] = pd.NA
        sample.loc[sample.index[2], column] = ""

    sample["transmission"] = sample["transmission"].astype(object)
    sample.loc[sample.index[0], "transmission"] = None
    sample.loc[sample.index[1], "transmission"] = pd.NA
    sample.loc[sample.index[2], "transmission"] = ""

    predictions = pipeline.predict(sample.loc[:, INPUT_COLUMNS])
    assert len(predictions) == len(sample)
    assert not np.isnan(predictions).any()


def test_entrada_invalida(trained_version, synthetic_data, tmp_path):
    version_dir = trained_version["version_dir"]
    missing_input = synthetic_data.drop(columns=["transmission"])
    with pytest.raises(InputValidationError, match="transmission"):
        validate_and_prepare_input(missing_input)

    missing_path = tmp_path / "missing_column.csv"
    missing_input.to_csv(missing_path, index=False)
    missing_output = tmp_path / "missing_output.csv"
    assert (
        predict.main(
            [
                "--model",
                str(version_dir),
                "--input",
                str(missing_path),
                "--output",
                str(missing_output),
            ]
        )
        != 0
    )
    assert not missing_output.exists()

    invalid_numeric = synthetic_data.copy()
    invalid_numeric["model_year"] = invalid_numeric["model_year"].astype(object)
    invalid_numeric.loc[invalid_numeric.index[0], "model_year"] = "abc"
    invalid_path = tmp_path / "invalid_model_year.csv"
    invalid_numeric.to_csv(invalid_path, index=False)
    invalid_output = tmp_path / "invalid_numeric_output.csv"
    assert (
        predict.main(
            [
                "--model",
                str(version_dir),
                "--input",
                str(invalid_path),
                "--output",
                str(invalid_output),
            ]
        )
        != 0
    )
    assert not invalid_output.exists()


def test_versionamiento(trained_version, training_csv):
    output_dir = trained_version["output_dir"]
    version_one = trained_version["version_dir"]
    model_path = version_one / "modelo.joblib"
    original_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    assert (
        train.main(
            [
                "--data",
                str(training_csv),
                "--version",
                "v2",
                "--output-dir",
                str(output_dir),
                "--n-estimators",
                "5",
                "--test-size",
                "0.2",
            ]
        )
        == 0
    )
    version_two = output_dir / "v2"
    assert version_one.is_dir()
    assert version_two.is_dir()
    assert load_model(version_one)[0] is not None
    assert load_model(version_two)[0] is not None

    assert (
        train.main(
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
        != 0
    )
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == original_hash


def test_categoria_desconocida(trained_version, synthetic_data):
    pipeline, _ = load_model(trained_version["version_dir"])
    sample = synthetic_data.iloc[:4].loc[:, INPUT_COLUMNS].copy()
    sample.loc[sample.index[0], ["brand", "model", "ext_col", "int_col"]] = [
        "Marca nueva",
        "Modelo nuevo",
        "Color exterior nuevo",
        "Color interior nuevo",
    ]
    sample.loc[
        sample.index[0],
        ["fuel_type", "accident", "clean_title"],
    ] = ["Combustible nuevo", "Estado nuevo", "Título nuevo"]

    predictions = pipeline.predict(sample)
    assert len(predictions) == len(sample)
    assert np.isfinite(predictions).all()


def test_columnas_reordenadas_y_extra(synthetic_data):
    reordered = synthetic_data.loc[:, list(reversed(INPUT_COLUMNS))].copy()
    reordered["columna_extra"] = "ignorar"

    prepared, ignored = validate_and_prepare_input(reordered)

    assert prepared.columns.tolist() == INPUT_COLUMNS
    assert ignored == ["columna_extra"]
    assert prepared.index.equals(reordered.index)


@pytest.mark.parametrize("entrypoint", [train.main, predict.main])
def test_help(entrypoint):
    with pytest.raises(SystemExit) as exc_info:
        entrypoint(["--help"])

    assert exc_info.value.code == 0
