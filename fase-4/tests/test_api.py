"""Pruebas de contrato para la API REST de predicción y entrenamiento."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.schema import INPUT_COLUMNS
from src.validation import validate_and_prepare_input


def test_health_responde_200(api_client):
    """Confirma el estado del servicio y la versión sintética disponible."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["estado"] == "ok"
    assert "v1" in response.json()["versiones_disponibles"]


def test_predict_valido_devuelve_prediccion_y_version(api_client, registro_valido):
    """Comprueba la predicción exitosa y la versión usada."""
    response = api_client.post("/predict", json=registro_valido)

    assert response.status_code == 200
    assert np.isfinite(response.json()["prediccion"])
    assert response.json()["version"] == "v1"


def test_predict_campo_obligatorio_ausente_se_rechaza(api_client, registro_valido):
    """Rechaza una solicitud que no incluye una columna obligatoria."""
    record = registro_valido.copy()
    del record["brand"]

    response = api_client.post("/predict", json=record)

    assert response.status_code == 422
    assert "brand" in response.json()["mensaje"]


def test_predict_acepta_nulos(api_client, registro_valido):
    """Acepta null en todos los campos de entrada."""
    record = {column: None for column in INPUT_COLUMNS}

    response = api_client.post("/predict", json=record)

    assert response.status_code == 200


def test_predict_tipo_incompatible(api_client, registro_valido):
    """Rechaza tipos incompatibles y comunica ambos campos afectados."""
    record = registro_valido.copy()
    record["model_year"] = "abc"
    record["brand"] = 123

    response = api_client.post("/predict", json=record)

    assert response.status_code == 422
    assert "model_year" in response.json()["mensaje"]
    assert "brand" in response.json()["mensaje"]


def test_predict_ignora_price_e_id(api_client, registro_valido):
    """Ignora las columnas adicionales de precio e identificador."""
    record = {**registro_valido, "price": 25_000, "id": 123}

    response = api_client.post("/predict", json=record)

    assert response.status_code == 200


def test_predict_modelo_no_disponible(api_client, registro_valido, monkeypatch):
    """Devuelve 503 y enumera las versiones cuando falta la solicitada."""
    monkeypatch.setenv("MODEL_VERSION", "v99")

    response = api_client.post("/predict", json=registro_valido)

    assert response.status_code == 503
    assert "v1" in response.json()["mensaje"]


def test_predict_coincide_con_pipeline_directo(
    api_client, registro_valido, trained_version
):
    """Compara la respuesta de la API con el pipeline sintético sin persistir."""
    prepared, _ = validate_and_prepare_input(pd.DataFrame([registro_valido]))
    expected = trained_version["pipeline_before_save"].predict(prepared)[0]

    response = api_client.post("/predict", json=registro_valido)

    assert response.status_code == 200
    assert np.isclose(response.json()["prediccion"], expected)


def test_train_crea_version_nueva_sin_tocar_v1(api_client, trained_version):
    """Crea v2 en el directorio temporal sin modificar el modelo v1."""
    model_v1 = trained_version["version_dir"] / "modelo.joblib"
    original_hash = hashlib.sha256(model_v1.read_bytes()).hexdigest()

    response = api_client.post("/train", json={"n_estimators": 5})

    assert response.status_code == 201
    assert response.json()["version"] == "v2"
    assert response.json()["activada"] is False
    assert (trained_version["output_dir"] / "v2" / "modelo.joblib").is_file()
    assert hashlib.sha256(model_v1.read_bytes()).hexdigest() == original_hash


def test_prediccion_coincide_con_referencia_v1(monkeypatch):
    """Compara las doce predicciones reales con el CSV de referencia v1."""
    from src import serving
    from fastapi.testclient import TestClient

    fase_dir = Path(__file__).resolve().parents[1]
    models_dir = fase_dir / "models"
    model_path = models_dir / "v1" / "modelo.joblib"
    input_path = fase_dir / "data" / "ejemplo_prediccion.csv"
    reference_path = (
        Path(__file__).resolve().parent
        / "referencia"
        / "predicciones_v1_ejemplo.csv"
    )
    if not model_path.is_file():
        pytest.skip("No existe models/v1/modelo.joblib para la referencia.")
    if not input_path.is_file():
        pytest.skip("No existe data/ejemplo_prediccion.csv para la referencia.")

    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    serving.clear_model_cache()

    import api

    input_data = pd.read_csv(input_path)
    expected = pd.read_csv(reference_path)
    assert len(input_data) == 12
    assert len(expected) == 12
    assert input_data["id"].tolist() == expected["id"].tolist()

    client = TestClient(api.app)
    for row_number, row in input_data.iterrows():
        record = {}
        for column in INPUT_COLUMNS:
            value = row[column]
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, np.generic):
                record[column] = value.item()
            else:
                record[column] = value

        response = client.post("/predict", json=record)

        assert response.status_code == 200
        assert np.isclose(
            response.json()["prediccion"],
            expected.iloc[row_number]["prediccion"],
            rtol=1e-6,
        )
