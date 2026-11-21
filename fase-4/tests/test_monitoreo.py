"""Pruebas del registro HTTP y del resumen de monitoreo."""

import csv
import math
import os
from pathlib import Path

import monitor
from src.monitoring import LOG_COLUMNS


def _leer_csv(ruta):
    """Devuelve encabezado y filas de un CSV de registro."""
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        reader = csv.reader(archivo)
        return next(reader), list(reader)


def test_registro_tres_intentos_tres_filas(api_client, registro_valido):
    """Registra dos predicciones exitosas y un rechazo por campo faltante."""
    api_client.post("/predict", json=registro_valido)
    api_client.post("/predict", json=registro_valido)
    registro_incompleto = registro_valido.copy()
    registro_incompleto.pop("brand")
    api_client.post("/predict", json=registro_incompleto)

    encabezado, filas = _leer_csv(Path(os.environ["LOG_PATH"]))

    assert encabezado == LOG_COLUMNS
    assert filas.count(LOG_COLUMNS) == 0
    assert [int(fila[1]) for fila in filas] == [200, 200, 422]
    for fila in filas[:2]:
        assert fila[2] == "v1"
        assert math.isfinite(float(fila[3]))
    assert filas[2][2] == ""
    assert filas[2][3] == ""


def test_registro_se_conserva_entre_ejecuciones(api_client, registro_valido):
    """Conserva las filas al usar una nueva instancia de TestClient."""
    api_client.post("/predict", json=registro_valido)

    from fastapi.testclient import TestClient

    import api

    with TestClient(api.app) as segundo_cliente:
        segundo_cliente.post("/predict", json=registro_valido)

    encabezado, filas = _leer_csv(Path(os.environ["LOG_PATH"]))

    assert encabezado == LOG_COLUMNS
    assert encabezado not in filas
    assert len(filas) == 2
    assert [int(fila[1]) for fila in filas] == [200, 200]


def test_registro_503_modelo_no_disponible(
    api_client, registro_valido, monkeypatch
):
    """Registra el rechazo 503 sin versión ni predicción."""
    monkeypatch.setenv("MODEL_VERSION", "v99")

    response = api_client.post("/predict", json=registro_valido)

    encabezado, filas = _leer_csv(Path(os.environ["LOG_PATH"]))

    assert response.status_code == 503
    assert encabezado == LOG_COLUMNS
    assert len(filas) == 1
    assert filas[0][1] == "503"
    assert filas[0][2] == ""
    assert filas[0][3] == ""


def test_registro_json_invalido(api_client):
    """Registra como 422 un cuerpo que no contiene JSON válido."""
    response = api_client.post(
        "/predict",
        content="{json inválido",
        headers={"Content-Type": "application/json"},
    )

    encabezado, filas = _leer_csv(Path(os.environ["LOG_PATH"]))

    assert response.status_code == 422
    assert encabezado == LOG_COLUMNS
    assert len(filas) == 1
    assert filas[0][1] == "422"


def test_health_y_train_no_se_registran(api_client, tmp_path):
    """Evita registrar las rutas de salud y entrenamiento."""
    health_response = api_client.get("/health")
    train_response = api_client.post("/train", json={"n_estimators": 5})

    assert health_response.status_code == 200
    assert train_response.status_code == 201
    assert not (tmp_path / "logs" / "predicciones.csv").exists()


def test_monitor_calcula_total_y_porcentaje(
    api_client, registro_valido, capsys
):
    """Resume tres intentos con una solicitud fallida."""
    api_client.post("/predict", json=registro_valido)
    api_client.post("/predict", json=registro_valido)
    registro_incompleto = registro_valido.copy()
    registro_incompleto.pop("brand")
    api_client.post("/predict", json=registro_incompleto)

    result = monitor.main(["--log", os.environ["LOG_PATH"]])
    output = capsys.readouterr().out

    assert result == 0
    assert "Total de solicitudes: 3" in output
    assert "Solicitudes fallidas: 1" in output
    assert "Porcentaje de fallos: 33,3%" in output


def test_monitor_sin_solicitudes(tmp_path, capsys):
    """Resume de igual forma un archivo ausente o vacío."""
    archivo_inexistente = tmp_path / "inexistente.csv"
    archivo_vacio = tmp_path / "vacio.csv"
    archivo_vacio.touch()

    for ruta in (archivo_inexistente, archivo_vacio):
        result = monitor.main(["--log", str(ruta)])
        output = capsys.readouterr().out

        assert result == 0
        assert "Total de solicitudes: 0" in output
        assert "Porcentaje de fallos: sin solicitudes" in output


def test_monitor_registro_invalido(tmp_path, capsys):
    """Devuelve código 1 y un mensaje de error ante un encabezado inválido."""
    ruta = tmp_path / "registro_invalido.csv"
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        csv.writer(archivo).writerow(["fecha", "estado"])

    result = monitor.main(["--log", str(ruta)])
    captured = capsys.readouterr()

    assert result == 1
    assert "ERROR" in captured.err
