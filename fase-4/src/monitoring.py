"""Registro y cálculo de métricas para las solicitudes de la API."""

import csv
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.schema import AppError


LOG_COLUMNS = ["fecha_hora", "estado_http", "version", "prediccion"]
_LOG_LOCK = threading.Lock()


class RegistroInvalidoError(AppError):
    """Indica que el archivo de registro no cumple el formato esperado."""


def get_log_path() -> Path:
    """Devuelve la ruta de registro configurada o su valor predeterminado."""
    return Path(os.environ.get("LOG_PATH", "logs/predicciones.csv"))


def registrar_solicitud(
    estado_http: int,
    version: str | None,
    prediccion: float | None,
    ruta: str | Path | None = None,
) -> None:
    """Agrega una solicitud al CSV sin propagar errores de escritura."""
    try:
        log_path = get_log_path() if ruta is None else Path(ruta)
        with _LOG_LOCK:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not log_path.exists() or log_path.stat().st_size == 0
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            prediction = "" if prediccion is None else round(prediccion, 2)
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                writer = csv.writer(log_file)
                if needs_header:
                    writer.writerow(LOG_COLUMNS)
                writer.writerow(
                    [
                        timestamp,
                        estado_http,
                        "" if version is None else version,
                        prediction,
                    ]
                )
    except Exception as error:
        print(
            f"Aviso: no se pudo escribir el registro: {error}",
            file=sys.stderr,
        )


def leer_registro(ruta: str | Path | None = None) -> list[dict[str, str]]:
    """Lee y valida el encabezado y el código HTTP de cada fila del registro."""
    log_path = get_log_path() if ruta is None else Path(ruta)
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []

    with log_path.open("r", encoding="utf-8", newline="") as log_file:
        reader = csv.reader(log_file)
        try:
            header = next(reader)
        except StopIteration:
            return []
        if header != LOG_COLUMNS:
            raise RegistroInvalidoError("Encabezado del registro inválido.")

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(LOG_COLUMNS):
                raise RegistroInvalidoError(
                    f"Fila {line_number} del registro inválida."
                )
            try:
                status_code = int(row[1])
            except ValueError as error:
                raise RegistroInvalidoError(
                    f"estado_http inválido en la fila {line_number}."
                ) from error
            rows.append(dict(zip(LOG_COLUMNS, row, strict=True)))
            rows[-1]["estado_http"] = str(status_code)
    return rows


def calcular_metricas(filas: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Calcula el total, solicitudes fallidas y su porcentaje."""
    total = len(filas)
    failed = sum(400 <= int(row["estado_http"]) <= 599 for row in filas)
    percentage = None if total == 0 else failed / total * 100
    return {"total": total, "fallidas": failed, "porcentaje": percentage}
