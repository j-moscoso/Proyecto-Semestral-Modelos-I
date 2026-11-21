"""Interfaz de línea de comandos para resumir el registro de solicitudes."""

import argparse
import sys
from collections.abc import Sequence

from src.monitoring import (
    RegistroInvalidoError,
    calcular_metricas,
    get_log_path,
    leer_registro,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Imprime las métricas del registro y devuelve el código de salida."""
    parser = argparse.ArgumentParser(description="Resume el registro de la API.")
    parser.add_argument(
        "--log",
        default=str(get_log_path()),
        help="Ruta del archivo CSV de solicitudes.",
    )
    arguments = parser.parse_args(argv)

    try:
        rows = leer_registro(arguments.log)
    except RegistroInvalidoError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    metrics = calcular_metricas(rows)
    total = metrics["total"]
    failed = metrics["fallidas"]
    percentage = metrics["porcentaje"]
    print(f"Total de solicitudes: {total}")
    print(f"Solicitudes fallidas: {failed}")
    if percentage is None:
        print("Porcentaje de fallos: sin solicitudes")
    else:
        formatted_percentage = f"{percentage:.1f}".replace(".", ",")
        print(f"Porcentaje de fallos: {formatted_percentage}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
