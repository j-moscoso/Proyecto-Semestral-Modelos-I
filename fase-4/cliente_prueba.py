"""Cliente de prueba de la API REST mediante la biblioteca estándar."""

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def leer_filas() -> list[dict[str, Any]]:
    """Lee las filas de ejemplo y convierte los valores de entrada."""
    ruta = Path(__file__).resolve().parent / "data" / "ejemplo_prediccion.csv"
    filas: list[dict[str, Any]] = []
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            registro: dict[str, Any] = {}
            for clave, valor in fila.items():
                if valor is None or not valor.strip():
                    registro[clave] = None
                elif clave in ("model_year", "milage"):
                    try:
                        registro[clave] = float(valor)
                    except ValueError:
                        registro[clave] = valor
                else:
                    registro[clave] = valor
            filas.append(registro)
    return filas


def solicitar(
    url: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 30,
) -> tuple[int, Any]:
    """Realiza una solicitud HTTP y devuelve el estado y el cuerpo JSON."""
    datos = None
    headers = {"Accept": "application/json"}
    if body is not None:
        datos = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    solicitud = urllib.request.Request(
        url,
        data=datos,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
            return respuesta.status, json.loads(respuesta.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        contenido = error.read().decode("utf-8")
        try:
            return error.code, json.loads(contenido)
        except json.JSONDecodeError:
            return error.code, contenido
    except urllib.error.URLError as error:
        raise ConnectionError(str(error.reason)) from error
    except TimeoutError as error:
        raise ConnectionError(str(error)) from error


def imprimir_respuesta(codigo: int, cuerpo: Any) -> None:
    """Imprime el código HTTP y el cuerpo JSON de una respuesta."""
    print(codigo)
    print(json.dumps(cuerpo, indent=2, ensure_ascii=False))


def crear_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos del cliente."""
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API")
    parser.add_argument("--url", default="http://localhost:8000")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("health", help="Consulta el estado del servicio")
    predecir = subparsers.add_parser("predecir", help="Predice una fila")
    predecir.add_argument("--fila", type=int, default=1)
    subparsers.add_parser("todos", help="Predice las 12 filas")
    subparsers.add_parser("faltante", help="Envía una fila sin brand")
    subparsers.add_parser("entrenar", help="Solicita un entrenamiento")
    subparsers.add_parser("demo", help="Envía dos predicciones y una entrada inválida")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el subcomando solicitado y devuelve su código de salida."""
    args = crear_parser().parse_args(argv)
    try:
        if args.comando == "health":
            codigo, cuerpo = solicitar(f"{args.url}/health")
            imprimir_respuesta(codigo, cuerpo)
        elif args.comando == "predecir":
            filas = leer_filas()
            if args.fila < 1 or args.fila > len(filas):
                print(
                    f"ERROR: --fila debe estar entre 1 y {len(filas)}",
                    file=sys.stderr,
                )
                return 1
            codigo, cuerpo = solicitar(
                f"{args.url}/predict",
                method="POST",
                body=filas[args.fila - 1],
            )
            imprimir_respuesta(codigo, cuerpo)
        elif args.comando == "todos":
            for fila in leer_filas():
                codigo, cuerpo = solicitar(
                    f"{args.url}/predict",
                    method="POST",
                    body=fila,
                )
                imprimir_respuesta(codigo, cuerpo)
                if isinstance(cuerpo, dict) and codigo == 200:
                    print(
                        f"id={fila.get('id')} "
                        f"prediccion={cuerpo.get('prediccion')} "
                        f"version={cuerpo.get('version')}"
                    )
        elif args.comando == "faltante":
            fila = dict(leer_filas()[0])
            fila.pop("brand", None)
            codigo, cuerpo = solicitar(
                f"{args.url}/predict",
                method="POST",
                body=fila,
            )
            imprimir_respuesta(codigo, cuerpo)
        elif args.comando == "entrenar":
            codigo, cuerpo = solicitar(
                f"{args.url}/train",
                method="POST",
                body={},
                timeout=900,
            )
            imprimir_respuesta(codigo, cuerpo)
        else:
            filas = leer_filas()
            for fila in filas[:2]:
                codigo, cuerpo = solicitar(
                    f"{args.url}/predict",
                    method="POST",
                    body=fila,
                )
                imprimir_respuesta(codigo, cuerpo)

            fila_incompleta = dict(filas[0])
            fila_incompleta.pop("brand", None)
            codigo, cuerpo = solicitar(
                f"{args.url}/predict",
                method="POST",
                body=fila_incompleta,
            )
            imprimir_respuesta(codigo, cuerpo)
        return 0
    except (ConnectionError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
