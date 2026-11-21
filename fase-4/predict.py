"""CLI para generar predicciones de precios de vehículos usados."""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from src.schema import AppError, InputValidationError, ModelNotFoundError
from src.validation import validate_and_prepare_input
from src.versioning import load_model


def main(argv: list[str] | None = None) -> int:
    """Valida los archivos, predice y guarda el CSV de forma atómica."""
    parser = argparse.ArgumentParser(
        description="Predice precios de vehículos usados a partir de un CSV.",
        epilog=(
            "Ejemplo: python predict.py --model models/v1 "
            "--input data/ejemplo_prediccion.csv --output outputs/predicciones.csv"
        ),
    )
    parser.add_argument("--model", required=True, help="Carpeta o archivo del modelo.")
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada.")
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    args = parser.parse_args(argv)

    temporary_path: Path | None = None
    try:
        model_path = Path(args.model)
        input_path = Path(args.input)
        output_path = Path(args.output)
        model_file = (
            model_path / "modelo.joblib" if model_path.is_dir() else model_path
        )
        if not model_file.is_file():
            raise ModelNotFoundError(f"No existe el modelo: '{model_file}'.")
        if not input_path.is_file():
            raise InputValidationError(
                f"No existe el archivo CSV de entrada: '{input_path}'."
            )

        try:
            input_df = pd.read_csv(input_path)
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            PermissionError,
            FileNotFoundError,
        ) as error:
            raise InputValidationError(
                f"No se pudo leer el CSV de entrada '{input_path}': {error}"
            ) from error

        pipeline, metadata = load_model(model_path)
        schema = metadata.get("input_schema")
        if schema is not None:
            X, ignored_columns = validate_and_prepare_input(
                input_df,
                expected_columns=schema["columns"],
                numeric_columns=schema["numeric"],
            )
        else:
            X, ignored_columns = validate_and_prepare_input(input_df)

        predictions = pipeline.predict(X)
        output_data: dict[str, object] = {}
        if "id" in input_df.columns:
            output_data["id"] = input_df["id"]
        output_data["prediccion"] = pd.Series(
            predictions, index=X.index, dtype="float64"
        )
        output_df = pd.DataFrame(output_data, index=X.index)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        output_df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
        temporary_path = None

        ignored_summary = ", ".join(ignored_columns) if ignored_columns else "ninguna"
        print(f"Filas predichas: {len(output_df)}")
        print(f"Columnas ignoradas: {ignored_summary}")
        print(f"Salida: {output_path}")
        return 0
    except AppError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"ERROR inesperado (posible bug): {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError as error:
                print(
                    f"ERROR: No se pudo eliminar el archivo temporal "
                    f"'{temporary_path}': {error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main())
