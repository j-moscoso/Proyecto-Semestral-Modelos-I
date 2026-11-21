"""Validación y preparación de los datos de entrada para predicción."""

import pandas as pd

from src.features import normalize_missing
from src.schema import INPUT_COLUMNS, NUMERIC_INPUT, InputValidationError


def validate_and_prepare_input(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Valida las columnas y devuelve los predictores en orden de entrenamiento."""
    if not isinstance(df, pd.DataFrame):
        raise InputValidationError("La entrada debe ser un DataFrame.")
    if df.empty:
        raise InputValidationError("El DataFrame de entrada está vacío.")

    required_columns = list(
        dict.fromkeys([*(expected_columns or INPUT_COLUMNS), *INPUT_COLUMNS])
    )
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise InputValidationError(
            "Faltan columnas requeridas: " + ", ".join(missing_columns) + "."
        )

    numeric = NUMERIC_INPUT if numeric_columns is None else numeric_columns
    original_non_null = {
        column: ~df[column].isna()
        for column in numeric
        if column in df.columns
    }
    normalized = normalize_missing(df)

    for column, non_null_mask in original_non_null.items():
        invalid_mask = non_null_mask & normalized[column].isna()
        if invalid_mask.any():
            row_numbers = [
                position + 2
                for position, invalid in enumerate(invalid_mask.to_numpy())
                if invalid
            ][:5]
            raise InputValidationError(
                f"Valores no numéricos en la columna '{column}' "
                f"(filas del CSV: {', '.join(map(str, row_numbers))})."
            )

    ignored_columns = [
        str(column) for column in df.columns if column not in INPUT_COLUMNS
    ]
    return normalized.loc[:, INPUT_COLUMNS].copy(), ignored_columns
