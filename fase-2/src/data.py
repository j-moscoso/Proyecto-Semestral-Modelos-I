"""Carga y partición de datos para entrenamiento."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.schema import ID_COLUMN, INPUT_COLUMNS, InputValidationError


def load_training_data(path: str | Path, target: str) -> pd.DataFrame:
    """Carga un CSV de entrenamiento y valida sus columnas obligatorias."""
    source = Path(path)
    try:
        data = pd.read_csv(source)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise InputValidationError(
            f"No se pudo leer el archivo de entrenamiento '{source}': {error}"
        ) from error
    if data.empty:
        raise InputValidationError(
            f"El archivo de entrenamiento '{source}' está vacío."
        )
    required = [target, *INPUT_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Faltan columnas requeridas en '{source}': {', '.join(missing)}."
        )
    return data


def split_data(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa predictores y objetivo usando una partición reproducible."""
    x = df.drop(columns=[target, ID_COLUMN], errors="ignore")
    y = df[target]
    return train_test_split(x, y, test_size=test_size, random_state=seed)
