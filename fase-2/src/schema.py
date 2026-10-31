"""Constantes y excepciones compartidas por la aplicación."""

TARGET_COLUMN = "price"
ID_COLUMN = "id"
NUMERIC_INPUT = ["model_year", "milage"]
INPUT_COLUMNS = [
    "brand",
    "model",
    "model_year",
    "milage",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
    "clean_title",
]
CATEGORICAL_INPUT = [column for column in INPUT_COLUMNS if column not in NUMERIC_INPUT]


class AppError(Exception):
    """Error base de la aplicación."""


class InputValidationError(AppError):
    """Error de validación de datos de entrada."""


class VersionExistsError(AppError):
    """Indica que una versión ya existe."""


class ModelNotFoundError(AppError):
    """Indica que un modelo no existe o no puede cargarse."""
