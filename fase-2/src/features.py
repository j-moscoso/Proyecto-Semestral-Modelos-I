"""Funciones de ingeniería de características y preprocesamiento."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.schema import CATEGORICAL_INPUT, NUMERIC_INPUT


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Copia el DataFrame y normaliza nulos en las columnas de entrada presentes.

    En categóricas, ``None``, ``pd.NA``, ``NaN``, cadenas vacías y espacios
    pasan a ``np.nan``; los demás valores se convierten a ``str`` y el dtype
    queda como ``object``. Las numéricas se convierten a ``float64`` usando
    ``errors="coerce"``. Las columnas ausentes se ignoran.
    """
    result = df.copy()
    for column in CATEGORICAL_INPUT:
        if column in result.columns:
            values = result[column]
            result[column] = pd.Series(
                [
                    np.nan
                    if pd.isna(value) or (isinstance(value, str) and not value.strip())
                    else str(value)
                    for value in values
                ],
                index=result.index,
                dtype=object,
            )
    for column in NUMERIC_INPUT:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("float64")
    return result


def extract_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copia el DataFrame y extrae HP, litros y cilindros de ``engine``.

    El texto se procesa temporalmente con dtype ``string``; las extracciones
    numéricas quedan como ``float64`` y usan ``np.nan`` si el texto está
    ausente o no coincide con el patrón.
    """
    result = df.copy()
    engine = result["engine"].astype("string")
    result["horsepower"] = pd.to_numeric(
        engine.str.extract(r"(?i)(\d+(?:\.\d+)?)\s*HP", expand=False),
        errors="coerce",
    ).astype("float64")
    result["engine_liters"] = pd.to_numeric(
        engine.str.extract(r"(?i)(\d+(?:\.\d+)?)\s*L\b", expand=False),
        errors="coerce",
    ).astype("float64")
    result["cylinders"] = pd.to_numeric(
        engine.str.extract(r"(?i)(\d+)\s*Cyl(?:inder)?s?\b", expand=False),
        errors="coerce",
    ).astype("float64")
    return result


def extract_transmission_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copia el DataFrame y extrae tipo y velocidades de transmisión.

    El texto se procesa temporalmente con dtype ``string``. Las velocidades
    quedan como ``float64`` y ``transmission_type`` como ``object``; texto
    ausente, vacío o compuesto solo por espacios produce ``np.nan`` para que
    el imputador posterior lo sustituya por ``Desconocido``.
    """
    result = df.copy()
    transmission = result["transmission"].astype("string").str.strip()
    normalized = transmission.str.lower()
    is_cvt = normalized.str.contains(r"\bcvt\b|continuously variable", na=False)
    is_manual = normalized.str.contains(
        r"\bm/t\b|manual|dual-clutch|\bdct\b", na=False
    )
    is_automatic = normalized.str.contains(
        r"\ba/t\b|automatic|auto\b|dual shift", na=False
    )
    transmission_type = pd.Series(np.nan, index=result.index, dtype=object)
    transmission_type.loc[is_cvt] = "CVT"
    transmission_type.loc[is_manual] = "manual"
    transmission_type.loc[is_automatic] = "automatica"
    has_transmission = transmission.notna() & transmission.fillna("").ne("")
    transmission_type.loc[
        has_transmission & transmission_type.isna()
    ] = "otra"
    result["transmission_type"] = transmission_type.astype(object)
    result["transmission_speeds"] = pd.to_numeric(
        transmission.str.extract(
            r"(?i)\b(\d+)\s*[- ]?\s*speed\b", expand=False
        ),
        errors="coerce",
    ).astype("float64")
    return result


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador de variables numéricas y categóricas."""
    numeric_features = [
        *NUMERIC_INPUT,
        "horsepower",
        "engine_liters",
        "cylinders",
        "transmission_speeds",
    ]
    ordinal_features = ["brand", "model", "ext_col", "int_col"]
    onehot_features = [
        "fuel_type",
        "transmission_type",
        "accident",
        "clean_title",
    ]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    onehot_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("ordinal", ordinal_pipeline, ordinal_features),
            ("onehot", onehot_pipeline, onehot_features),
        ],
        remainder="drop",
    )
