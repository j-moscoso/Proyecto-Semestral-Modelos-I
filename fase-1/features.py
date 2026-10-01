"""Funciones de ingeniería de características y preprocesamiento para autos usados."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def extract_engine_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Devuelve una copia con HP, litros y cilindros extraídos de `engine`.

	Las extracciones son fila a fila y no calculan estadísticas del conjunto.
	Los textos ausentes o que no coincidan con los patrones quedan como NaN.
	"""
	result = df.copy()
	engine = result["engine"].astype("string")

	result["horsepower"] = pd.to_numeric(
		engine.str.extract(r"(?i)(\d+(?:\.\d+)?)\s*HP", expand=False),
		errors="coerce",
	)
	result["engine_liters"] = pd.to_numeric(
		engine.str.extract(r"(?i)(\d+(?:\.\d+)?)\s*L\b", expand=False),
		errors="coerce",
	)
	result["cylinders"] = pd.to_numeric(
		engine.str.extract(r"(?i)(\d+)\s*Cyl(?:inder)?s?\b", expand=False),
		errors="coerce",
	)
	return result


def extract_transmission_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Devuelve una copia con el tipo simplificado y velocidades de transmisión.

	Los tipos resultantes son `automatica`, `manual`, `CVT` y `otra`;
	valores ausentes se mantienen como NaN para imputarlos dentro del pipeline.
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

	result["transmission_type"] = pd.Series(pd.NA, index=result.index, dtype="string")
	result.loc[is_cvt, "transmission_type"] = "CVT"
	result.loc[is_manual, "transmission_type"] = "manual"
	result.loc[is_automatic, "transmission_type"] = "automatica"
	result.loc[transmission.notna() & result["transmission_type"].isna(), "transmission_type"] = "otra"
	result["transmission_speeds"] = pd.to_numeric(
		transmission.str.extract(r"(?i)\b(\d+)\s*[- ]?\s*speed\b", expand=False),
		errors="coerce",
	)
	return result


def build_preprocessor() -> ColumnTransformer:
	"""Construye el preprocesador sin ajustarlo ni consultar datos externos.

	Las variables numéricas se imputan con la mediana y se estandarizan. Las
	categorías compactas se imputan con `Desconocido` y se codifican con one-hot;
	brand, model y colores usan OrdinalEncoder para evitar una matriz one-hot
	potencialmente enorme. En modelos lineales esa codificación ordinal puede
	imponer un orden artificial; para ellos conviene evaluar hashing o encoding
	supervisado dentro de validación cruzada.
	"""
	numeric_features = [
		"model_year",
		"milage",
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
