"""Analiza el efecto de no disponer de ``accident`` y ``clean_title``."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

FASE_2 = Path(__file__).resolve().parents[1]
if str(FASE_2) not in sys.path:
    sys.path.insert(0, str(FASE_2))

from src.data import load_training_data, split_data
from src.modeling import build_pipeline, compute_metrics


DEFAULT_PARAMS = {
    "n_estimators": 40,
    "max_depth": 20,
    "min_samples_leaf": 3,
    "max_features": 0.7,
}
MISSING_SCENARIOS = {
    "B1": ["accident"],
    "B2": ["clean_title"],
    "B3": ["accident", "clean_title"],
}


def read_metadata(path: Path) -> tuple[int, float, dict[str, Any]]:
    """Lee la configuración reproducible de una versión del modelo."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer metadata.json: {error}") from error

    seed = int(metadata.get("seed", 42))
    split = metadata.get("split", {})
    test_size = float(split.get("test_size", 0.2))
    stored_params = metadata.get("params", {})
    params = {
        name: stored_params.get(name, default)
        for name, default in DEFAULT_PARAMS.items()
    }
    return seed, test_size, params


def set_missing(data: Any, columns: list[str]) -> Any:
    """Devuelve una copia con las columnas indicadas ausentes al predecir."""
    result = data.copy()
    for column in columns:
        result[column] = np.nan
    return result


def evaluate(
    pipeline: Any,
    x_fit: Any,
    y_fit: Any,
    x_val: Any,
    y_val: Any,
) -> dict[str, float]:
    """Ajusta un pipeline y calcula sus métricas sobre la validación interna."""
    pipeline.fit(x_fit, y_fit)
    return compute_metrics(y_val, pipeline.predict(x_val))


def variation(rmse: float, reference_rmse: float) -> float:
    """Calcula el cambio porcentual del RMSE frente a la referencia."""
    return (rmse - reference_rmse) / reference_rmse * 100


def build_row(
    scenario: str,
    description: str,
    metrics: dict[str, float],
    reference_rmse: float,
) -> dict[str, object]:
    """Construye una fila del informe tabular."""
    return {
        "escenario": scenario,
        "descripcion": description,
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
        "variacion_RMSE_%": variation(metrics["rmse"], reference_rmse),
    }


def print_report(
    x_fit: Any,
    rows: list[dict[str, object]],
) -> None:
    """Imprime el porcentaje de nulos y la tabla de resultados."""
    print("Porcentaje de nulos en X_fit:")
    for column in ("accident", "clean_title"):
        print(f"  {column}: {x_fit[column].isna().mean() * 100:.6f}%")

    headers = [
        "escenario",
        "descripcion",
        "RMSE",
        "MAE",
        "R2",
        "variacion_RMSE_%",
    ]
    print("\n" + "\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row[header])
                    if header in {"escenario", "descripcion"}
                    else f"{float(row[header]):.6f}"
                    for header in headers
                ]
            )
        )
    print(
        "\nAnálisis descriptivo sobre una validación interna de train con datos "
        "sintéticos; la prueba reservada no se usó ni se usa para elegir nada."
    )


def main() -> int:
    """Ejecuta el análisis de disponibilidad con una validación interna."""
    parser = argparse.ArgumentParser(
        description=(
            "Compara predicciones con y sin disponibilidad de accident y clean_title."
        )
    )
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--metadata", default="models/v1/metadata.json")
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    if not 0 < args.val_size < 1:
        parser.error("--val-size debe estar entre 0 y 1.")

    seed, test_size, params = read_metadata(Path(args.metadata))
    data = load_training_data(args.data, "price")
    x_train, _x_test, y_train, _y_test = split_data(
        data, "price", test_size, seed
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.val_size,
        random_state=seed,
    )

    reference_pipeline = build_pipeline(seed, **params)
    reference_metrics = evaluate(
        reference_pipeline, x_fit, y_fit, x_val, y_val
    )
    reference_rmse = reference_metrics["rmse"]
    rows = [
        build_row(
            "A",
            "Referencia: ambas disponibles",
            reference_metrics,
            reference_rmse,
        )
    ]

    for scenario, columns in MISSING_SCENARIOS.items():
        metrics = compute_metrics(
            y_val,
            reference_pipeline.predict(set_missing(x_val, columns)),
        )
        rows.append(
            build_row(
                scenario,
                f"Sin {', '.join(columns)}; mismo modelo",
                metrics,
                reference_rmse,
            )
        )

    for scenario, columns in MISSING_SCENARIOS.items():
        retrained_pipeline = build_pipeline(seed, **params)
        metrics = evaluate(
            retrained_pipeline,
            set_missing(x_fit, columns),
            y_fit,
            set_missing(x_val, columns),
            y_val,
        )
        rows.append(
            build_row(
                f"C{scenario[1]}",
                f"Sin {', '.join(columns)}; modelo reentrenado",
                metrics,
                reference_rmse,
            )
        )

    print_report(x_fit, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
