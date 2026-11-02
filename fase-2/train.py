"""Punto de entrada para entrenar y versionar modelos de precios."""

import argparse
import sys
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold, cross_validate

from src.data import load_training_data, split_data
from src.modeling import build_baseline, build_pipeline, compute_metrics
from src.schema import AppError, InputValidationError
from src.versioning import (
    build_metadata,
    resolve_version_dir,
    save_version,
)


NOTES = (
    "Partición aleatoria; no hay identificador de vehículo para agrupar. "
    "Anuncios similares entre train y test pueden hacer optimista la evaluación."
)


def run_training(args: argparse.Namespace) -> int:
    """Ejecuta el flujo completo de entrenamiento y persistencia."""
    version_dir = resolve_version_dir(args.output_dir, args.version)
    data = load_training_data(args.data, args.target)
    x_train, x_test, y_train, y_test = split_data(
        data, args.target, args.test_size, args.seed
    )

    if args.cv_folds > len(x_train):
        raise InputValidationError(
            "cv-folds no puede ser mayor que el número de filas de entrenamiento."
        )

    pipeline = build_pipeline(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    )
    pipeline.fit(x_train, y_train)

    test_metrics = {
        "random_forest": compute_metrics(y_test, pipeline.predict(x_test))
    }
    for strategy in ("mean", "median"):
        baseline = build_baseline(strategy)
        baseline.fit(x_train, y_train)
        test_metrics[f"dummy_{strategy}"] = compute_metrics(
            y_test, baseline.predict(x_test)
        )

    cv_metrics: dict[str, Any] | None = None
    if args.cv_folds > 0:
        cv_results = cross_validate(
            build_pipeline(
                seed=args.seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
            ),
            x_train,
            y_train,
            cv=KFold(
                n_splits=args.cv_folds,
                shuffle=True,
                random_state=args.seed,
            ),
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
        )
        cv_metrics = {
            "folds": args.cv_folds,
            "rmse_mean": float(-cv_results["test_rmse"].mean()),
            "rmse_std": float(cv_results["test_rmse"].std()),
            "mae_mean": float(-cv_results["test_mae"].mean()),
            "r2_mean": float(cv_results["test_r2"].mean()),
        }

    metrics = {
        "version": args.version,
        "metrica_principal": "RMSE (USD)",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test": test_metrics,
        "cv": cv_metrics,
    }
    metadata = build_metadata(
        version=args.version,
        target=args.target,
        params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "cv_folds": args.cv_folds,
        },
        seed=args.seed,
        data_path=args.data,
        n_rows=len(data),
        n_train=len(x_train),
        n_test=len(x_test),
        test_size=args.test_size,
        notes=NOTES,
    )
    save_version(version_dir, pipeline, metrics, metadata)

    model_metrics = test_metrics["random_forest"]
    mean_metrics = test_metrics["dummy_mean"]
    median_metrics = test_metrics["dummy_median"]
    rmse_improvement = 100 * (
        mean_metrics["rmse"] - model_metrics["rmse"]
    ) / mean_metrics["rmse"]
    mae_improvement = 100 * (mean_metrics["mae"] - model_metrics["mae"]) / mean_metrics[
        "mae"
    ]
    print(f"Versión guardada: {args.version}")
    print(
        "Modelo - "
        f"RMSE: {model_metrics['rmse']:.4f}, "
        f"MAE: {model_metrics['mae']:.4f}, "
        f"R²: {model_metrics['r2']:.4f}"
    )
    print(
        "Baseline media - "
        f"RMSE: {mean_metrics['rmse']:.4f}, "
        f"MAE: {mean_metrics['mae']:.4f}, "
        f"R²: {mean_metrics['r2']:.4f}"
    )
    print(
        "Baseline mediana - "
        f"RMSE: {median_metrics['rmse']:.4f}, "
        f"MAE: {median_metrics['mae']:.4f}, "
        f"R²: {median_metrics['r2']:.4f}"
    )
    print(
        f"Mejora frente a la media - RMSE: {rmse_improvement:.2f}%, "
        f"MAE: {mae_improvement:.2f}%"
    )
    print(
        "Rutas generadas: "
        f"{version_dir / 'modelo.joblib'}, "
        f"{version_dir / 'metrics.json'}, "
        f"{version_dir / 'metadata.json'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Valida argumentos, ejecuta el entrenamiento y traduce errores esperables."""
    parser = argparse.ArgumentParser(
        description="Entrena y versiona un modelo de precios de vehículos usados.",
        epilog=(
            "Ejemplo: python train.py --data data/train.csv "
            "--version v1 --n-estimators 40"
        ),
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento.")
    parser.add_argument(
        "--target", default="price", help="Columna objetivo (por defecto: price)."
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directorio de salida (por defecto: models).",
    )
    parser.add_argument("--version", required=True, help="Nombre de la versión.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción para test, entre 0 y 1 (por defecto: 0.2).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=40,
        help="Número de árboles (por defecto: 40).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Profundidad máxima de los árboles (por defecto: 20).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=3,
        help="Mínimo de muestras por hoja (por defecto: 3).",
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=0.7,
        help="Fracción de variables por árbol (por defecto: 0.7).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Número de folds; 0 desactiva CV (por defecto: 0).",
    )
    args = parser.parse_args(argv)

    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1.")
    if args.n_estimators < 1:
        parser.error("--n-estimators debe ser positivo.")
    if args.max_depth < 1:
        parser.error("--max-depth debe ser positivo.")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf debe ser positivo.")
    if not 0 < args.max_features <= 1:
        parser.error("--max-features debe estar en el intervalo (0, 1].")
    if args.cv_folds < 0:
        parser.error("--cv-folds no puede ser negativo.")

    try:
        return run_training(args)
    except AppError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
