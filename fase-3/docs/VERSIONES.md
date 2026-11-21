# Versiones de modelos

Las métricas y los parámetros de esta tabla se transcriben de los archivos
`metrics.json` y `metadata.json` de cada versión. RMSE y MAE están expresados en
dólares estadounidenses; R² no tiene unidad.

| Versión | Algoritmo y parámetros | Semilla | Tamaños train/test | Métricas del modelo (RMSE / MAE / R²) | Baseline media (RMSE / MAE / R²) | Baseline mediana (RMSE / MAE / R²) | CV | Commit | Python / scikit-learn | Tamaño `.joblib` (MB) |
|---|---|---:|---:|---|---|---|---|---|---|---:|
| `v1` | `RandomForestRegressor`; `n_estimators=40`, `max_depth=20`, `min_samples_leaf=3`, `max_features=0.7` | 42 | 150826 / 37707 | 69170.89672101349 / 20316.502452683846 / 0.13963386499245978 | 74573.05090417106 / 29086.84720393411 / -7.329771751773961e-07 | 75702.6362182056 / 26547.89704829342 / -0.03052493054437 | 5 folds; RMSE medio 75119.69762383674, desviación 5217.000215458253; MAE medio 20467.86587797509; R² medio 0.11105930934552262 | `ab421783ca596277f2ee88eb990110c27c9a753d` | 3.11.9 / 1.9.1 | 26.347 |
| `v2` | `RandomForestRegressor`; `n_estimators=60`, `max_depth=20`, `min_samples_leaf=5`, `max_features=0.7` | 7 | 150826 / 37707 | 69019.22197302482 / 19873.000920595146 / 0.15244748135646324 | 74969.86904234429 / 29070.713164742847 / -1.0437808395380443e-08 | 76101.29940972055 / 26575.73999522635 / -0.030411375829898057 | no se corrió | `ab421783ca596277f2ee88eb990110c27c9a753d` | 3.11.9 / 1.9.1 | 28.717 |

## Comparabilidad

v1 usó semilla 42 y v2 semilla 7, por lo tanto sus conjuntos de prueba son
distintos y **NO se pueden comparar sus métricas como si fueran una mejora**.
Para comparar versiones hay que evaluarlas sobre los mismos casos reservados
(misma semilla y `test_size`). Los tamaños train/test iguales no implican que
sean las mismas filas.

## Cómo cambiar de versión

Configure `MODEL_VERSION` con el identificador de una versión disponible antes
de iniciar la API. Por ejemplo, en PowerShell use `$env:MODEL_VERSION = "v2"` y
reinicie Uvicorn; en Docker, pase `-e "MODEL_VERSION=v2"` al comando `docker
run`. Una versión entrenada con `/train` no se activa automáticamente.

## Cómo volver a la anterior

Configure `MODEL_VERSION` con el identificador de la versión anterior que aún
esté disponible en el directorio de modelos y reinicie la API. Por ejemplo, si
se está usando `v2` y `v1` sigue disponible, configure `MODEL_VERSION=v1`.
Consulte `/health` para ver las versiones disponibles.
