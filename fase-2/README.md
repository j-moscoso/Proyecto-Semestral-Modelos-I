# Regresión de precios de vehículos usados — Fase 2

**Modelos y Simulación de Sistemas I · Universidad de Antioquia**

**Integrantes:** Juan Manuel Moscoso Torres, Slleider David Rojas Aleman y Samuel Castaño Arenas.

El proyecto estima el precio (`price`) de vehículos usados a partir de once variables descriptivas. Esta fase convierte el flujo explorado en la Fase 1 en una aplicación reproducible para entrenar, versionar y ejecutar predicciones; amplía la validación y documenta sus límites.

## Correcciones respecto a la entrega anterior

- Se corrigió el tratamiento de valores faltantes en `transmission` para que las predicciones no fallen.
- Se normalizaron las rutas de PowerShell de la documentación.
- Se aclaró que Playground Series S4E9 es sintético, generado a partir del conjunto real *Used Car Price Prediction Dataset*; sus 188.533 filas no son registros directos del mercado.
- Se documentó que el EDA de la Fase 1 consultó todo el conjunto, incluido `price`, aunque los transformadores y el modelo se ajustaron solo con `train`.
- El notebook de la Fase 1 se guardó con las salidas de su ejecución.

## Entorno

Requiere Python 3.11.9. Verifica que `python --version` indique esa versión. Ejecuta estos comandos desde `fase-2/` en PowerShell:

```powershell
python -m venv .\.venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Las versiones instaladas están fijadas en [requirements.txt](./requirements.txt).

## Datos

El CSV de entrenamiento se ubica en `.\data\train.csv` y corresponde a los datos de la competición [Regression of Used Car Prices — Kaggle](https://www.kaggle.com/competitions/playground-series-s4e9/data). El CSV de predicción debe incluir las once columnas de entrada; `id` es opcional, se excluye de los predictores y, si está presente, se conserva en la salida. Se aceptan nulos, pero las columnas numéricas `model_year` y `milage` deben tener valores convertibles a número. Consulta el [contrato de entrada](./docs/CONTRATO_ENTRADA.md) para el esquema, validaciones y formato de salida.

## Entrenamiento y predicción

Los siguientes comandos se ejecutan desde `fase-2/`. El entrenamiento indicado con `v-readme` es de verificación; esa versión temporal se elimina después de comprobar el comando.

```powershell
.\.venv\Scripts\python.exe .\train.py --help
.\.venv\Scripts\python.exe .\predict.py --help
.\.venv\Scripts\python.exe .\train.py --data .\data\train.csv --target price --output-dir .\models --version v-readme --n-estimators 5 --test-size 0.2
.\.venv\Scripts\python.exe .\predict.py --model .\models\v2 --input .\data\ejemplo_prediccion.csv --output .\outputs\predicciones-readme.csv
```

Cada versión se guarda en `.\models\<version>\` e incluye `modelo.joblib`, `metrics.json` y `metadata.json`. El CSV de predicciones se escribe en la ruta indicada, por ejemplo `.\outputs\predicciones-readme.csv`.

## Docker

Los comandos de Docker se ejecutan en PowerShell desde `fase-2/`; se construye la imagen, se monta `data` en modo lectura y `models` con permisos de escritura al entrenar. Para predecir, `models` también se monta como solo lectura y `outputs` como escritura. La variable `GIT_COMMIT` se pasa al contenedor porque la imagen no contiene `.git`. Comandos completos en una línea cada uno: [guía Docker](./docs/DOCKER_COMANDOS.md).

```powershell
docker build -t modelos-fase2:1.0 .
docker run --rm -e "GIT_COMMIT=$env:GIT_COMMIT" -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data /app/data/train.csv --output-dir /app/models --version v-docker
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model /app/models/v-docker --input /app/data/ejemplo_prediccion.csv --output /app/outputs/predicciones.csv
```

## Pruebas

Ejecuta la suite desde `fase-2/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

| Prueba | Comprueba |
|---|---|
| `test_entrenamiento_y_almacenamiento` | Entrenamiento y persistencia del modelo, métricas y metadatos. |
| `test_carga_y_prediccion` | Carga equivalente del pipeline guardado, predicciones y conservación del orden e identificadores. |
| `test_valores_faltantes` | Predicción sin errores ni resultados NaN ante campos vacíos, incluido `transmission`. |
| `test_entrada_invalida` | Rechazo de columnas requeridas ausentes y valores numéricos inválidos, sin crear salida. |
| `test_versionamiento` | Persistencia de dos versiones y rechazo de sobrescribir una versión existente. |
| `test_categoria_desconocida` | Predicción con categorías no vistas durante el entrenamiento. |
| `test_columnas_reordenadas_y_extra` | Reordenamiento de predictores e ignorado de columnas adicionales. |
| `test_help` | Ayuda CLI de `train.py` y `predict.py` con salida exitosa. |

## Métricas y limitaciones

Resultados de test tomados de los archivos `metrics.json` de cada versión. RMSE y MAE se expresan en USD; valores menores son mejores. R² se muestra sin unidad.

| Versión | Modelo | RMSE (USD) | MAE (USD) | R² |
|---|---|---:|---:|---:|
| v1 | Random Forest | 69,170.897 | 20,316.502 | 0.139634 |
| v1 | Baseline media | 74,573.051 | 29,086.847 | -0.000001 |
| v1 | Baseline mediana | 75,702.636 | 26,547.897 | -0.030525 |
| v2 | Random Forest | 69,019.222 | 19,873.001 | 0.152447 |
| v2 | Baseline media | 74,969.869 | 29,070.713 | -0.00000001 |
| v2 | Baseline mediana | 76,101.299 | 26,575.740 | -0.030411 |

Playground Series S4E9 es sintético y se generó a partir del conjunto real *Used Car Price Prediction Dataset*; sus 188.533 filas no son observaciones directas del mercado. Además, la división aleatoria puede situar anuncios iguales o similares en entrenamiento y prueba; faltan identificadores confiables para agrupar vehículos. El desempeño no garantiza resultados sobre vehículos nuevos ni uso comercial. Más detalles en [Datos y limitaciones](./docs/DATOS_Y_LIMITACIONES.md).

## Evidencia

- [Entrenamiento v1](./docs/evidencia/train_v1.txt) y [entrenamiento v2](./docs/evidencia/train_v2.txt).
- [Predicción exitosa](./docs/evidencia/predict_ok.txt) y [predicción con error](./docs/evidencia/predict_error.txt).
- [Pytest local](./docs/evidencia/pytest.txt).
- Docker: [build](./docs/evidencia/docker_build.txt), [entrenamiento](./docs/evidencia/docker_train.txt), [predicción](./docs/evidencia/docker_predict.txt), [entrada inválida](./docs/evidencia/docker_error.txt) y [pytest](./docs/evidencia/docker_pytest.txt).

## Git Flow

Se trabaja sobre `main` (estable) y `develop` (integración), con ramas `feature/*`, `fix/*`, `docs/*` y `chore/*` para los cambios. Los cambios se integran mediante Pull Request hacia la rama correspondiente, con revisión antes de aceptar la integración.

La entrega está identificada con la etiqueta de Git `entrega-2`.

## Fuentes consultadas y código adaptado

- Datos de la competición: [Kaggle — Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/data).
- Contratos y decisiones del proyecto: [contrato técnico](./docs/CONTRATO_TECNICO.md), [contrato de entrada](./docs/CONTRATO_ENTRADA.md) y [datos y limitaciones](./docs/DATOS_Y_LIMITACIONES.md).
- Se adaptaron para esta fase el flujo de preparación de variables y el pipeline de modelado de la fase anterior, incorporando el manejo de valores faltantes y la persistencia versionada.
