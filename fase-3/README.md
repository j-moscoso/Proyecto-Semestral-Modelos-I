# Regresión de precios de vehículos usados — Fase 3

**Modelos y Simulación de Sistemas I · Universidad de Antioquia**

**Integrantes:** Juan Manuel Moscoso Torres, Slleider David Rojas Aleman y Samuel Castaño Arenas.

Esta fase expone como API REST el problema de regresión del precio de autos usados de Kaggle Playground Series S4E9. La Fase 1 desarrolló el análisis y el modelo; la Fase 2 organizó el pipeline reproducible, su entrenamiento y persistencia versionada. La API de esta fase reutiliza el pipeline persistido de la Fase 2 para predecir y permite entrenar nuevas versiones.

## Contenido y autonomía

```text
fase-3/
├── api.py, cliente_prueba.py, train.py, predict.py
├── src/                 # Validación, pipeline, serving y versionamiento
├── data/                # train.csv y ejemplo_prediccion.csv
├── models/              # Versiones persistidas v1, v2 y v3
├── tests/               # Pruebas heredadas, API y referencia
└── docs/                # Contratos, versiones, Docker y evidencias
```

La carpeta es autónoma: contiene su propio código fuente, datos, modelos y dependencias; sus módulos no importan código ni leen rutas de otras fases.

## Entorno Python

Requiere Python 3.11.9. Ejecuta estos comandos desde `fase-3/` en PowerShell:

```powershell
python --version
python -m venv .\.venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Comprueba que ambas verificaciones de versión indiquen `Python 3.11.9`.

## Arrancar la API

Desde `fase-3/`, inicia Uvicorn con un solo worker:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

La API queda en `http://127.0.0.1:8000`; la documentación interactiva está en `http://127.0.0.1:8000/docs`.

## Versiones y configuración

`MODEL_VERSION` selecciona la versión activa para predicción y, si no se define, usa `v1`. Configura una versión existente antes de arrancar la API y reiníciala después de cambiarla. Por ejemplo:

```powershell
$env:MODEL_VERSION = "v2"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Variables disponibles:

| Variable | Uso | Valor predeterminado |
|---|---|---|
| `MODELS_DIR` | Carpeta que contiene las versiones de modelo. | `models` |
| `MODEL_VERSION` | Versión usada para predecir. | `v1` |
| `TRAIN_DATA` | CSV usado por `POST /train`. | `data/train.csv` |
| `GIT_COMMIT` | Identificador guardado en metadatos de entrenamiento. | Sin valor predeterminado especificado |

`LOG_PATH` aplica al registro de solicitudes de la Fase 4, no al API de esta fase. Si `/predict` solicita una versión inexistente, responde `503` con el código `modelo_no_disponible` y enumera las versiones disponibles. `/health` informa la configuración activa y las versiones encontradas.

## Ejemplos de la API

La siguiente solicitud usa los datos de ejemplo. Se omiten `id` y `price`, que no son necesarios para predecir:

```http
POST /predict
Content-Type: application/json
```

```json
{
  "brand": "Nissan",
  "model": "Murano SV",
  "model_year": 2022,
  "milage": 23677,
  "fuel_type": "Gasoline",
  "engine": "3.5L V6 24V MPFI DOHC",
  "transmission": "Automatic CVT",
  "ext_col": "Sunset Drift Chromaflair",
  "int_col": "Graphite",
  "accident": "None reported",
  "clean_title": null
}
```

Respuesta real de [cliente_predecir.txt](./docs/evidencia/cliente_predecir.txt):

```json
{
  "prediccion": 47061.047144440714,
  "version": "v1"
}
```

Si falta un campo obligatorio, la respuesta real es HTTP `422`. Por ejemplo, al omitir `brand`:

```json
{
  "error": "entrada_invalida",
  "mensaje": "falta el campo 'brand'"
}
```

Consulta el [contrato detallado de la API](./docs/CONTRATO_API.md) para los campos, tipos, estados y errores.

## Entrenamiento de versiones

`POST /train` entrena y persiste una nueva versión con el CSV configurado en `TRAIN_DATA`. El cuerpo es opcional; admite una versión explícita y los parámetros `n_estimators` y `seed`. La versión nueva **no se activa automáticamente**. Para activarla, configura `MODEL_VERSION` con su identificador y reinicia Uvicorn.

La ejecución registrada en [train_api.txt](./docs/evidencia/train_api.txt) creó `v3` y devolvió HTTP `201`:

```json
{
  "version": "v3",
  "activada": false,
  "metricas_test": {
    "rmse": 69170.89672101349,
    "mae": 20316.502452683846,
    "r2": 0.13963386499245978
  },
  "ruta": "models/v3",
  "mensaje": "Versión guardada. Para usarla configure MODEL_VERSION=v3 y reinicie la API."
}
```

## Cliente de prueba

El cliente lee `data/ejemplo_prediccion.csv`. El argumento global `--url` puede cambiar el servidor y debe ir antes del subcomando:

```powershell
.\.venv\Scripts\python.exe .\cliente_prueba.py health
.\.venv\Scripts\python.exe .\cliente_prueba.py predecir
.\.venv\Scripts\python.exe .\cliente_prueba.py predecir --fila 3
.\.venv\Scripts\python.exe .\cliente_prueba.py todos
.\.venv\Scripts\python.exe .\cliente_prueba.py faltante
.\.venv\Scripts\python.exe .\cliente_prueba.py entrenar
.\.venv\Scripts\python.exe .\cliente_prueba.py --url http://127.0.0.1:8000 health
```

`health` consulta el estado; `predecir` envía una fila (por defecto, la 1); `todos` envía las 12 filas; `faltante` omite `brand`; y `entrenar` solicita un entrenamiento con el cuerpo vacío.

## Docker

Los siguientes comandos se usan desde PowerShell en `fase-3/`. Construyen la imagen, publican el puerto `8000`, montan `data` como solo lectura y `models` con escritura, y ejecutan pytest en el contenedor:

```powershell
docker build -t modelos-fase3:1.0 .
docker run --rm -p 8000:8000 -e "MODEL_VERSION=v1" -e "GIT_COMMIT=$env:GIT_COMMIT" `
  -v "$(Resolve-Path .\data):/app/data:ro" `
  -v "$(Resolve-Path .\models):/app/models" `
  modelos-fase3:1.0
docker run --rm `
  -v "$(Resolve-Path .\data):/app/data:ro" `
  -v "$(Resolve-Path .\models):/app/models:ro" `
  modelos-fase3:1.0 python -m pytest -q
```

El montaje de `models` apunta a la carpeta anfitriona donde se guardan las versiones entrenadas; debe ser escribible para que `/train` pueda persistirlas. Para predecir, el montaje puede ser de solo lectura. La evidencia de construcción y pruebas está en [docker_build.txt](./docs/evidencia/docker_build.txt) y [docker_pytest.txt](./docs/evidencia/docker_pytest.txt). Consulta los [comandos Docker](./docs/DOCKER_COMANDOS.md). Esta guía y sus evidencias se trabajaron con PowerShell.

## Pruebas

Ejecuta la suite desde `fase-3/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

| Prueba de `tests/test_api.py` | Comprueba |
|---|---|
| `test_health_responde_200` | Estado HTTP, estado del servicio y versiones disponibles. |
| `test_predict_valido_devuelve_prediccion_y_version` | Predicción finita y versión devuelta. |
| `test_predict_campo_obligatorio_ausente_se_rechaza` | Rechazo 422 e identificación del campo faltante. |
| `test_predict_acepta_nulos` | Aceptación de nulos en los once campos. |
| `test_predict_tipo_incompatible` | Rechazo 422 de tipos incompatibles en campos numérico y de texto. |
| `test_predict_ignora_price_e_id` | Aceptación de columnas extra `price` e `id`. |
| `test_predict_modelo_no_disponible` | Respuesta 503 e indicación de versiones disponibles. |
| `test_predict_coincide_con_pipeline_directo` | Equivalencia de la API con el pipeline antes de persistirlo. |
| `test_train_crea_version_nueva_sin_tocar_v1` | Creación de `v2`, sin activarla ni alterar `v1`. |
| `test_prediccion_coincide_con_referencia_v1` | Regresión de las doce predicciones contra la referencia de la Fase 2. |

La [evidencia de pytest local](./docs/evidencia/pytest_local.txt) registra 19 pruebas aprobadas.

## Versiones

RMSE y MAE se expresan en USD. La tabla resume los parámetros principales y métricas `test` de los JSON de cada versión.

| Versión | Semilla | `n_estimators` | `min_samples_leaf` | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|---:|
| `v1` | 42 | 40 | 3 | 69,170.897 | 20,316.502 | 0.139634 |
| `v2` | 7 | 60 | 5 | 69,019.222 | 19,873.001 | 0.152447 |
| `v3` | 42 | 40 | 3 | 69,170.897 | 20,316.502 | 0.139634 |

`v1` y `v2` usaron semillas distintas, por lo que sus conjuntos de prueba no son los mismos y sus métricas no se deben interpretar directamente como una comparación de mejora. Los detalles están en [VERSIONES.md](./docs/VERSIONES.md).

## Limitaciones conocidas

- Los datos de la competencia son sintéticos y no garantizan predicciones representativas del mercado real.
- El entrenamiento de `/train` es síncrono y bloquea la petición hasta finalizar.
- Uvicorn debe ejecutarse con un solo worker.
- La API no implementa autenticación.
- La primera predicción puede tardar más porque carga el modelo.

## Evidencia

- [Health](./docs/evidencia/health.txt)
- [Predicción de una fila](./docs/evidencia/cliente_predecir.txt)
- [Predicción de todas las filas](./docs/evidencia/cliente_todos.txt)
- [Solicitud con campo faltante](./docs/evidencia/cliente_faltante.txt)
- [Entrenamiento por API](./docs/evidencia/train_api.txt)
- [Versiones después del entrenamiento](./docs/evidencia/models_despues_de_train.txt)
- [Pruebas locales](./docs/evidencia/pytest_local.txt)
- [Construcción Docker](./docs/evidencia/docker_build.txt)
- [Pruebas Docker](./docs/evidencia/docker_pytest.txt)

## Git Flow

`main` es la rama estable y `develop` la de integración. El trabajo se organiza en ramas `feature/*`, `fix/*`, `docs/*` y `chore/*`; los cambios se integran mediante Pull Request con revisión.

La entrega 3 se identifica con la etiqueta de Git `entrega-3`.

## Fuentes consultadas y código adaptado

- Datos del problema: [Kaggle — Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/data).
- El pipeline de preparación, modelado y persistencia se adaptó del núcleo desarrollado en la Fase 2; esta fase lo carga desde sus artefactos versionados.
- La capa HTTP se implementó con FastAPI. Contratos y configuración: [CONTRATO_TECNICO.md](./docs/CONTRATO_TECNICO.md), [CONTRATO_API.md](./docs/CONTRATO_API.md) y [requirements.txt](./requirements.txt).
