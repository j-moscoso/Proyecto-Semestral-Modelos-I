# Regresión de precios de vehículos usados — Fase 4

**Modelos y Simulación de Sistemas I · Universidad de Antioquia**

**Integrantes:** Juan Manuel Moscoso Torres, Slleider David Rojas Aleman y Samuel Castaño Arenas.

El proyecto estima precios de vehículos usados con datos de la competencia Kaggle Playground Series S4E9. La Fase 1 desarrolló el análisis y el modelo; la Fase 2 organizó el pipeline reproducible, su entrenamiento y el versionamiento; la Fase 3 expuso predicción y entrenamiento mediante una API REST. Esta fase agrega el registro persistente de intentos a `POST /predict` y el comando `monitor.py` para resumir los códigos HTTP del servicio.

## Contenido y autonomía

```text
fase-4/
├── api.py, cliente_prueba.py, monitor.py, train.py, predict.py
├── src/                 # API, validación, serving, pipeline y monitoreo
├── data/                # Datos de entrenamiento y ejemplos de solicitudes
├── models/              # Artefactos de modelos v1 y v2
├── tests/               # Pruebas heredadas, API y monitoreo
└── docs/                # Contratos, Docker, versiones y evidencia
```

La carpeta es autónoma: contiene sus propias copias de código, datos, modelos, pruebas y configuración, y puede ejecutarse desde su raíz sin importar archivos ni leer rutas de otras fases.

## Python y entorno

Se requiere Python 3.11.9. Ejecuta los comandos desde `fase-4/` en PowerShell. Si el entorno virtual aún no existe, créalo con el lanzador de Python 3.11:

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.11 -m venv .\.venv
}
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

La versión debe indicar `Python 3.11.9`. El entorno usa las dependencias fijadas en `requirements.txt`.

## Arrancar la API con registro

Para una prueba local desechable, asigna el registro a un archivo temporal fuera del repositorio. Abre una terminal PowerShell en `fase-4/` y ejecuta:

```powershell
$env:MODEL_VERSION = "v1"
$env:LOG_PATH = Join-Path $env:TEMP "fase4-predicciones.csv"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Deja esa terminal abierta mientras haces las solicitudes y detenla con `Ctrl+C`. Para conservar el registro localmente entre ejecuciones, configura `$env:LOG_PATH = ".\logs\predicciones.csv"` en vez de la ruta temporal. Si no se define `LOG_PATH`, el valor predeterminado es `logs/predicciones.csv`, relativo a la carpeta desde la que se inicia la API.

En Docker, desde `fase-4/`, publica el puerto y monta datos, modelos y registros:

```powershell
docker build -t modelos-fase4:1.0 .
docker run --rm -p 8000:8000 -e "MODEL_VERSION=v1" -e "LOG_PATH=/app/logs/predicciones.csv" `
  -v "$(Resolve-Path .\data):/app/data:ro" `
  -v "$(Resolve-Path .\models):/app/models" `
  -v "$(Resolve-Path .\logs):/app/logs" `
  modelos-fase4:1.0
```

La ruta persistente dentro del contenedor es `/app/logs/predicciones.csv`; el montaje conserva el registro en la carpeta local `logs/`. `data/` queda montada en solo lectura y `models/` con escritura para permitir el entrenamiento por API.

## Formato y política del registro

El archivo CSV usa UTF-8, agrega filas y escribe su encabezado una sola vez. Las columnas son:

| Columna | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora UTC con formato `YYYY-MM-DDTHH:MM:SS`. |
| `estado_http` | Código HTTP devuelto por la solicitud. |
| `version` | Versión que produjo la predicción; vacía si no se usó un modelo. |
| `prediccion` | Predicción redondeada a dos decimales; vacía si hubo error. |

Se registra **una fila por cada `POST /predict`, incluidos los rechazos** de validación y los errores `4xx` o `5xx`. `GET /health` y `POST /train` no se registran.

## Monitor

Con la API en ejecución, desde otra terminal PowerShell en `fase-4/`, envía la demostración y consulta el archivo temporal elegido al arrancarla:

```powershell
.\.venv\Scripts\python.exe .\cliente_prueba.py demo
.\.venv\Scripts\python.exe .\monitor.py --log (Join-Path $env:TEMP "fase4-predicciones.csv")
```

El cliente envía dos predicciones válidas y una solicitud con `brand` ausente. El monitor imprime el total, cuántas solicitudes fallaron (códigos HTTP entre 400 y 599) y el porcentaje de fallos con una cifra decimal y coma. Para resumir el registro persistente local predeterminado, ejecuta:

```powershell
.\.venv\Scripts\python.exe .\monitor.py
```

En Docker, con un archivo existente en el montaje `logs/`, puedes ejecutar el monitor sin iniciar el servidor:

```powershell
docker run --rm -v "$(Resolve-Path .\logs):/app/logs" modelos-fase4:1.0 `
  python monitor.py --log /app/logs/predicciones.csv
```

## Reporte

La prueba envió dos solicitudes válidas con `v1` y una solicitud sin la clave `brand`. Los indicadores registrados en [monitor_salida.txt](./docs/evidencia/monitor_salida.txt) son:

```text
Total de solicitudes: 3
Solicitudes fallidas: 1
Porcentaje de fallos: 33,3%
```

La API procesó dos registros con `v1` y rechazó el tercero porque faltaba el campo `brand`. La tasa de fallos fue 1/3 (33,3%). El rechazo era el resultado esperado de la prueba de validación.

El porcentaje describe el funcionamiento del servicio y contabiliza las respuestas fallidas, no la calidad predictiva del modelo; medir esa calidad en vehículos reales requeriría conocer sus precios reales.

La evidencia de las respuestas y sus filas de registro está en [muestra_predicciones.csv](./docs/evidencia/muestra_predicciones.csv):

```csv
fecha_hora,estado_http,version,prediccion
2026-11-21T04:15:04,200,v1,47061.05
2026-11-21T04:15:04,200,v1,18202.71
2026-11-21T04:15:04,422,,
```

## Versiones y cambio de modelo

La carpeta de esta fase contiene los identificadores `v1` y `v2`. La evidencia de [health.txt](./docs/evidencia/health.txt) muestra que la API usa `v1`, la versión predeterminada de `MODEL_VERSION`; [rollback_v2.txt](./docs/evidencia/rollback_v2.txt) documenta una ejecución seleccionada explícitamente con `v2` y su predicción. Para cambiar, configura `MODEL_VERSION` con la versión deseada antes de iniciar la API y reinicia Uvicorn. Para volver a `v1`, configura `$env:MODEL_VERSION = "v1"` y reinicia. El entrenamiento por `POST /train` en la Fase 3 creó además una versión `v3`; su disponibilidad en la carpeta autónoma de esta fase no está afirmada aquí.

Consulta los parámetros y métricas en [VERSIONES.md](./docs/VERSIONES.md) y el cambio evidenciado en [rollback_v2.txt](./docs/evidencia/rollback_v2.txt). `v1` y `v2` usaron semillas distintas, así que sus conjuntos de prueba no son los mismos y sus métricas no se pueden comparar como si una versión hubiera mejorado respecto de la otra.

## Política de reentrenamiento

Se revisará el modelo cuando el equipo reúna 1.000 vehículos nuevos con su precio real conocido, o si el monitor muestra fallos `4xx`/`5xx` sostenidos que no se expliquen por datos mal enviados; un campo ausente se corrige en quien envía los datos, no reentrenando. Juan Manuel ejecutará `POST /train` con el CSV ampliado, la misma semilla y el mismo `test_size`, y generará una versión candidata. La candidata se comparará con la versión activa sobre los mismos casos reservados usando RMSE y MAE; se activará mediante `MODEL_VERSION` y reinicio solo si reduce RMSE sin empeorar MAE; de lo contrario, se conserva `v1`. Hoy no hay resultados con precios reales porque el conjunto disponible es sintético: esta política es una regla documentada, no una automatización.

## Pruebas

Ejecuta la suite desde `fase-4/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

La evidencia de [pytest_local.txt](./docs/evidencia/pytest_local.txt) registra las pruebas de API, aplicación y monitoreo. Las pruebas de monitoreo son:

| Prueba | Comprueba |
|---|---|
| `test_registro_tres_intentos_tres_filas` | Dos predicciones exitosas y un rechazo por campo faltante producen tres filas y un solo encabezado. |
| `test_registro_se_conserva_entre_ejecuciones` | Un segundo `TestClient` agrega una fila sin borrar el historial ni repetir el encabezado. |
| `test_registro_503_modelo_no_disponible` | El error `503` queda registrado sin versión ni predicción. |
| `test_registro_json_invalido` | El JSON inválido recibe `422` y queda registrado. |
| `test_health_y_train_no_se_registran` | Las rutas de salud y entrenamiento no generan filas. |
| `test_monitor_calcula_total_y_porcentaje` | El monitor calcula totales y porcentaje para el registro de la demostración. |
| `test_monitor_sin_solicitudes` | Un archivo ausente o vacío se reporta sin solicitudes. |
| `test_monitor_registro_invalido` | El encabezado inválido produce código de salida `1` y un mensaje `ERROR`. |

## Evidencia y Docker

Los archivos de evidencia están en [`docs/evidencia/`](./docs/evidencia/):

- [demo_solicitudes.txt](./docs/evidencia/demo_solicitudes.txt): respuestas de la demostración del cliente.
- [health.txt](./docs/evidencia/health.txt): respuesta de salud con `v1` activa.
- [muestra_predicciones.csv](./docs/evidencia/muestra_predicciones.csv): tres filas registradas.
- [monitor_salida.txt](./docs/evidencia/monitor_salida.txt): resumen de las solicitudes.
- [registro_contenido.txt](./docs/evidencia/registro_contenido.txt): contenido de un registro.
- [rollback_v2.txt](./docs/evidencia/rollback_v2.txt): salud y predicción usando `v2`.
- [pytest_local.txt](./docs/evidencia/pytest_local.txt): resultados de pytest local.
- [docker_build.txt](./docs/evidencia/docker_build.txt): evidencia de construcción de la imagen.
- [docker_pytest.txt](./docs/evidencia/docker_pytest.txt): pruebas dentro del contenedor.

Los comandos completos de construcción, ejecución, montaje del registro y monitor están en [DOCKER_COMANDOS.md](./docs/DOCKER_COMANDOS.md). La imagen expone el servicio en el puerto `8000`; monta `data/` como solo lectura, `models/` con escritura para entrenamiento y `logs/` para conservar los registros.

## Limitaciones

- El registro es un CSV local; no tiene rotación ni retención automática.
- La API debe ejecutarse con un solo worker para respetar la configuración prevista.
- El monitor cuenta estados HTTP y no mide calidad predictiva.
- La API no implementa autenticación.
- El candado de `POST /train` coordina el entrenamiento dentro de un proceso; no sincroniza procesos distintos.

## Git Flow

`main` es la rama estable y `develop` la de integración. El trabajo se organiza en ramas `feature/*`, `fix/*`, `docs/*` y `chore/*`; los cambios se integran mediante Pull Request con revisión.

La entrega 3 (Fases 3 y 4) está identificada con la etiqueta de Git `entrega-3`.

## Fuentes consultadas

- [Contrato técnico de las Fases 3 y 4](./docs/CONTRATO_TECNICO.md).
- [Contrato de la API](./docs/CONTRATO_API.md).
- [Versiones y métricas del modelo](./docs/VERSIONES.md).
- [Comandos Docker](./docs/DOCKER_COMANDOS.md).
- [Kaggle — Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/data).
- [Kaggle — Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
