# Contrato técnico – Fases 3 y 4 (fuente de verdad)

## Principios
- Python 3.11.9 y FastAPI. Todo se ejecuta DESDE la carpeta de la fase (fase-3/ o fase-4/). Imports: `from src.modulo import ...`.
- fase-3/ y fase-4/ son autónomas: cada una trae su copia de src/, train.py, predict.py, models/v1, models/v2 y data/. fase-1/ y fase-2/ NO se modifican. Ningún archivo de fase-3/ o fase-4/ importa ni lee rutas de otra carpeta.
- La API NO reimplementa lógica del modelo: reutiliza src.validation.validate_and_prepare_input, src.versioning.load_model y train.run_training. Las funciones del pipeline viven solo en src/features.py.
- Ningún módulo entrena, lee archivos ni imprime al importarse.
- Configuración por variables de entorno, leídas en CADA llamada (nunca al importar):
  MODELS_DIR (default "models"), MODEL_VERSION (default "v1"), TRAIN_DATA (default "data/train.csv"), LOG_PATH (default "logs/predicciones.csv", solo fase-4), GIT_COMMIT (ya existente).
- Un solo worker de uvicorn. Rutas de las funciones de la API con `def` (no `async def`), porque el modelo bloquea.
- Errores previsibles = excepciones AppError. La API responde siempre JSON {"error": "<codigo>", "mensaje": "<texto claro en español>"}.
- Sin rutas absolutas, sin input() interactivo. Docstrings y comentarios en español.

## API (api.py)
app = FastAPI(title="API de precios de vehículos usados", version="3.0" en fase-3 y "4.0" en fase-4).

GET /health -> 200
  {"estado": "ok", "version_activa": "<MODEL_VERSION>", "versiones_disponibles": ["v1", "v2"]}
  Nunca falla por un modelo ausente (es solo un chequeo de que el servicio corre).

POST /predict -> 200
  Cuerpo: los 11 campos de entrada, TODOS obligatorios (la clave debe venir), cada uno puede ser null.
  Respuesta: {"prediccion": <float>, "version": "<version usada>"}
  Reglas: price e id NO son necesarios; campos extra se ignoran. model_year y milage son números (una cadena numérica como "2020" se acepta y se convierte; texto no numérico es error). Los demás son texto (un número en un campo de texto es error).
  Errores:
    422 {"error": "entrada_invalida", "mensaje": "falta el campo 'brand'; campo 'model_year': ..."}  (falta de clave, tipo incompatible, JSON mal formado, o InputValidationError)
    503 {"error": "modelo_no_disponible", "mensaje": "... Versiones disponibles: v1, v2. Configure MODEL_VERSION con una de ellas."}
    500 {"error": "error_interno", "mensaje": "Error interno del servidor."}

POST /train -> 201
  Cuerpo OPCIONAL (puede venir vacío o {}): {"version": str|null, "n_estimators": int entre 1 y 500 (default 40), "seed": int (default 42)}.
  Usa TRAIN_DATA, target "price", test_size 0.2, max_depth 20, min_samples_leaf 3, max_features 0.7, cv_folds 0 (la configuración de la Fase 2).
  Si no viene version, genera el siguiente "vN" libre (v1, v2 -> v3). NUNCA sobrescribe y NUNCA cambia la versión activa.
  Respuesta: {"version": "v3", "activada": false, "metricas_test": {"rmse": ..., "mae": ..., "r2": ...}, "ruta": "models/v3", "mensaje": "Versión guardada. Para usarla configure MODEL_VERSION=v3 y reinicie la API."}
  Errores: 409 {"error": "version_existente"} · 409 {"error": "entrenamiento_en_curso"} · 422 {"error": "entrada_invalida"} · 500 {"error": "error_interno"}

Mapeo de excepciones: InputValidationError -> 422; VersionExistsError -> 409; ModelNotFoundError -> 503; TrainingInProgressError -> 409; cualquier otra -> 500.

## src/api_schema.py
VehiculoEntrada(BaseModel): model_config = ConfigDict(extra="ignore", protected_namespaces=()).
  brand, model, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title: `str | None` SIN valor por defecto.
  model_year, milage: `float | None` SIN valor por defecto.
  El orden y los nombres son los de src.schema.INPUT_COLUMNS.
PrediccionRespuesta(BaseModel): prediccion: float; version: str.
EntrenarSolicitud(BaseModel): version: str | None = None; n_estimators: int = Field(40, ge=1, le=500); seed: int = 42.

## src/serving.py
class TrainingInProgressError(AppError)
get_models_dir()->Path ; get_active_version()->str ; get_train_data_path()->Path   (leen las variables de entorno con sus defaults)
list_versions(models_dir=None)->list[str] : nombres de carpetas que contienen modelo.joblib, sin los que terminan en .tmp, ordenados.
load_active_model(version=None)->(pipeline, metadata, version) : version por defecto = get_active_version(); valida el nombre con ^[A-Za-z0-9._-]+$ (si no, ModelNotFoundError); usa src.versioning.load_model; cachea por ruta con un Lock; tras cargar pone n_jobs=1 en el paso "model" si existe. Si no existe -> ModelNotFoundError con la lista de versiones disponibles en el mensaje.
clear_model_cache()->None
predict_record(record: dict)->(float, str) : DataFrame de una fila -> validate_and_prepare_input(df, expected_columns=schema["columns"], numeric_columns=schema["numeric"]) (si el modelo no trae input_schema, sin esos argumentos) -> pipeline.predict -> (float, version).
next_version_id(models_dir=None)->str : "v"+(máximo N de las carpetas vN + 1); "v1" si no hay; ignora nombres que no sean exactamente v<entero>.
train_new_version(version=None, n_estimators=40, seed=42)->dict : candado global NO bloqueante (si está ocupado, TrainingInProgressError); arma argparse.Namespace con los argumentos de train.run_training; importa `train` DENTRO de la función; llama train.run_training; lee models/<version>/metrics.json y devuelve {"version","metricas_test","ruta"}.

## Fase 4: registro y monitoreo
Formato del registro (CSV, UTF-8, encabezado escrito una sola vez al crear el archivo):
  fecha_hora,estado_http,version,prediccion
  fecha_hora = UTC, formato YYYY-MM-DDTHH:MM:SS ; estado_http = código HTTP devuelto ; version = versión que produjo la respuesta (vacío si no se usó modelo) ; prediccion = valor redondeado a 2 decimales (vacío si hubo error).
Regla: UNA fila por cada POST a /predict, incluidos los rechazados por validación (422), 503 y 500. /health y /train NO se registran. Se abre en modo agregar (a), así que el historial se conserva entre ejecuciones.

src/monitoring.py
LOG_COLUMNS = ["fecha_hora","estado_http","version","prediccion"]
class RegistroInvalidoError(AppError)
get_log_path()->Path (LOG_PATH o "logs/predicciones.csv")
registrar_solicitud(estado_http:int, version:str|None, prediccion:float|None, ruta=None)->None : Lock, crea la carpeta padre, escribe encabezado si el archivo no existe o está vacío, nunca lanza excepción (si falla escribir, imprime un aviso por stderr).
leer_registro(ruta=None)->list[dict] : archivo inexistente o vacío -> []; encabezado distinto de LOG_COLUMNS o estado_http no entero -> RegistroInvalidoError.
calcular_metricas(filas)->{"total": int, "fallidas": int, "porcentaje": float|None} : fallida = 400 <= estado_http <= 599; porcentaje None si total == 0.

monitor.py (CLI): --log (default get_log_path()). main(argv=None)->int. Imprime EXACTAMENTE:
  Total de solicitudes: 3
  Solicitudes fallidas: 1
  Porcentaje de fallos: 33,3%
y con registro vacío o inexistente:
  Total de solicitudes: 0
  Solicitudes fallidas: 0
  Porcentaje de fallos: sin solicitudes
(el porcentaje con 1 decimal y coma decimal). Códigos de salida: 0 ok, 1 RegistroInvalidoError (mensaje "ERROR: ..." por stderr).

Integración en api.py (fase-4): middleware HTTP que, SOLO para POST /predict, registra después de obtener la respuesta usando response.status_code y request.state.version / request.state.prediccion (las fija la ruta cuando la predicción tuvo éxito; si no existen, se registran vacías). Si la ruta lanza una excepción no controlada, el middleware registra 500 y devuelve el JSON de error_interno.

## Cliente (cliente_prueba.py, solo biblioteca estándar)
Subcomandos: health | predecir [--fila N] | todos | faltante | entrenar | demo (demo solo en fase-4). Argumento global --url (default http://localhost:8000). Lee data/ejemplo_prediccion.csv, convierte vacíos a null y model_year/milage a números. Imprime código HTTP y cuerpo JSON. Código de salida 0 si el servidor respondió (aunque sea 4xx/5xx) y 1 si no hubo conexión.
  predecir -> envía la fila N (default 1) | todos -> envía las 12 filas en orden e imprime id, prediccion, version | faltante -> envía la fila 1 sin la clave "brand" | entrenar -> POST /train con cuerpo vacío (timeout 900 s) | demo -> 2 solicitudes válidas + 1 con la clave "brand" ausente.

## Docker
Imágenes: modelos-fase3:1.0 y modelos-fase4:1.0. Puerto 8000. CMD: uvicorn api:app --host 0.0.0.0 --port 8000. Montajes: data (solo lectura) en /app/data, models (escritura, porque /train guarda ahí) en /app/models y, en fase-4, logs en /app/logs. Variables: MODEL_VERSION, GIT_COMMIT.

## Pruebas
Fase-3 conserva los tests de la Fase 2 y agrega tests/test_api.py. Fase-4 agrega tests/test_monitoreo.py. Fixtures nuevas en tests/conftest.py: registro_valido, api_client (fija MODELS_DIR, MODEL_VERSION, TRAIN_DATA y LOG_PATH a rutas temporales con monkeypatch y limpia la caché).

## Git
Ramas: main (estable), develop (integración), feature/*, fix/*, docs/*, chore/*. Mensajes de commit: `feat(api): ...`, `test(api): ...`, `docs(readme): ...`, `chore(fase-4): ...`. Etiqueta final: entrega-3.
