# Contrato técnico – Fase 2 (fuente de verdad)

## Principios

- Python 3.11.9. Todo se ejecuta DESDE la carpeta fase-2/. Imports: `from src.modulo import ...`.
- Ningún módulo entrena, lee archivos ni imprime al importarse. Los scripts usan `if __name__ == "__main__":`.
- Toda función usada por el pipeline guardado (normalize_missing, extract_*, etc.) vive en src/features.py. NINGUNA se define en train.py ni predict.py (se pickelearía como __main__ y predict.py no podría cargarla).
- Errores previsibles = excepciones de src/schema.py. Los scripts las capturan, escriben `ERROR: <mensaje>` en stderr y retornan código 1. Nunca dejan un archivo de resultados a medias.
- Sin rutas absolutas, sin input() interactivo. Docstrings y comentarios en español.
- fase-1/ NO se modifica salvo en la rama docs/fase1-feedback (documentación y salidas del notebook).

## Esquema de datos

- Objetivo: `price`. Identificador: `id` (no es predictor; en predicción es opcional y, si viene, se copia a la salida).
- 11 columnas de entrada, en este orden: brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title.
- Numéricas: model_year, milage (deben ser convertibles a número). El resto es texto.
- Nulos: permitidos en cualquier celda de las 11 columnas; la COLUMNA debe existir.
- Columnas extra (incluidos `id` y `price`): se ignoran con advertencia, no es error.
- Categorías no vistas en entrenamiento: no es error (OneHot handle_unknown="ignore"; OrdinalEncoder -> -1).

## src/schema.py

TARGET_COLUMN="price"; ID_COLUMN="id"; NUMERIC_INPUT=["model_year","milage"];
INPUT_COLUMNS = las 11 en el orden de arriba; CATEGORICAL_INPUT = INPUT_COLUMNS sin las numéricas.
Excepciones: AppError(Exception); InputValidationError(AppError); VersionExistsError(AppError); ModelNotFoundError(AppError).

## src/features.py  (copiado de fase-1/features.py + correcciones)

normalize_missing(df)->df : copia; en categóricas reemplaza pd.NA/None/""/solo espacios por np.nan, deja dtype object y convierte a str los valores no nulos; en numéricas hace pd.to_numeric(errors="coerce") a float64. Solo procesa las columnas que existan.
extract_engine_features(df)->df ; extract_transmission_features(df)->df ; build_preprocessor()->ColumnTransformer
Las columnas extraídas son float64 con np.nan; transmission_type es object con np.nan. NUNCA pd.NA ni dtype "string" en la salida.

## src/data.py

load_training_data(path, target)->DataFrame  (archivo inexistente, ilegible o vacío, falta el target o falta alguna de las 11 columnas -> InputValidationError con mensaje claro)
split_data(df, target, test_size, seed)->(X_train, X_test, y_train, y_test)
  X = df.drop(columns=[target, "id"], errors="ignore"), y = df[target]; usa train_test_split(X, y, test_size=test_size, random_state=seed) igual que la Fase 1.

## src/modeling.py

build_pipeline(seed, n_estimators=40, max_depth=20, min_samples_leaf=3, max_features=0.7)->Pipeline
  pasos con estos nombres: "normalize" (FunctionTransformer(normalize_missing, validate=False)), "engine_features", "transmission_features", "preprocessor", "model" (RandomForestRegressor con n_jobs=-1, random_state=seed)
build_baseline(strategy)->DummyRegressor ; compute_metrics(y_true, y_pred)->{"rmse","mae","r2"} (floats de Python; rmse = raíz de mean_squared_error)

## src/versioning.py

resolve_version_dir(output_dir, version)->Path : valida el nombre con ^[A-Za-z0-9._-]+$ (si no, InputValidationError); si <output_dir>/<version> ya existe lanza VersionExistsError; NO crea nada; devuelve la ruta final.
save_version(version_dir, pipeline, metrics, metadata) : borra <version_dir>.tmp si quedó de un fallo anterior, escribe modelo.joblib (compress=3), metrics.json y metadata.json (indent=2, ensure_ascii=False) dentro de <version_dir>.tmp (creando carpetas padre) y al final lo renombra a version_dir; si version_dir apareció mientras tanto lanza VersionExistsError y borra el temporal.
load_model(path)->(pipeline, metadata) : path puede ser la carpeta de la versión o el modelo.joblib; metadata se lee de metadata.json junto al modelo (si no existe, devuelve {}); si el modelo no existe o no carga, ModelNotFoundError con mensaje claro.
build_metadata(*, version, target, params, seed, data_path, n_rows, n_train, n_test, test_size, notes)->dict
  input_schema, python_version, package_versions, sha256 del CSV, created_at (UTC ISO) y git_commit se calculan dentro.
get_git_commit()->str : variable de entorno GIT_COMMIT; si no existe, intenta `git rev-parse HEAD` con subprocess (cualquier fallo se ignora); si no, "desconocido".

## src/validation.py

validate_and_prepare_input(df, expected_columns=None, numeric_columns=None)->(X, columnas_ignoradas)
  X trae exactamente las 11 columnas en el orden de entrenamiento y conserva índice y orden de filas.

## Artefactos por versión: models/<version>/

modelo.joblib = Pipeline COMPLETO (normalize + extracciones + preprocesador + modelo).
metrics.json = {"version","metrica_principal":"RMSE (USD)","n_train","n_test",
  "test":{"random_forest":{rmse,mae,r2},"dummy_mean":{...},"dummy_median":{...}},
  "cv":null o {"folds","rmse_mean","rmse_std","mae_mean","r2_mean"}}
metadata.json = version, created_at, problem_type="regresion", target, algorithm, params, seed,
  input_schema{columns,numeric,categorical,dtypes}, data{fuente,archivo,sha256,n_filas},
  split{estrategia,test_size,n_train,n_test}, metric, python_version, requirements_file, package_versions, git_commit, notes.

## CLI

train.py:   --data (req) --target (default price) --output-dir (default models) --version (req) --seed 42 --test-size 0.2
            --n-estimators 40 --max-depth 20 --min-samples-leaf 3 --max-features 0.7 --cv-folds 0
            expone main(argv=None)->int y run_training(args)
predict.py: --model (req) --input (req) --output (req); expone main(argv=None)->int
            Salida CSV: [id si existía en la entrada], prediccion. Mismo orden de filas que la entrada.
Códigos de salida: 0 ok, 1 error previsible. argparse usa 2 por sí solo.

## Git

Ramas: main (estable), develop (integración), feature/*, fix/*, docs/*, chore/*. Mensajes de commit: `feat(train): ...`, `fix(features): ...`, `docs(readme): ...`, `test(app): ...`
