# Contrato de la API de predicción de vehículos

La API FastAPI de la Fase 3 expone predicción, consulta de salud y entrenamiento
de versiones. Los ejemplos corresponden al contrato y a la primera fila de
`data/ejemplo_prediccion.csv`; la predicción se deja como `<valor>` para no
presentarla como un resultado medido aquí.

## Rutas

### `GET /health`

Consulta el estado del servicio y las versiones de modelo disponibles. No carga
el modelo ni falla solo porque falte.

**Respuesta `200`:**

```json
{
  "estado": "ok",
  "version_activa": "v1",
  "versiones_disponibles": ["v1", "v2"]
}
```

### `POST /predict`

Envía todos los campos del vehículo; cada clave es obligatoria, aunque su valor
puede ser `null`. Esta solicitud representa la primera fila del CSV, sin las
columnas `id` y `price`:

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

**Respuesta `200` (forma):**

```json
{
  "prediccion": "<valor>",
  "version": "v1"
}
```

### `POST /train`

El cuerpo es opcional. Sin cuerpo, usa `version: null`, `n_estimators: 40` y
`seed: 42`; si no se especifica una versión, selecciona el siguiente
identificador disponible. Por ejemplo, se puede enviar:

```json
{
  "version": null,
  "n_estimators": 40,
  "seed": 42
}
```

**Respuesta `201` (forma):**

```json
{
  "version": "v3",
  "activada": false,
  "metricas_test": {
    "rmse": "<valor>",
    "mae": "<valor>",
    "r2": "<valor>"
  },
  "ruta": "models/v3",
  "mensaje": "Versión guardada. Para usarla configure MODEL_VERSION=v3 y reinicie la API."
}
```

El cuerpo de respuesta es ilustrativo: las métricas dependen del entrenamiento.
El contrato no sobrescribe versiones existentes ni activa automáticamente la
nueva.

## Campos de predicción

| Nombre | Tipo JSON | ¿Obligatorio? | Ejemplo | Unidades o significado |
|---|---|---|---|---|
| `brand` | texto | Sí; admite `null` | `"Nissan"` | Marca del vehículo |
| `model` | texto | Sí; admite `null` | `"Murano SV"` | Modelo |
| `model_year` | número | Sí; admite `null` | `2022` | Año del modelo |
| `milage` | número | Sí; admite `null` | `23677` | Kilometraje en millas (`mi`) |
| `fuel_type` | texto | Sí; admite `null` | `"Gasoline"` | Tipo de combustible |
| `engine` | texto | Sí; admite `null` | `"3.5L V6 24V MPFI DOHC"` | Texto del motor; se derivan potencia (HP), litros y cilindros cuando el texto coincide con los patrones |
| `transmission` | texto | Sí; admite `null` | `"Automatic CVT"` | Texto de transmisión; se derivan tipo y velocidades cuando se reconocen |
| `ext_col` | texto | Sí; admite `null` | `"Sunset Drift Chromaflair"` | Color exterior |
| `int_col` | texto | Sí; admite `null` | `"Graphite"` | Color interior |
| `accident` | texto | Sí; admite `null` | `"None reported"` | Antecedentes de accidentes o daños reportados |
| `clean_title` | texto | Sí; admite `null` | `null` | Disponibilidad de título limpio |

`model_year` y `milage` aceptan números JSON; las cadenas numéricas también se
convierten y las no numéricas producen un error `422`. Los demás campos esperan
texto o `null`; un número JSON en uno de ellos produce `422`.

El objetivo `price` está expresado en dólares estadounidenses y no se envía en
la solicitud de predicción. `id` tampoco es necesario. Las claves adicionales
se ignoran.

## Nulos y categorías

- Los valores `null` y las cadenas vacías (incluidas las que solo tienen
  espacios) se tratan como faltantes.
- Los faltantes de las variables categóricas se imputan con `Desconocido`; los
  numéricos se imputan con la mediana aprendida durante el entrenamiento.
- Las categorías nuevas se ignoran en la codificación OneHot. En variables con
  codificación ordinal, las categorías desconocidas se representan como `-1`.
- Aunque el valor sea `null`, las 11 claves deben estar presentes.
- Los campos extra se ignoran.

## Códigos de respuesta y errores

Los errores usan el formato `{"error":"<codigo>","mensaje":"<texto>"}`.

| Código HTTP | Significado | Ejemplo de cuerpo |
|---:|---|---|
| `200` | Solicitud de salud o predicción exitosa | `{"prediccion": "<valor>", "version": "v1"}` |
| `201` | Nueva versión entrenada, no activada | `{"version":"v3","activada":false,"metricas_test":{"rmse":"<valor>","mae":"<valor>","r2":"<valor>"},"ruta":"models/v3","mensaje":"Versión guardada. Para usarla configure MODEL_VERSION=v3 y reinicie la API."}` |
| `409` | Versión ya existente o entrenamiento en curso | `{"error":"entrenamiento_en_curso","mensaje":"Ya hay un entrenamiento en curso."}` |
| `422` | Cuerpo mal formado o dato inválido | `{"error":"entrada_invalida","mensaje":"falta el campo 'brand'"}` |
| `503` | El modelo solicitado no está disponible | `{"error":"modelo_no_disponible","mensaje":"No existe el modelo: 'models\\v99'. Versiones disponibles: v1, v2. Configure MODEL_VERSION con una de ellas."}` |
| `500` | Error interno no previsto | `{"error":"error_interno","mensaje":"Error interno del servidor."}` |

El texto exacto de los errores puede variar según la causa. Para `409`, el
código puede ser `version_existente` o `entrenamiento_en_curso`.

## Selección de la versión

La API usa la variable de entorno `MODEL_VERSION` para elegir el modelo de
predicción. Su valor predeterminado es `v1`. Si se solicita una versión que no
existe, `/predict` responde `503` e incluye las versiones disponibles en el
mensaje.

En PowerShell, antes de iniciar la API:

```powershell
$env:MODEL_VERSION = "v2"
.\.venv\Scripts\python.exe -m uvicorn api:app --port 8000
```

En Docker, configure la variable al crear el contenedor:

```powershell
docker run --rm -p 8000:8000 -e "MODEL_VERSION=v2" `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models:ro" `
  modelos-fase3:1.0
```

## Limitaciones

Los datos de la competencia Kaggle Playground Series S4E9 fueron generados
sintéticamente mediante un modelo entrenado con otro conjunto de datos, que a su
vez combina datos observados y sintéticos; no se conoce qué proporción de las
filas originales corresponde a cada tipo. Por ello, las predicciones describen
el comportamiento del modelo sobre estos datos y no representan una estimación
garantizada del mercado real.

`accident` y `clean_title` deben conocerse al momento de predecir para
proporcionar su valor; si no se conocen, envíe `null` y el modelo imputará el
faltante.
