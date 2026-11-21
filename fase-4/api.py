"""API REST para la predicción y el entrenamiento de modelos de vehículos."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api_schema import EntrenarSolicitud, VehiculoEntrada
from src import serving
from src.schema import (
    AppError,
    InputValidationError,
    ModelNotFoundError,
    VersionExistsError,
)


app = FastAPI(title="API de precios de vehículos usados", version="4.0")


def _mensaje_validacion(error: dict) -> str:
    """Convierte un error de validación de Pydantic en un mensaje en español."""
    if error.get("type") == "json_invalid":
        return "JSON mal formado"

    location = [str(part) for part in error.get("loc", ()) if part != "body"]
    field = ".".join(location)
    error_type = error.get("type")
    messages = {
        "string_type": "se esperaba un texto",
        "float_type": "se esperaba un número",
        "float_parsing": "se esperaba un número válido",
        "int_type": "se esperaba un número entero",
        "int_parsing": "se esperaba un número entero válido",
        "greater_than_equal": f"debe ser mayor o igual que {error.get('ctx', {}).get('ge')}",
        "less_than_equal": f"debe ser menor o igual que {error.get('ctx', {}).get('le')}",
    }

    if error_type == "missing":
        return f"falta el campo '{field}'" if field else "falta el cuerpo de la solicitud"

    detail = messages.get(error_type, error.get("msg", "valor no válido"))
    if not field:
        return detail
    return f"campo '{field}': {detail}"


@app.exception_handler(RequestValidationError)
def manejar_error_validacion(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    """Devuelve todos los errores de validación de la solicitud en español."""
    del request
    messages = [_mensaje_validacion(item) for item in error.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "error": "entrada_invalida",
            "mensaje": "; ".join(messages),
        },
    )


@app.exception_handler(AppError)
def manejar_error_aplicacion(request: Request, error: AppError) -> JSONResponse:
    """Traduce los errores de aplicación al formato de la API."""
    del request
    if isinstance(error, InputValidationError):
        error_code, status_code = "entrada_invalida", 422
        message = str(error)
    elif isinstance(error, VersionExistsError):
        error_code, status_code = "version_existente", 409
        message = str(error)
    elif isinstance(error, ModelNotFoundError):
        error_code, status_code = "modelo_no_disponible", 503
        message = str(error)
        if "Configure MODEL_VERSION" not in message:
            message += " Configure MODEL_VERSION con una de ellas."
    elif isinstance(error, serving.TrainingInProgressError):
        error_code, status_code = "entrenamiento_en_curso", 409
        message = str(error)
    else:
        error_code, status_code = "error_interno", 500
        message = "Error interno del servidor."

    return JSONResponse(
        status_code=status_code,
        content={"error": error_code, "mensaje": message},
    )


@app.exception_handler(Exception)
def manejar_error_interno(request: Request, error: Exception) -> JSONResponse:
    """Oculta detalles internos y devuelve una respuesta de error genérica."""
    del request, error
    return JSONResponse(
        status_code=500,
        content={
            "error": "error_interno",
            "mensaje": "Error interno del servidor.",
        },
    )


@app.get("/health")
def health() -> dict:
    """Informa que el servicio está activo y enumera las versiones disponibles."""
    return {
        "estado": "ok",
        "version_activa": serving.get_active_version(),
        "versiones_disponibles": serving.list_versions(),
    }


@app.post("/predict")
def predict(entrada: VehiculoEntrada, request: Request) -> dict:
    """Predice el precio de un vehículo con el modelo activo."""
    prediction, version = serving.predict_record(entrada.model_dump())
    request.state.version = version
    request.state.prediccion = prediction
    return {"prediccion": prediction, "version": version}


@app.post("/train", status_code=201)
def train(solicitud: EntrenarSolicitud | None = None) -> dict:
    """Entrena una versión nueva sin activarla ni sobrescribir versiones."""
    request = solicitud or EntrenarSolicitud()
    result = serving.train_new_version(
        version=request.version,
        n_estimators=request.n_estimators,
        seed=request.seed,
    )
    return {
        "version": result["version"],
        "activada": False,
        "metricas_test": result["metricas_test"],
        "ruta": result["ruta"],
        "mensaje": "Versión guardada. Para usarla configure MODEL_VERSION="
        f"{result['version']} y reinicie la API.",
    }
