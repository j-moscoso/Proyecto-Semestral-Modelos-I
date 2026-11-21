"""Esquemas Pydantic de entrada y salida de la API."""

from pydantic import BaseModel, ConfigDict, Field


class VehiculoEntrada(BaseModel):
    """Representa un registro completo de vehículo recibido por la API."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    brand: str | None
    model: str | None
    model_year: float | None
    milage: float | None
    fuel_type: str | None
    engine: str | None
    transmission: str | None
    ext_col: str | None
    int_col: str | None
    accident: str | None
    clean_title: str | None


class PrediccionRespuesta(BaseModel):
    """Representa la respuesta exitosa de una predicción."""

    prediccion: float
    version: str


class EntrenarSolicitud(BaseModel):
    """Representa los parámetros opcionales de una nueva versión."""

    version: str | None = None
    n_estimators: int = Field(40, ge=1, le=500)
    seed: int = 42
