# Docker: construcción y ejecución

Ejecuta estos comandos desde `fase-2/` en PowerShell. La imagen instala las dependencias desde `requirements.txt`; los datos y resultados se montan desde el host y no se copian al construirla.

## Construcción

```powershell
docker build -t modelos-fase2:1.0 .
```

## Entrenamiento

`GIT_COMMIT` se pasa explícitamente porque la imagen no contiene `.git`; así, los metadatos conservan el identificador de commit del host en lugar de quedar como `desconocido`.

```powershell
docker run --rm -e "GIT_COMMIT=$env:GIT_COMMIT" -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data /app/data/train.csv --output-dir /app/models --version v-docker
```

## Predicción

```powershell
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model /app/models/v-docker --input /app/data/ejemplo_prediccion.csv --output /app/outputs/predicciones.csv
```

## Pruebas

```powershell
docker run --rm modelos-fase2:1.0 python -m pytest -v
```
