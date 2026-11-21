# Comandos Docker de la Fase 4

```powershell
docker build -t modelos-fase4:1.0 .
docker run --rm -p 8000:8000 -e "MODEL_VERSION=v1" -e "GIT_COMMIT=$env:GIT_COMMIT" -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" modelos-fase4:1.0
.\.venv\Scripts\python.exe cliente_prueba.py health
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" modelos-fase4:1.0 python -m pytest -q
```

El servicio queda publicado en el puerto local 8000 y reenviado al puerto 8000 del contenedor.
Los datos se conectan desde `data/` al contenedor en `/app/data` con montaje de solo lectura.
Los modelos se guardan en `models/` mediante el montaje `/app/models`, donde `/train` también necesita permiso de escritura.
