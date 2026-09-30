# Fase 1: Modelo predictivo de precios de autos usados

## Equipo y problema

**Integrantes:** Juan Manuel Moscoso Torres, Slleider David Rojas Aleman y Samuel Castaño Arenas.  
**Curso:** Modelos y Simulación de Sistemas I, Universidad de Antioquia.

El objetivo es predecir `price` a partir de las características de un vehículo usado. Es un problema supervisado de regresión tabular, no una serie temporal. El problema fue aprobado por el docente. Los datos provienen de [Regression of Used Car Prices, Kaggle Playground Series S4E9](https://www.kaggle.com/competitions/playground-series-s4e9).

## Datos

El archivo `data/train.csv` contiene 188.533 observaciones y 13 columnas: `id`, 11 predictores y el objetivo `price`. `id` es un identificador consecutivo y no se usa como predictor. Los predictores incluyen marca, modelo, año, kilometraje, combustible, motor, transmisión, colores, reporte de accidente y estado del título.

| Variable | Valores faltantes | Porcentaje | Tratamiento |
| --- | ---: | ---: | --- |
| `accident` | 2.452 | 1,301% | Imputación como categoría `Desconocido`; `None reported` se conserva como valor observado distinto. |
| `fuel_type` | 5.083 | 2,696% | Imputación categórica como `Desconocido`. |
| `clean_title` | 21.419 | 11,361% | Imputación categórica como `Desconocido`; los registros observados solo contienen `Yes`. |

El requisito del curso de una predictora con 0,1%-2% de faltantes se satisface con `accident`. `fuel_type` y `clean_title` superan ese intervalo, por lo que no se describen como si también lo cumplieran. No se eliminaron filas automáticamente por tener predictores faltantes.

Los precios observados van de 2.000 a 2.954.083; la mediana es 30.825 y la media 43.878. La distribución tiene cola derecha pronunciada. Los extremos se conservan porque podrían corresponder a autos válidos; el RMSE es sensible a sus errores y se acompaña de MAE y R². `price` es el campo del dataset; no se asume que siempre equivalga al precio transaccional efectivamente pagado.

## Preparación y prevención de fuga

El notebook separa `price` y descarta `id`, después realiza una partición aleatoria reproducible 80/20 con `RANDOM_STATE = 42`. Las extracciones de HP, litros y cilindros desde `engine`, y de tipo/velocidades desde `transmission`, son funciones por fila definidas en `features.py`.

Las extracciones, `build_preprocessor()` y el estimador están juntos en un `Pipeline`. La imputación numérica usa mediana; las variables categóricas se imputan como `Desconocido`; las categorías compactas usan One-Hot Encoding y las de alta cardinalidad usan codificación ordinal con manejo de categorías no vistas. El módulo escala las numéricas; este paso se conserva aunque Random Forest no dependa del escalamiento. El preprocesador y el modelo se ajustan solo con entrenamiento y, en CV, dentro de cada fold. El test no participa en los ajustes ni en la selección del modelo.

No hay duplicados exactos de predictores, pero se repiten combinaciones de marca, modelo y año. Sin VIN no puede determinarse si anuncios repetidos corresponden al mismo vehículo; por ello, el split aleatorio podría dar una evaluación optimista para vehículos o versiones muy similares.

## Modelo y evaluación

Se compara el modelo contra `DummyRegressor` con estrategias de media y mediana. El modelo predictivo es un `RandomForestRegressor` acotado para mantener razonable el tiempo de entrenamiento. Se usa validación cruzada K-Fold de cinco particiones sobre `X_train` y una evaluación final en `X_test`.

La competición utiliza RMSE, reportado en dólares y como métrica principal. MAE también se expresa en dólares y resume el error absoluto promedio; R² no tiene unidades y puede ser negativo. Resultados de la ejecución limpia del notebook:

| Evaluación | Modelo | RMSE (dólares) | MAE (dólares) | R² |
| --- | --- | ---: | ---: | ---: |
| Test | Dummy (media) | 74.573 | 29.087 | ≈ 0,0000 |
| Test | Dummy (mediana) | 75.703 | 26.548 | -0,0305 |
| Test | Random Forest | 69.171 | 20.317 | 0,1396 |
| CV 5-fold (media) | Dummy (media) | 79.667 | 29.137 | -0,0001 |
| CV 5-fold, media ± desviación | Random Forest | 75.120 ± 5.833 | 20.468 ± 390 | 0,1111 ± 0,0092 |

En el test, Random Forest reduce RMSE en 7,24% y MAE en 30,15% frente al baseline de media. Es una mejora útil, pero R² de 0,1396 muestra que queda mucha variabilidad por explicar; no se presenta como una estimación precisa de cada vehículo. El notebook incluye gráficos de real contra predicho, residuos e importancia de variables. La importancia señala `milage` como principal atributo del bosque, pero es descriptiva, no causal; la codificación ordinal también limita su interpretación.

## Archivos

```text
fase-1/
├── data/train.csv
├── features.py
├── modelo.joblib
├── notebook.ipynb
├── README.md
└── requirements.txt
```

El notebook ejecutable de la entrega es `fase-1/notebook.ipynb`. Los notebooks de `fase-1/notebooks/` son documentos complementarios.

## Instalación y ejecución

Se probó con Python 3.11.9 y las versiones fijadas en `requirements.txt`. Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.11 -m venv .venv
\.venv\Scripts\python.exe -m pip install --upgrade pip
\.venv\Scripts\python.exe -m pip install -r fase-1/requirements.txt
code .
```

En VS Code, instalar las extensiones Python y Jupyter si no están disponibles, abrir `fase-1/notebook.ipynb`, seleccionar `.venv` como kernel y ejecutar **Run All** desde el inicio. La primera celda instala o verifica automáticamente los paquetes de `requirements.txt` en el kernel activo; si se actualizan paquetes que ya estaban cargados, reiniciar el kernel y volver a ejecutar **Run All**. El notebook resuelve sus rutas desde la raíz, `fase-1/` o la carpeta complementaria. No requiere Google Drive ni rutas personales.

**Archivos necesarios:** aunque el notebook instala sus paquetes, no es autónomo. Para entrenar desde cero también deben estar disponibles `fase-1/data/train.csv` y `fase-1/features.py`; ambos forman parte de la entrega y el notebook los carga durante el entrenamiento.

## Modelo guardado e inferencia

El pipeline completo se guarda en `fase-1/modelo.joblib` e incluye extracción de variables, preprocesador y estimador. La carga debe realizarse con las dependencias instaladas y con `fase-1/features.py` accesible para Python, porque el pipeline serializado referencia sus funciones. El notebook comprueba la carga con `joblib.load`, valida con `np.allclose` que las predicciones del modelo recargado coincidan y predice tres filas crudas con los 11 predictores originales, sin `id` ni `price`.

## Limitaciones y mejoras futuras

- El objetivo es muy asimétrico y algunos errores extremos elevan RMSE.
- `model`, marca y colores tienen cardinalidad alta; la codificación ordinal impone un orden artificial.
- Puede haber vehículos o anuncios muy similares entre train y test; no se dispone de un identificador de vehículo para hacer una separación agrupada confiable.
- Como trabajo futuro se puede comparar `log1p(price)` con `TransformedTargetRegressor`, otros modelos de boosting y codificaciones categóricas; cualquier elección debe hacerse con CV en train y reportarse en dólares antes de consultar el test.

Docker y API REST corresponden a fases posteriores. La organización del curso contempla tres entregas para cuatro fases: la tercera entrega reúne las fases 3 y 4.
