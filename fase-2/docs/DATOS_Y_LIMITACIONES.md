# Datos y limitaciones

## 1. Procedencia del dataset

Los datos de la competición Playground Series S4E9 no son observaciones directas del mercado: fueron generados con un modelo de aprendizaje profundo entrenado sobre el dataset [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) (Kaggle, autor taeefnajib), que según su propia descripción combina datos reales observados y datos sintéticos. No es posible determinar qué proporción de las 188.533 filas refleja registros reales, por lo que las conclusiones describen el comportamiento del modelo sobre estos datos y no se extrapolan al mercado real de autos usados ni a un uso comercial.

El conjunto permite comparar modelos bajo su distribución y analizar el comportamiento de sus predicciones. No permite estimar por sí solo precios del mercado real, confirmar que anuncios repetidos correspondan a vehículos repetidos ni justificar una aplicación comercial.

## 2. Prueba reservada y EDA

En la Fase 1, el EDA consultó todo el conjunto, incluido `price`, para orientar decisiones de preparación. Esto es una exposición exploratoria de la prueba, aunque los imputadores, codificadores, escalador y modelo no se ajustaron con ella.

En la Fase 2, `train.py` separa `train` y `test` antes de ajustar transformadores o modelos. Los hiperparámetros de v2 se fijaron sin mirar `test`. Cualquier cambio de codificación o de transformación del objetivo (por ejemplo, `log1p(price)`) debe evaluarse únicamente mediante validación cruzada dentro de `train`; el `test` reservado se consulta solo para la evaluación final.

## 3. Disponibilidad al predecir

### 3.1 Evidencia documental

Según la descripción del dataset original, `accident` indica si el vehículo tiene antecedentes de accidentes o daños previos, información relevante para decidir; `clean_title` indica la disponibilidad de un título limpio, que puede afectar el valor de reventa y la situación legal. Esa descripción no indica en qué momento se conocen estas dos variables.

### 3.2 Evidencia empírica

En el ajuste de la validación interna, el porcentaje de nulos fue:

- `accident`: **1.306978 %**
- `clean_title`: **11.277971 %**

La siguiente tabla reproduce los resultados del análisis. El escenario A es la referencia con ambas columnas disponibles. En B1, B2 y B3 se usa el mismo modelo de A, pero las columnas correspondientes se dejan vacías al predecir. En C1, C2 y C3 se reentrena un modelo sin esas columnas.

| escenario | descripcion | RMSE | MAE | R2 | variacion_RMSE_% |
|---|---|---:|---:|---:|---:|
| A | Referencia: ambas disponibles | 64662.302215 | 19683.816133 | 0.122494 | 0.000000 |
| B1 | Sin accident; mismo modelo | 64918.040703 | 20462.910256 | 0.115540 | 0.395499 |
| B2 | Sin clean_title; mismo modelo | 65430.934356 | 21111.407292 | 0.101509 | 1.188687 |
| B3 | Sin accident, clean_title; mismo modelo | 65913.703661 | 21892.311253 | 0.088201 | 1.935287 |
| C1 | Sin accident; modelo reentrenado | 64636.108924 | 19720.011354 | 0.123205 | -0.040508 |
| C2 | Sin clean_title; modelo reentrenado | 64423.394689 | 19679.253801 | 0.128967 | -0.369470 |
| C3 | Sin accident, clean_title; modelo reentrenado | 64782.900756 | 19726.235143 | 0.119218 | 0.186505 |

Frente a A, al dejar `accident` vacío en B1 el RMSE aumenta **0.395499 %** y el MAE **3.96 %**; ambas diferencias son pequeñas en RMSE, al ser menor que 1 %. En B2, al dejar `clean_title` vacío, el RMSE aumenta **1.188687 %** y el MAE **7.25 %**. En B3, al dejar ambas columnas vacías, el RMSE aumenta **1.935287 %** y el MAE **11.22 %**. Los escenarios C recuperan parte de esa pérdida frente a B: C1 y C2 quedan por debajo del RMSE de A, mientras C3 queda ligeramente por encima. Las variaciones de C son menores al 1 % en RMSE y de signo mixto, por lo que están dentro del ruido de una sola partición y una sola semilla; no se concluye que quitar una variable mejore el modelo.

En los escenarios C, scikit-learn emite el aviso "Skipping features without any observed values" porque `SimpleImputer` omite las columnas sin valores observados; es un comportamiento esperado y equivale a entrenar sin esas variables. Los avisos no forman parte del log guardado.

### 3.3 Decisión y contrato de entrada

Ambas columnas deben existir en el CSV de entrada, pero pueden venir vacías; en ese caso se tratan como nulos y se imputan a la categoría `Desconocido`. Si el sistema que consume el modelo no tendrá alguna de estas variables, debe utilizar una versión entrenada sin ellas. El contrato completo está documentado en el [contrato de entrada](./CONTRATO_ENTRADA.md).

### 3.4 Límite de esta validación

No hay fechas ni un identificador de vehículo, los datos son en parte sintéticos y la fuente no indica cuándo se conocen estas variables. Por ello, su disponibilidad real debe confirmarse con quien envíe los datos. El análisis usa una validación interna de `train` y no la prueba reservada.

## 4. Posibles vehículos repetidos

No hay VIN ni otro identificador de vehículo que permita agrupar de forma confiable. Aunque no se pueden confirmar vehículos repetidos, sí hay combinaciones repetidas o muy similares de marca, modelo y año. Por ello, un `train_test_split` aleatorio puede colocar anuncios del mismo vehículo, o de vehículos/versiones muy similares, en ambos subconjuntos y producir una evaluación optimista. El resultado no debe interpretarse como desempeño garantizado sobre vehículos completamente nuevos.

## 5. Otras limitaciones

- El R² del Random Forest en `test` fue **0.13963386499245978** en v1 y **0.15244748135646324** en v2. Son valores bajos: la mejora sobre el baseline no implica explicar la mayor parte de la variabilidad ni valorar con precisión cada vehículo.
- `price` presenta una cola derecha pronunciada. Los valores extremos pueden elevar el RMSE; por ello conviene acompañarlo con MAE y R² y no interpretar una sola métrica como descripción completa del error.
- `model`, `brand` y los colores tienen cardinalidad alta. Su codificación ordinal impone un orden artificial entre categorías y limita la interpretación de importancias y efectos.
- La comparación de otra codificación o de una transformación del objetivo debe hacerse mediante CV dentro de entrenamiento antes de usar la prueba reservada.
