# Datos y limitaciones

## 1. Procedencia del dataset

Playground Series S4E9 es un conjunto **sintético** generado a partir del conjunto real *Used Car Price Prediction Dataset*. Sus 188.533 filas no son observaciones directas del mercado. Por tanto, los resultados describen el desempeño sobre el conjunto de competición y sí permiten comparar modelos bajo esa distribución; no permiten estimar por sí solos precios del mercado real, confirmar que anuncios repetidos correspondan a vehículos repetidos ni justificar una aplicación comercial.

La relación entre cada fila sintética y un anuncio o vehículo del conjunto original debe considerarse **[VERIFICAR]** si se requiere una interpretación a nivel de registro.

## 2. Prueba reservada y EDA

En la Fase 1, el EDA consultó todo el conjunto, incluido `price`, para orientar decisiones de preparación. Esto es una exposición exploratoria de la prueba, aunque los imputadores, codificadores, escalador y modelo no se ajustaron con ella.

En la Fase 2, `train.py` separa `train` y `test` antes de ajustar transformadores o modelos. Los hiperparámetros de v2 se fijaron sin mirar `test`. Cualquier cambio de codificación o de transformación del objetivo (por ejemplo, `log1p(price)`) debe evaluarse únicamente mediante validación cruzada dentro de `train`; el `test` reservado se consulta solo para la evaluación final.

## 3. Disponibilidad al predecir

Se supone que `accident` y `clean_title` están disponibles en el momento real de la predicción, porque forman parte de las variables de entrada. Esta disponibilidad debe validarse en el proceso que genere las predicciones: **[VERIFICAR]**.

Si alguno de estos campos llega vacío, se trata como nulo y se imputa a la categoría `Desconocido`. La predicción sigue siendo posible, pero la capacidad predictiva puede cambiar frente a registros con información observada; la magnitud de ese cambio debe medirse con validación que reproduzca esa ausencia.

## 4. Posibles vehículos repetidos

No hay VIN ni otro identificador de vehículo que permita agrupar de forma confiable. Aunque no se pueden confirmar vehículos repetidos, sí hay combinaciones repetidas o muy similares de marca, modelo y año. Por ello, un `train_test_split` aleatorio puede colocar anuncios del mismo vehículo, o de vehículos/versiones muy similares, en ambos subconjuntos y producir una evaluación optimista. El resultado no debe interpretarse como desempeño garantizado sobre vehículos completamente nuevos.

## 5. Otras limitaciones

- El R² del Random Forest en `test` fue **0.13963386499245978** en v1 y **0.15244748135646324** en v2. Son valores bajos: la mejora sobre el baseline no implica explicar la mayor parte de la variabilidad ni valorar con precisión cada vehículo.
- `price` presenta una cola derecha pronunciada. Los valores extremos pueden elevar el RMSE; por ello conviene acompañarlo con MAE y R² y no interpretar una sola métrica como descripción completa del error.
- `model`, `brand` y los colores tienen cardinalidad alta. Su codificación ordinal impone un orden artificial entre categorías y limita la interpretación de importancias y efectos.
- La comparación de otra codificación o de una transformación del objetivo debe hacerse mediante CV dentro de entrenamiento antes de usar la prueba reservada.

