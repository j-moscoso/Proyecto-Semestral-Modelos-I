# Contrato de entrada para predicción

Este contrato describe el CSV que recibe `predict.py`. Ejecuta los comandos
desde `fase-2/`. La entrada debe contener las once columnas predictoras; el
orden de las columnas del CSV puede variar porque se reorganizan antes de
predecir.

## Columnas de entrada

| Nombre | Tipo esperado | Ejemplo | ¿Admite nulos? |
|---|---|---|---|
| `brand` | Texto | `Nissan` | Sí |
| `model` | Texto | `Murano SV` | Sí |
| `model_year` | Numérico, convertible a número | `2022` | Sí |
| `milage` | Numérico, convertible a número | `23677` | Sí |
| `fuel_type` | Texto | `Gasoline` | Sí |
| `engine` | Texto descriptivo | `3.5L V6 24V MPFI DOHC` | Sí |
| `transmission` | Texto descriptivo | `Automatic CVT` | Sí |
| `ext_col` | Texto | `Sunset Drift Chromaflair` | Sí |
| `int_col` | Texto | `Graphite` | Sí |
| `accident` | Texto | `None reported` | Sí |
| `clean_title` | Texto | `Yes` | Sí |

Las columnas categóricas se convierten a texto durante la normalización. En las
columnas numéricas, los valores no nulos deben convertirse a número; un valor
no numérico no se considera un nulo válido.

## Validación y tratamiento de valores

- **Columnas adicionales:** se ignoran como predictores y se enumeran en la
  línea `Columnas ignoradas`. Esto incluye `price`, que no se usa en predicción,
  e `id`. Si la entrada incluye `id`, se copia a la salida aunque también
  figure en el resumen de columnas ignoradas.
- **Columnas ausentes:** se produce un error; deben estar presentes las once
  columnas, incluso cuando todos sus valores sean nulos. El programa escribe
  `ERROR: ...` en stderr y termina con código 1.
- **Nulos:** se admiten celdas nulas en las once columnas. En particular,
  `transmission`, `engine`, `fuel_type`, `accident` y `clean_title` pueden
  estar vacías. Para las variables categóricas usadas como categorías (incluido
  el tipo derivado de `transmission`), los imputadores usan `Desconocido`.
  `engine` se convierte en características numéricas extraídas del texto; las
  que falten se imputan con la mediana aprendida en el entrenamiento. También
  se deriva el tipo y el número de velocidades de `transmission`; las
  velocidades faltantes se imputan con la mediana. Una cadena vacía o con
  espacios en una columna categórica también se trata como nulo.
- **Tipos inválidos:** `model_year` y `milage` se convierten a número. Si un
  valor no nulo no es convertible, la predicción falla con código 1 y el error
  indica la columna y hasta las primeras cinco filas problemáticas del CSV.
  Los valores de las columnas categóricas se convierten a texto; no se valida
  que coincidan con un catálogo cerrado.
- **Categorías no vistas durante el entrenamiento:** no causan error. Para
  `fuel_type`, el tipo de transmisión derivado, `accident` y `clean_title`,
  OneHotEncoder las ignora (`handle_unknown="ignore"`), por lo que esa
  característica queda representada por ceros en su bloque one-hot. Para
  `brand`, `model`, `ext_col` e `int_col`, OrdinalEncoder asigna `-1`.
  En la práctica, el modelo recibe `-1` como un valor numérico inferior a los
  códigos ordinales conocidos; no es una categoría desconocida separada y
  puede quedar agrupado con valores bajos en las particiones del bosque.

## Formato de salida y códigos de salida

El archivo de salida es CSV sin índice, con una fila por cada fila de entrada y
en el mismo orden. Contiene `prediccion` y, si estaba presente en la entrada,
`id` como primera columna. El predictor es el precio estimado.

| Código | Significado |
|---|---|
| `0` | Predicción guardada correctamente. |
| `1` | Error al leer/validar la entrada o cargar/predecir con el modelo; se escribe `ERROR: ...` en stderr. |
| `2` | Error de argumentos de línea de comandos, gestionado por `argparse` (por ejemplo, falta un argumento requerido). |

En caso de éxito, stdout informa el número de filas predichas, las columnas
ignoradas y la ruta de salida. El archivo se guarda mediante un reemplazo
atómico.

## Ejemplos de `predict.py`

Los siguientes comandos se ejecutan desde `fase-2/`. Para comprobarlos se
entrenó temporalmente el modelo `v-test-c` con
`python train.py --data data/train.csv --version v-test-c --n-estimators 5`;
el modelo temporal se eliminó al terminar la comprobación.
Cada ejemplo predice las 12 filas de `data/ejemplo_prediccion.csv`; la muestra
del CSV de salida es igual en los tres casos:

```csv
id,prediccion
111355,48864.272012856934
182258,24018.85536585366
14147,36491.81410979679
...
```

### 1. Cargar el modelo desde su carpeta

```powershell
python predict.py --model models/v-test-c --input data/ejemplo_prediccion.csv --output outputs/predicciones-contrato.csv
```

Resultado esperado:

```text
Filas predichas: 12
Columnas ignoradas: id
Salida: outputs\predicciones-contrato.csv
```

Código de salida: `0`.

### 2. Cargar directamente `modelo.joblib`

```powershell
python predict.py --model models/v-test-c/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones-contrato-directo.csv
```

Resultado esperado:

```text
Filas predichas: 12
Columnas ignoradas: id
Salida: outputs\predicciones-contrato-directo.csv
```

Código de salida: `0`.

### 3. Guardar la predicción con otro nombre

```powershell
python predict.py --model models/v-test-c --input data/ejemplo_prediccion.csv --output outputs/predicciones-contrato-repeticion.csv
```

Resultado esperado:

```text
Filas predichas: 12
Columnas ignoradas: id
Salida: outputs\predicciones-contrato-repeticion.csv
```

Código de salida: `0`.
