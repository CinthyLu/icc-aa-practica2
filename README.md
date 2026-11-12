# Práctica 2: Predicción de demanda con SVM

**Aprendizaje Automático · Universidad Politécnica Salesiana**
Desarrollado por: Daniel Guanga, Alex Guamán y Cinthya Ramón · Docente: Ing. Remigio Hurtado

Predicción de la demanda diaria de 10 productos (escenario sintético de La Favorita) con **SVR**, la SVM para regresión. El modelo parte de una versión base, se mejora con fine-tuning y cada prueba queda registrada en **MLflow**. El resultado se compara con la LSTM de la Práctica 1.

## Documentación

Este README contiene la explicación del trabajo, los resultados, los comandos
de ejecución y las capturas de MLflow. En `docs/` se organizan las evidencias y
el seguimiento:

- [Estado de la práctica: trabajo terminado y pendientes](docs/estado.md).
- [Capturas reales de MLflow](docs/capturas/mlflow/).

## Estado actual

El notebook actual contiene las Partes 1 y 2. La última ejecución completa, del
12 de noviembre de 2026, terminó sus 15 celdas de código sin errores y actualizó
el HTML, las figuras y los archivos de resultados.

| Trabajo | Estado |
| --- | --- |
| Preparación de datos y evaluación del SVR de la Parte 1 | Hecho |
| Registro del modelo base y el ajuste manual | Hecho |
| GridSearch, RandomSearch y comparación de tiempo y error | Hecho |
| Registro del mejor pipeline y sus versiones en MLflow | Hecho |
| Comparación base, ajuste manual y fine-tuning con ejemplos por producto | Hecho |
| Función para casos nuevos e interpretación con SHAP | No incluidas en el notebook actual |
| Random Forest y ejercicio de árboles de decisión | Pendientes |

MLflow conserva **132 ejecuciones terminadas**. La última ejecución añadió 67
registros y mantuvo los 65 anteriores. El modelo vigente es
`models:/svr_demanda/2`; la versión 1 sigue en el registro. Estos números describen
el estado de esta ejecución y aumentarán cuando se repita el entrenamiento.

## Contenido de la carpeta

| Archivo o carpeta | Contenido |
| --- | --- |
| `practica2_svm.ipynb` | Cuaderno de la práctica |
| `practica2_svm.html` | Cuaderno ejecutado, para verlo sin Python |
| `data/` | `sales_history.csv` (ventas y stock diario) e `inventory.csv` (productos) |
| `figuras/` | 7 figuras generadas por el cuaderno |
| `docs/estado.md` | Trabajo terminado, comprobaciones y pendientes con sus evidencias |
| `docs/capturas/mlflow/` | 3 capturas reales de la interfaz local de MLflow |
| `resultados/` | CSV de los experimentos y datos del modelo registrado |
| `requirements.txt` | Librerías necesarias |

`mlflow.db` y `mlartifacts/` se crean al ejecutar el cuaderno y no se suben al repositorio.

## Estructura del cuaderno

**Parte 1: Datos y SVR**
- 22 variables por fila: fecha como timestamp, calendario (día, mes, feriados de Ecuador, víspera, día de pago, Navidad), rezagos de venta, stock del día anterior y producto.
- De las 3.650 filas originales quedan 3.580 después de retirar las que no tienen historial suficiente para calcular los rezagos.
- División por fechas: 2.980 filas de entrenamiento hasta el 1 de agosto de 2026, 300 de validación del 2 al 31 de agosto y 300 de prueba en septiembre. No se barajan las fechas.
- Comparación de kernels y efecto de C y épsilon (subajuste y sobreajuste).

Se prueban los kernels lineal, RBF y polinomial con el mismo C. En validación,
sus RMSE son 4,24; 3,58 y 3,63, respectivamente. Con RBF se estudian C y épsilon
por separado y se eligen C = 10 y épsilon = 0,1. El modelo se reajusta con
entrenamiento y agosto antes de medir su resultado en septiembre: RMSE 3,21 y
MAE 2,29. Las cuatro primeras figuras muestran estas comparaciones y las
predicciones frente a las ventas reales de dos productos.

**Parte 2: Fine-tuning con MLflow**
- Modelo base: SVR lineal con valores por defecto.
- GridSearch (42 combinaciones) y RandomSearch (20) sobre kernel, C, gamma y épsilon.
- Registro del mejor pipeline completo, incluido el escalador, como `svr_demanda`. En la última ejecución se creó la versión 2.
- Comparación base → ajuste manual → fine-tuning y ejemplos de real frente a predicción.

Ambas búsquedas utilizan la validación fija de agosto mediante `PredefinedSplit`.
El escalador se ajusta dentro del pipeline con entrenamiento. El conjunto de
septiembre queda fuera de la selección de parámetros.

| Parámetro | Valores explorados |
| --- | --- |
| Kernel | `linear`, `rbf`, `poly` |
| C | 1, 10, 100 |
| Épsilon | 0,1 y 1 |
| Gamma para RBF y polinomial | `scale`, 0,01 y 0,1 |

Gamma no interviene en el kernel lineal. Esto deja 6 combinaciones lineales y
36 entre RBF y polinomial. RandomSearch toma 20 del mismo espacio, sin reemplazo
y con semilla 42. Se elige el menor RMSE de agosto; si hay empate entre búsquedas,
se toma la de menor duración. Después se reajusta el modelo elegido con
entrenamiento y agosto para la evaluación final en septiembre.

## Resultados (prueba en septiembre de 2026)

| Modelo | RMSE | MAE |
| --- | ---: | ---: |
| Base: SVR lineal por defecto | 3,50 | 2,50 |
| Ajuste manual: RBF, C = 10 | 3,21 | 2,29 |
| **Fine-tuning: RBF, C = 100, γ = 0,01, ε = 1** | **3,05** | **2,21** |
| LSTM (Práctica 1) | 4,40 | 3,10 |

- GridSearch encontró el mejor modelo (RMSE de validación 3,29); RandomSearch tardó menos de la mitad, pero llegó a 3,51.
- En septiembre, el desfase total entre lo pedido según el modelo y la demanda real baja de 127,5 (base) a 89,3 unidades/kg.
- La SVR ajustada tiene 31 % menos error que la LSTM de la Práctica 1.

La LSTM no se entrena en este notebook: sus métricas se toman de la práctica
anterior. La comparación corresponde a este conjunto sintético y no demuestra
que una SVR siempre supere a una red recurrente.

### Comparación de búsquedas de la última ejecución

| Búsqueda | Candidatos | Mejor RMSE de agosto | Tiempo de búsqueda |
| --- | ---: | ---: | ---: |
| GridSearch | 42 | 3,290 | 52,51 s |
| RandomSearch | 20 | 3,506 | 23,88 s |

Los tiempos corresponden a `fit`; no incluyen guardar las figuras ni registrar
los resultados en MLflow. Pueden cambiar según el equipo y la carga de trabajo.
El archivo `resultados/parte2_comparacion.csv` conserva los valores de la última
ejecución, y MLflow permite consultar las anteriores.

## Ejecución

Los comandos son para PowerShell en Windows. Ejecutarlos desde la carpeta que
contiene este README, el notebook y los datos.

### 1. Preparar el entorno, si todavía no existe

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si `.venv` ya está preparado, se puede pasar al siguiente paso.

### 2. Iniciar la interfaz web de MLflow

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Dejar esa terminal abierta y entrar a [MLflow local](http://127.0.0.1:5000).
Si la página ya abre, el servidor está funcionando y no hace falta arrancarlo
otra vez. `127.0.0.1` corresponde a este equipo; la interfaz no está publicada
en internet.

### 3. Ejecutar el notebook

En otra terminal, situada en la misma carpeta:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 practica2_svm.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to html practica2_svm.ipynb
```

También se puede abrir `practica2_svm.ipynb` en un editor compatible con Jupyter,
seleccionar el Python de `.venv` y pulsar **Ejecutar todo**. La Parte 2 necesita
las variables de la Parte 1.

Ejecutar el notebook guarda resultados en MLflow, pero **no inicia el servidor
web automáticamente**. El servidor y el notebook son procesos separados. Al
terminar, actualizar la página de MLflow para ver las nuevas ejecuciones.

Cada ejecución completa vuelve a entrenar, agrega registros al historial y crea
otra versión de `svr_demanda`, sin borrar las anteriores. Para leer los resultados
existentes basta con abrir el HTML. Al cerrar el servidor o reiniciar el equipo,
el historial permanece guardado; solo habrá que volver a iniciar la interfaz.

## Interfaz de MLflow

En [MLflow local](http://127.0.0.1:5000), seleccionar **Model training** y entrar
al experimento **practica2_svr**. En **Runs** se ve el historial:

- `base`: SVR lineal antes del ajuste.
- `ajuste_manual`: RBF con los parámetros elegidos en la Parte 1.
- `GridSearch` y `RandomSearch`: una ejecución principal por búsqueda. El botón
  de expansión junto a cada nombre muestra sus candidatos.
- `mejor_modelo`: parámetros elegidos, evaluación final y pipeline guardado.

Cada candidato registra sus parámetros, `rmse_train`, `rmse_val`, `tiempo_fit_s`
y la figura `rmse.png`. Las búsquedas guardan además la comparación gráfica y
los CSV. Para comparar ejecuciones, seleccionarlas y usar **Compare**. El filtro
**All time** incluye las anteriores; si aparece **Load more**, pulsarlo para
cargar el resto del historial.

Las versiones se consultan en **Model registry → svr_demanda**. La sección
**Models** dentro del experimento muestra los modelos guardados por sus
ejecuciones; el registro permite consultar las versiones 1 y 2.

### Capturas reales de MLflow

Capturas tomadas el 12 de noviembre de 2026 sobre la base local del proyecto.
Documentan ese estado; no se actualizan solas al repetir el notebook.

#### Historial de ejecuciones

Se ven los experimentos recientes y los anteriores. El contador inferior indica
132 registros; los candidatos están agrupados bajo cada búsqueda.

![Historial del experimento practica2_svr con 132 ejecuciones y búsquedas agrupadas](docs/capturas/mlflow/01_historial.jpg)

#### Métricas y parámetros del mejor modelo

La ejecución `mejor_modelo` conserva RMSE de validación 3,2897, RMSE de prueba
3,0464 y RMSE final de entrenamiento 2,7982. Sus parámetros son RBF, C = 100,
gamma = 0,01 y épsilon = 1.

![Métricas y parámetros reales de la ejecución mejor_modelo en MLflow](docs/capturas/mlflow/02_metricas_mejor_modelo.jpg)

#### Versión registrada

La última ejecución produjo la versión 2 de `svr_demanda`, con un esquema de
22 entradas y una salida. La versión 1 sigue disponible. El nombre, la versión
y la URI vigentes se guardan en `resultados/parte2_modelo.json`.

![Detalle de la versión 2 de svr_demanda y su esquema de 22 entradas en MLflow](docs/capturas/mlflow/03_versiones_modelo.jpg)

La base `mlflow.db` y la carpeta `mlartifacts/` deben conservarse juntas para
consultar los modelos y sus artefactos. No se suben a Git. El JSON de entrega
describe el modelo, pero no contiene sus archivos entrenados. En otro equipo se
debe preparar el entorno y ejecutar el notebook para crear su registro local.

## Pendiente

- Random Forest y explicación de predicciones, según el alcance que confirme el docente.
- La función de predicción para muestras nuevas y SHAP no están en el notebook actual. No se consideran terminados en esta versión.
- Ejercicio de árboles de decisión (Information Gain) de la guía.
- Confirmar con el docente el enfoque: este modelo aprende demanda y después calcula stock; no predice stock directamente.
- Revisar las referencias APA, confirmar los integrantes permitidos y realizar la entrega en AVAC y la sustentación.

El [estado de la práctica](docs/estado.md) detalla qué se comprobó, qué falta
implementar y qué requiere confirmación del docente. Las capturas y los archivos
de resultados documentan una ejecución concreta; hay que revisarlos al incorporar
nuevos cambios.
