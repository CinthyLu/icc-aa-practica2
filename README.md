# Práctica 2: SVR y MLflow

El proyecto estima la demanda diaria de productos con SVR, una SVM para regresión.
El trabajo está en `practica2_svm.ipynb`; `practica2_svm.html` permite consultar los
resultados guardados sin ejecutar Python.

## Estado del trabajo

| Parte | Estado | Contenido |
| --- | --- | --- |
| Parte 1: datos y SVR | Terminada | Preparación de variables, división temporal, comparación de kernels y ajuste de C y épsilon. |
| Parte 2: optimización con MLflow | Terminada y comprobada | GridSearch, RandomSearch, registro de experimentos y del mejor modelo, comparación de RMSE y tiempo. |
| Parte 3: predicción e interpretación | Pendiente, a cargo del compañero | Cargar una versión del modelo, predecir dos casos nuevos y explicar sus resultados con SHAP. |

### Lo que ya está hecho

- [x] Reutilizar los datos de la práctica anterior y convertir la fecha en una variable numérica.
- [x] Separar entrenamiento, validación de agosto y prueba de septiembre, sin barajar las fechas.
- [x] Ejecutar GridSearch con 42 combinaciones y RandomSearch con 20, usando la misma validación.
- [x] Registrar en MLflow los parámetros, RMSE de entrenamiento y validación, tiempo de ajuste y figura de cada candidato.
- [x] Comparar ambas búsquedas en una figura y exportar sus resultados a CSV.
- [x] Guardar el pipeline completo, con su escalador, como `models:/svr_demanda/1`.
- [x] Ejecutar el cuaderno completo, comprobar la carga del modelo y actualizar el HTML.

### Resultados guardados de la Parte 2

| Búsqueda | Combinaciones | Mejor RMSE de agosto | Tiempo de búsqueda |
| --- | ---: | ---: | ---: |
| GridSearch | 42 | 3,290 | 128,21 s |
| RandomSearch | 20 | 3,506 | 57,09 s |

GridSearch obtuvo el menor error de validación. El modelo elegido usa kernel RBF,
C = 100, gamma = 0,01 y épsilon = 1. Tras entrenarlo con los datos de entrenamiento
y agosto, su RMSE en septiembre fue 3,046. Los tiempos corresponden a esta ejecución
y pueden cambiar según el equipo.

### Lo que falta hacer

- [ ] Crear `predecir(muestra, version)` para cargar el modelo registrado sin volver a entrenarlo.
- [ ] Probar la función con dos casos distintos: un día normal y un feriado, con sus variables y rezagos completos.
- [ ] Explicar las predicciones con SHAP y comentar qué variables influyeron en cada caso.
- [ ] Incorporar las conclusiones de la Parte 3 y volver a exportar el HTML.
- [ ] Revisar las referencias APA y confirmar la entrega de los archivos en AVAC y la sustentación.

Antes de cerrar la práctica hay que aclarar con el docente si acepta predecir
demanda y calcular después el stock, porque en la grabación menciona stock como
objetivo. El ejercicio de árboles e Information Gain de la guía sigue pendiente;
el profesor lo dejó para otra etapa. La guía permite hasta dos integrantes y la
portada tiene tres, por lo que también conviene confirmar la conformación del grupo.

## Datos y validación

Se usa el escenario sintético de La Favorita de la práctica anterior: 10 productos,
365 días y 3.650 registros, del 1 de octubre de 2025 al 30 de septiembre de 2026.

- `data/sales_history.csv` contiene las ventas por producto y fecha, la cantidad
  vendida y el stock disponible al inicio del día.
- `data/inventory.csv` contiene los 10 productos, sus categorías y unidades.

La variable objetivo es `cantidad_vendida`. Las entradas incluyen la fecha como
timestamp, calendario, feriados de Ecuador, ventas anteriores, la media de los
últimos siete días y el stock de cierre del día anterior. El producto se representa
con columnas indicadoras. Al retirar las filas sin historial suficiente quedan
3.580 muestras y 22 columnas de entrada.

| Conjunto | Fechas | Muestras | Uso |
| --- | --- | ---: | --- |
| Entrenamiento | Antes del 2 de agosto de 2026 | 2.980 | Ajustar el escalador y los candidatos. |
| Validación | Del 2 al 31 de agosto de 2026 | 300 | Comparar parámetros y elegir el modelo. |
| Prueba | Septiembre de 2026 | 300 | Evaluar el modelo elegido al final. |

No se barajan las fechas. `PredefinedSplit` mantiene agosto separado durante las
búsquedas y `StandardScaler` se ajusta dentro del pipeline, solo con entrenamiento.
Después de elegir los parámetros se entrena el modelo final con entrenamiento y
agosto, y se evalúa en septiembre. Ese error de prueba no se usa para seleccionar
otra configuración.

## Cómo funciona la Parte 2

Ambas búsquedas usan este espacio de parámetros:

| Parámetro | Valores |
| --- | --- |
| Kernel | `linear`, `rbf`, `poly` |
| C | 1, 10, 100 |
| Épsilon | 0,1 y 1 |
| Gamma, solo para RBF y polinomial | `scale`, 0,01 y 0,1 |

GridSearch recorre las 42 combinaciones: 6 lineales y 36 entre RBF y polinomial.
RandomSearch selecciona 20 del mismo espacio, sin reemplazo y con semilla 42.
Cada búsqueda usa un solo proceso. El menor RMSE de validación determina la
configuración elegida; si ambos métodos empatan, se toma el de menor duración.

El RMSE es la raíz del promedio de los errores al cuadrado; un valor menor indica
menor error. `rmse_train` y `rmse_val` corresponden al candidato entrenado sin
agosto. El RMSE final de septiembre corresponde al modelo reajustado con
entrenamiento y agosto. El tiempo de búsqueda incluye `fit`, pero deja fuera la
creación de figuras y el registro en MLflow.

## Ejecución local

Se necesita Python 3.12. Los comandos siguientes son para PowerShell en Windows
y se ejecutan desde la carpeta que contiene este README. No hacen falta GPU,
cuentas de MLflow ni un servidor externo.

Crear el entorno e instalar las dependencias, si todavía no está preparado:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Abrir `practica2_svm.ipynb` en un editor compatible con Jupyter, seleccionar el
Python de `.venv` como kernel y ejecutar las celdas en orden. La Parte 2 necesita
las variables que crea la Parte 1; no se puede comenzar desde sus celdas con el
kernel vacío. La Parte 3 sigue pendiente.

También se puede ejecutar todo el cuaderno desde la terminal:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 practica2_svm.ipynb
```

Este comando vuelve a entrenar, actualiza las salidas del notebook y agrega nuevas
ejecuciones y una nueva versión del modelo en MLflow. Para leer los resultados
existentes basta con abrir el HTML; no es necesario repetir el entrenamiento.

## Cómo consultar MLflow

Después de ejecutar la Parte 2, iniciar la interfaz desde la misma carpeta:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir http://127.0.0.1:5000 y entrar al experimento `practica2_svr`:

- `GridSearch` y `RandomSearch` son las ejecuciones principales. Sus candidatos
  aparecen como ejecuciones hijas, cada uno con sus parámetros, `rmse_train`,
  `rmse_val`, `tiempo_fit_s` y la figura `rmse.png`.
- Las ejecuciones principales guardan el mejor error, el número de candidatos,
  el tiempo total, la comparación gráfica y los CSV.
- `mejor_modelo` conserva la evaluación final y el pipeline registrado como
  `svr_demanda`. Su métrica `rmse_val_seleccion` es la que decidió la configuración;
  `rmse_test` es el resultado final de septiembre.

La primera ejecución completa registra 42 candidatos de GridSearch y 20 de
RandomSearch. Al repetirla se agregan registros; los anteriores no se borran.

La base `mlflow.db` y la carpeta `mlartifacts/` son locales y no se suben a Git.
No moverlas por separado: MLflow conserva rutas a los artefactos. En otro equipo,
instalar las dependencias y volver a ejecutar el cuaderno para crear su propio registro.

## Archivos del proyecto

| Archivo o carpeta | Contenido |
| --- | --- |
| `practica2_svm.ipynb` | Código, explicaciones y salidas de las Partes 1 y 2. |
| `practica2_svm.html` | Exportación del cuaderno con las salidas guardadas. |
| `requirements.txt` | Dependencias para preparar el entorno. |
| `data/` | Los dos CSV de entrada. |
| `figuras/01_kernels.png` a `figuras/04_prediccion_vs_real.png` | Figuras de la Parte 1. |
| `figuras/05_grid_vs_random.png` | Comparación de RMSE de agosto y tiempo de ambas búsquedas. |
| `resultados/parte2_experimentos.csv` | Una fila por candidato, con parámetros, errores, tiempo y `run_id`. |
| `resultados/parte2_comparacion.csv` | Resumen del mejor candidato de cada búsqueda. |
| `resultados/parte2_modelo.json` | Datos para localizar el modelo y conocer su esquema de entrada. |
| `tests/test_parte2.py` | Pruebas de integración y de renderizado Markdown. |
| `mlflow.db` y `mlartifacts/` | Metadatos de MLflow y archivos de modelos y figuras; se generan localmente. |
| `.venv/` | Entorno de Python local; no se comparte ni se sube a Git. |

## Entrega para la Parte 3

`resultados/parte2_modelo.json` contiene nombre, versión, URI, columnas y métricas
del modelo seleccionado. El registro se llama `svr_demanda` y guarda el pipeline
completo, incluido el escalador. Una primera ejecución crea la versión 1;
volver a ejecutar crea otra versión, sin sobrescribir las anteriores.

El compañero debe tomar la versión y la URI que aparezcan en ese JSON, sin asumir
que la versión siempre será 1. La muestra nueva debe respetar las 22 columnas
indicadas en `columnas`, incluidos los rezagos y las columnas del producto. Una
fecha sola no contiene todas las entradas que espera el modelo. El escalador ya
está guardado en el pipeline y no debe volver a ajustarse para predecir.

En el mismo equipo y carpeta se puede usar el registro existente. En otro equipo,
hay que preparar el entorno y ejecutar las Partes 1 y 2 una vez para generar el
registro local; después, la función de la Parte 3 debe cargarlo sin entrenar de
nuevo. Copiar el JSON no copia el modelo entrenado: solo contiene sus metadatos,
y las rutas del registro actual corresponden a este equipo.

Para compartir el proyecto se necesitan el notebook, los datos, las dependencias,
este README y los resultados exportados. La función `predecir(muestra, version)`
y la explicación con SHAP deben incorporarse en la Parte 3; no están implementadas.

## Comprobación y exportación

Desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m nbconvert --to html practica2_svm.ipynb
```

Las dos pruebas cubren lo siguiente:

- La integración ejecuta búsquedas con datos pequeños en un directorio temporal.
  Comprueba la separación temporal, parámetros, métricas y figuras de cada
  candidato, los archivos exportados y la carga de la versión 1. Verifica que el
  pipeline cargado reproduzca las predicciones del entrenado y que no se haya
  implementado la función de la Parte 3.
- La prueba de Markdown renderiza todas las celdas de texto y comprueba que las
  negritas y el código en línea no queden como marcas visibles. También revisa
  las listas de conclusiones y referencias.

La exportación a HTML usa las salidas que ya tiene el notebook; no vuelve a
entrenar. Hay que ejecutarlo antes si se quieren publicar resultados nuevos.
Las pruebas pequeñas no sustituyen esa ejecución completa con los datos reales.

## Problemas habituales

| Problema | Qué revisar |
| --- | --- |
| Falta `mlflow`, `sklearn` u otra librería | Instalar `requirements.txt` con el Python de `.venv` y seleccionar ese mismo kernel. |
| Aparece `NameError` con `X_tr` u otra variable | Ejecutar la Parte 1 antes de la Parte 2. |
| MLflow abre sin mostrar las ejecuciones | Comprobar que el comando se inició en la carpeta del proyecto y usa su `mlflow.db`; ejecutar la Parte 2 si el registro todavía no existe. |
| El modelo aparece con versión 2 o superior | Es normal tras repetir el entrenamiento; consultar la versión actual en el JSON de entrega. |
| Se ven asteriscos o comillas invertidas en el texto | Recargar el notebook actualizado y renderizar las celdas Markdown. Debe conservarse una línea en blanco entre cada cierre `</div>` y el texto Markdown siguiente. |

## Referencia visual

El notebook sigue el formato del [cuaderno de SVM de Remigio Hurtado](https://github.com/Remigiohurtadoins/MachineLearning-Engineering-Production-Portfolio/blob/main/14.SVM%20Fine%20Tuning%20-%20Loan%20approval%20-%20Financial%20Institution/SVM_OptimizacionConPipeline.ipynb):
título entre bordes azules, encabezados gris oscuro, separadores naranjas e índice
con enlaces. La redacción de la Parte 2 se revisó con Humanizer; los cálculos y los
resultados guardados se conservaron.
