# Práctica 2: SVR y MLflow

El proyecto estima la demanda diaria de productos con SVR, una SVM para regresión.
El trabajo está en `practica2_svm.ipynb`; `practica2_svm.html` permite consultar los
resultados guardados sin ejecutar Python. Las tres partes del cuaderno están
implementadas: modelo base, optimización y predicción con interpretación.

## Contenido

- [Parte 1: preparación de datos, SVR y evaluación](#parte1)
- [Parte 2: optimización con MLflow](#parte2)
- [Parte 3: predicción e interpretación con SHAP](#parte3)

## Estado del trabajo

| Parte | Estado | Contenido |
| --- | --- | --- |
| Parte 1: datos y SVR | Terminada | Preparación de variables, división temporal, comparación de kernels y ajuste de C y épsilon. |
| Parte 2: optimización con MLflow | Terminada y comprobada | GridSearch, RandomSearch, registro de experimentos y del mejor modelo, comparación de RMSE y tiempo. |
| Parte 3: predicción e interpretación | Implementada y comprobada | Carga por versión, validación de entradas, dos escenarios ilustrativos y explicaciones SHAP. |

### Lo que ya está hecho

- [x] Reutilizar los datos de la práctica anterior y convertir la fecha en una variable numérica.
- [x] Separar entrenamiento, validación de agosto y prueba de septiembre, sin barajar las fechas.
- [x] Comparar los kernels lineal, RBF y polinomial en la Parte 1.
- [x] Estudiar por separado C y épsilon, incluida la diferencia entre error de entrenamiento y validación.
- [x] Evaluar el modelo base con RMSE y MAE y compararlo con las líneas base y la LSTM anterior.
- [x] Ejecutar GridSearch con 42 combinaciones y RandomSearch con 20, usando la misma validación.
- [x] Registrar en MLflow los parámetros, RMSE de entrenamiento y validación, tiempo de ajuste y figura de cada candidato.
- [x] Comparar ambas búsquedas en una figura y exportar sus resultados a CSV.
- [x] Guardar el pipeline completo, con su escalador, como `models:/svr_demanda/1`.
- [x] Ejecutar el cuaderno completo, comprobar la carga del modelo y actualizar el HTML.
- [x] Crear `predecir(muestra, version)` sin reentrenar el modelo ni el escalador.
- [x] Probar dos escenarios, normal y feriado, con sus 22 variables preparadas.
- [x] Explicar ambos resultados con SHAP y comprobar que los aportes reconstruyen las predicciones.
- [x] Incorporar conclusiones, referencias de SHAP y exportaciones de la Parte 3.

### Resultado guardado de la Parte 1

El modelo base elegido es SVR RBF, con C = 10 y épsilon = 0,1. Obtiene RMSE 3,58
en validación y, después del ajuste con entrenamiento y agosto, RMSE 3,21 y MAE
2,29 en septiembre. Los pasos y las comparaciones están en la explicación de la
Parte 1.

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

<a id="parte1"></a>

## Cómo funciona la Parte 1

Esta etapa prepara los datos y establece un modelo base antes de la búsqueda
conjunta de la Parte 2. Sigue las secciones 1.1 a 1.6 del notebook.

### 1.1 Variables de entrada

Se cargan las ventas y el inventario, se ordenan las observaciones por producto
y fecha y se calcula `stock_cierre` como stock inicial menos cantidad vendida.
La función `crear_variables` prepara:

- Fecha como timestamp, día de la semana, mes y fin de semana.
- Feriados de Ecuador, vísperas de feriado, días de pago y periodo de Navidad.
- Venta del día anterior, venta de hace siete días y media de las siete ventas anteriores.
- Stock de cierre del día anterior y columnas indicadoras del producto.

Los rezagos se calculan por producto y solo usan días anteriores. Se descartan
las filas sin suficiente historial; quedan 3.580 muestras con 22 entradas. La
salida que aprende el modelo es `cantidad_vendida`.

### 1.2 División por fechas

Se reservan 2.980 filas para entrenamiento, 300 del 2 al 31 de agosto de 2026
para validación y 300 de septiembre para prueba. No se barajan las fechas.
Durante la comparación de configuraciones, el escalador se ajusta únicamente
con entrenamiento. Agosto se utiliza para medir el error de validación.

### 1.3 Comparación de kernels

Cada modelo combina `StandardScaler` y `SVR` en un pipeline. Se prueban los tres
kernels con C = 10 y épsilon = 0,1:

| Kernel | RMSE de entrenamiento | RMSE de validación |
| --- | ---: | ---: |
| Lineal | 3,88 | 4,24 |
| RBF | 2,71 | 3,58 |
| Polinomial, grado 3 | 2,66 | 3,63 |

RBF obtiene el menor error de validación y se utiliza en los siguientes ajustes.
La [figura de kernels](figuras/01_kernels.png) muestra los errores de ambos conjuntos.

### 1.4 Efecto de C

Se prueban C = 0,1; 0,3; 1; 3; 10; 30; 100; 300 y 1000. Con C = 0,1, el RMSE de
validación es 8,02. El menor error se encuentra en C = 10, con RMSE 3,58. Con
C = 1000, el error de entrenamiento baja a 1,98, pero el de validación sube a 4,45.
Esa separación muestra el sobreajuste que se observa en este experimento.

La [curva de C](figuras/02_curva_C.png) permite revisar este cambio sin utilizar
el conjunto de prueba para elegir el parámetro.

### 1.5 Efecto de épsilon

Con C = 10 se prueban épsilon = 0,1; 0,5; 1; 2; 4 y 8. Se comparan el RMSE de
validación y el porcentaje de vectores de soporte:

- Épsilon = 0,1 obtiene el menor RMSE, 3,58, y utiliza el 95,87 % de las filas como vectores de soporte.
- Épsilon = 2 obtiene RMSE 3,67 y utiliza el 41,74 %, con menos vectores de soporte y algo más de error.

Se elige épsilon = 0,1 por su menor error. La [figura de épsilon](figuras/03_epsilon.png)
muestra el compromiso entre error y cantidad de vectores de soporte.

### 1.6 Evaluación del modelo base

Una vez elegidos RBF, C = 10 y épsilon = 0,1, se ajusta el pipeline con
entrenamiento y agosto. Después se calculan las métricas en septiembre:

| Modelo o referencia | RMSE de prueba | MAE de prueba |
| --- | ---: | ---: |
| Venta del día anterior | 6,47 | 4,51 |
| Venta del mismo día de la semana anterior | 4,39 | 3,29 |
| LSTM de la Práctica 1 | 4,40 | 3,10 |
| SVR RBF de la Parte 1 | 3,21 | 2,29 |

Las métricas de la LSTM se toman de la práctica anterior; esa red no se entrena
en este notebook. La [comparación de predicción y demanda real](figuras/04_prediccion_vs_real.png)
muestra dos productos durante septiembre.

También se calcula el stock de cierre estimado restando la demanda predicha al
stock inicial y limitando el resultado a cero como mínimo. Su RMSE guardado es
3,21. El SVR sigue aprendiendo demanda: no hay un segundo modelo entrenado
directamente para stock.

Estos son los resultados del modelo base. La Parte 2 busca combinaciones de
parámetros de forma conjunta y registra sus experimentos y el modelo elegido en
MLflow.

<a id="parte2"></a>

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

<a id="parte3"></a>

## Cómo funciona la Parte 3

### Carga por versión y función de predicción

La sección 3.1 lee el nombre y la versión de `resultados/parte2_modelo.json` y
configura el registro SQLite de la carpeta actual. La sección 3.2 define
`predecir(muestra, version)`:

- `muestra` es un DataFrame de exactamente una fila, con las columnas del modelo.
- `version` es un entero positivo. Una cadena, un booleano o un número no entero
  se rechazan; una versión que no existe conserva el error de MLflow.
- Las columnas pueden llegar en otro orden: la función usa el orden del pipeline.
  Rechaza columnas faltantes, adicionales o repetidas, valores no numéricos,
  ausentes o infinitos y codificaciones que no identifican un único producto.
- Carga `models:/svr_demanda/<version>` y devuelve la demanda estimada como número
  decimal. No redondea ni recorta la salida, no entrena y no reajusta el escalador.

Se usan modelos locales generados por el proyecto. Para llamar solo a la función
en un kernel nuevo bastan las secciones 3.1 y 3.2, el registro local y una muestra
ya preparada. Los casos y SHAP también necesitan los datos de la Parte 1.

Después de ejecutar 3.1 y 3.2, se puede reutilizar una muestra exportada:

```python
muestras = pd.read_csv("resultados/parte3_muestras.csv")
predecir(muestras.drop(columns="caso").iloc[[0]], VERSION_MODELO)
```

Se elimina `caso` porque es una etiqueta del ejemplo, no una entrada del modelo.

### Casos normal y feriado

Los dos ejemplos corresponden al producto 103665, Pan de Molde Blanco. Usan los
valores de las últimas siete observaciones disponibles y desplazan sus fechas
al periodo anterior de cada escenario. `crear_variables`, de la Parte 1, calcula
el calendario y los rezagos; las columnas de los otros productos quedan en cero.
La fila nueva no tiene venta observada ni stock de cierre observado.

| Escenario ilustrativo | Fecha | Indicador de feriado | Demanda estimada con versión 1 |
| --- | --- | ---: | ---: |
| Normal | 1 de octubre de 2026 | 0 | 38,982 |
| Feriado | 9 de octubre de 2026 | 1 | 49,949 |

Son muestras simuladas, no ventas observadas de esas fechas. Comparten valores
históricos para ilustrar la función, pero también cambian la fecha y el calendario;
la diferencia no puede atribuirse únicamente al feriado. No se calcula RMSE con
estos casos porque no se dispone de su demanda real.

### Explicación con SHAP

Se aplica `shap.KernelExplainer` al pipeline completo, con 30 filas de entrenamiento
como fondo, 300 muestras de aproximación por caso y semilla 42. El valor base es
la predicción media sobre ese fondo. Los aportes positivos aumentan la predicción
respecto a la base y los negativos la reducen. La suma de base y aportes debe
coincidir con la salida de `predecir`, con tolerancia numérica de 1e-5.

Cada figura muestra los 11 aportes de mayor magnitud y agrupa los 11 restantes.
Los 22 aportes de cada caso se exportan a CSV y el notebook comenta los tres
principales. SHAP explica al modelo respecto al fondo elegido; no demuestra
causalidad. Las variables de calendario, rezagos y producto pueden estar
relacionadas, por lo que los resultados deben leerse junto con esta limitación.
La evaluación de calidad sigue siendo la prueba de septiembre de la Parte 2.

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
kernel vacío. La Parte 3 utiliza el modelo registrado por la Parte 2.

También se puede ejecutar todo el cuaderno desde la terminal:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 practica2_svm.ipynb
```

Este comando vuelve a entrenar, actualiza las salidas del notebook y agrega nuevas
ejecuciones y una nueva versión del modelo en MLflow. Para leer los resultados
existentes basta con abrir el HTML; no es necesario repetir el entrenamiento.

Para repetir solo la Parte 3 sin entrenar, en un kernel nuevo ejecutar la carga
de librerías y datos al inicio de la Parte 1, la sección 1.1 de variables y la 1.2
de división temporal. Después ejecutar 3.1 a 3.5, sin pasar por 1.3 a 1.6 ni por
la Parte 2. Así se preparan los datos para los ejemplos y SHAP, pero se carga la
versión que ya existe en el registro.

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
| `practica2_svm.ipynb` | Código, explicaciones y salidas de las tres partes. |
| `practica2_svm.html` | Exportación del cuaderno con las salidas guardadas. |
| `requirements.txt` | Dependencias para preparar el entorno. |
| `data/` | Los dos CSV de entrada. |
| `figuras/01_kernels.png` a `figuras/04_prediccion_vs_real.png` | Figuras de la Parte 1. |
| `figuras/05_grid_vs_random.png` | Comparación de RMSE de agosto y tiempo de ambas búsquedas. |
| `figuras/06_shap_normal.png` y `figuras/07_shap_feriado.png` | Aportes SHAP de los dos escenarios. |
| `resultados/parte2_experimentos.csv` | Una fila por candidato, con parámetros, errores, tiempo y `run_id`. |
| `resultados/parte2_comparacion.csv` | Resumen del mejor candidato de cada búsqueda. |
| `resultados/parte2_modelo.json` | Datos para localizar el modelo y conocer su esquema de entrada. |
| `resultados/parte3_predicciones.csv` | Fechas, producto, indicador de feriado, versión y predicciones de los dos casos. |
| `resultados/parte3_muestras.csv` | Las dos muestras con sus 22 variables y etiqueta del caso. |
| `resultados/parte3_shap.csv` | Los 44 aportes, con valor de entrada, base SHAP y predicción. |
| `tests/test_parte2.py` | Pruebas de integración y de renderizado Markdown. |
| `tests/test_parte3.py` | Carga por versión, validación, ausencia de reentrenamiento, escenarios y SHAP. |
| `mlflow.db` y `mlartifacts/` | Metadatos de MLflow y archivos de modelos y figuras; se generan localmente. |
| `.venv/` | Entorno de Python local; no se comparte ni se sube a Git. |

## Modelo y versiones entre las tres partes

`resultados/parte2_modelo.json` contiene nombre, versión, URI, columnas y métricas
del modelo seleccionado. El registro se llama `svr_demanda` y guarda el pipeline
completo, incluido el escalador. Una primera ejecución crea la versión 1;
volver a ejecutar crea otra versión, sin sobrescribir las anteriores.

La Parte 3 toma el nombre y la versión que aparecen en ese JSON, sin asumir
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
este README y los resultados exportados. El JSON de entrega no incluye los
archivos del modelo; la Parte 3 necesita que su versión exista en MLflow.

## Comprobación y exportación

Desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m nbconvert --to html practica2_svm.ipynb
```

Las cinco pruebas cubren lo siguiente:

- La integración ejecuta búsquedas con datos pequeños en un directorio temporal.
  Comprueba la separación temporal, parámetros, métricas y figuras de cada
  candidato, los archivos exportados y la carga de la versión 1. Verifica que el
  pipeline cargado reproduzca las predicciones del entrenado. La prueba de la
  Parte 2 se detiene antes de la Parte 3 para comprobar ambas etapas por separado.
- La prueba de Markdown renderiza todas las celdas de texto y comprueba que las
  negritas y el código en línea no queden como marcas visibles. También revisa
  las listas de conclusiones y referencias.
- Las pruebas de la Parte 3 comprueban la existencia de la función y los ejemplos,
  y ejecutan una integración con dos versiones registradas. Comprueban que elegir
  otra versión cambia el modelo, que los datos inválidos se rechazan y que no se
  llama a `fit` del SVR ni del escalador. También verifican los indicadores de
  feriado, la reconstrucción SHAP, las figuras, los CSV y la integridad de los datos.

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

### SQLAlchemy y Control de aplicaciones de Windows

Si Windows bloquea una extensión nativa de SQLAlchemy, se puede instalar su
variante oficial en Python puro. En este equipo se comprobó con SQLAlchemy 2.1.8:

```powershell
$env:DISABLE_SQLALCHEMY_CEXT = "1"
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps --no-binary=sqlalchemy "SQLAlchemy==2.1.8"
```

Esta instalación afecta solo a `.venv` y evita usar las extensiones compiladas de
SQLAlchemy; no cambia las políticas de seguridad de Windows. Reiniciar el kernel
después de instalar. La opción corresponde al procedimiento oficial para
[instalar SQLAlchemy sin extensiones Cython](https://docs.sqlalchemy.org/en/21/intro.html#building-the-cython-extensions).

## Referencia visual

El notebook sigue el formato del [cuaderno de SVM de Remigio Hurtado](https://github.com/Remigiohurtadoins/MachineLearning-Engineering-Production-Portfolio/blob/main/14.SVM%20Fine%20Tuning%20-%20Loan%20approval%20-%20Financial%20Institution/SVM_OptimizacionConPipeline.ipynb):
título entre bordes azules, encabezados gris oscuro, separadores naranjas e índice
con enlaces. La redacción de las Partes 2 y 3 se revisó con Humanizer; los cálculos
y resultados guardados de las dos primeras partes se conservaron.
