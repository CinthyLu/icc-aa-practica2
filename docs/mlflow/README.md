# Historial compartido de MLflow

Copia de la ejecución del 12 de noviembre de 2026: 132 ejecuciones terminadas
y las versiones 1 y 2 de `svr_demanda`. Incluye la base `historial.sqlite` y
152 archivos de artefactos en `artifacts/`, aproximadamente 3,8 MiB en total.

## Abrir después de clonar o hacer pull

Preparar el entorno según el [README principal](../../README.md). Desde la
carpeta del proyecto, ejecutar en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///docs/mlflow/historial.sqlite --host 127.0.0.1 --port 5000
```

Abrir [MLflow compartido](http://127.0.0.1:5000), seleccionar **Model training**
y entrar a **practica2_svr**. Expandir las búsquedas para ver sus candidatos;
usar **All time** y **Load more** si la tabla no muestra todo el historial.
Las versiones se encuentran en **Model registry → svr_demanda**.

El comando inicia directamente MLflow; no entrena modelos ni usa un visor
adicional. Las rutas de los artefactos son relativas a la raíz del repositorio,
por lo que el comando debe ejecutarse desde esa carpeta, no desde `docs/mlflow/`.
Conservar la terminal abierta y usar **Ctrl+C** para detener el servidor.
La interfaz es local, no está publicada en internet.

Si el puerto 5000 está ocupado, detener el servidor anterior o elegir otro:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///docs/mlflow/historial.sqlite --host 127.0.0.1 --port 5001
```

## Archivos que se comparten

- `historial.sqlite`: experimentos, ejecuciones, métricas, parámetros y registro de modelos.
- `artifacts/`: figuras, CSV, modelos y sus metadatos. Mantener toda la carpeta.

Esta copia se generó con un respaldo consistente de SQLite, sin modificar la
base original. Solo se sustituyeron las rutas locales en la copia y en los
metadatos `MLmodel` y `parte2_modelo.json` de los artefactos. Las métricas y los
identificadores de las ejecuciones y versiones se conservaron.

## Diferencia frente al entrenamiento

El notebook sigue usando la base `mlflow.db` y `mlartifacts/` de la raíz, que
no se suben a Git. Su interfaz se abre en el puerto 5000. Esta copia compartida
es fija: ejecutar de nuevo el notebook no actualiza su historial automáticamente.
Usar la base compartida para consulta: editar o borrar ejecuciones desde la
interfaz puede modificar el archivo de Git. No se deben combinar bases SQLite
de distintos integrantes mediante Git.

Se comprobó la copia con MLflow 3.18.0 y scikit-learn 1.9.1. Para cargar los
modelos entrenados conviene mantener esas versiones; sus dependencias también
figuran en los metadatos de cada modelo.

## Comprobación realizada

Se abrió la copia desde otra carpeta con espacios usando el comando directo
de MLflow. Se consultaron las 132 ejecuciones, se descargó una figura y se
cargaron ambas versiones para hacer predicciones de prueba, sin reentrenar.
Los archivos compartidos permanecieron sin cambios durante esa consulta.
