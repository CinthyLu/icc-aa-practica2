# Práctica 2: Predicción de demanda con SVM

**Aprendizaje Automático · Universidad Politécnica Salesiana**
Desarrollado por: Daniel Guanga, Alex Guamán y Cinthya Ramón · Docente: Ing. Remigio Hurtado

Predicción de la demanda diaria de 10 productos (escenario sintético de La Favorita) con **SVR**, la SVM para regresión. El modelo parte de una versión base, se mejora con fine-tuning y cada prueba queda registrada en **MLflow**. El resultado se compara con la LSTM de la Práctica 1.

## Contenido de la carpeta

| Archivo o carpeta | Contenido |
| --- | --- |
| `practica2_svm.ipynb` | Cuaderno de la práctica |
| `practica2_svm.html` | Cuaderno ejecutado, para verlo sin Python |
| `data/` | `sales_history.csv` (ventas y stock diario) e `inventory.csv` (productos) |
| `figuras/` | 7 figuras generadas por el cuaderno |
| `resultados/` | CSV de los experimentos y datos del modelo registrado |
| `requirements.txt` | Librerías necesarias |

`mlflow.db` y `mlartifacts/` se crean al ejecutar el cuaderno y no se suben al repositorio.

## Estructura del cuaderno

**Parte 1: Datos y SVR**
- 22 variables por fila: fecha como timestamp, calendario (día, mes, feriados de Ecuador, víspera, día de pago, Navidad), rezagos de venta, stock del día anterior y producto.
- División por fechas: train hasta el 1 de agosto de 2026, validación del 2 al 31 de agosto y prueba en septiembre (300 filas cada una).
- Comparación de kernels y efecto de C y épsilon (subajuste y sobreajuste).

**Parte 2: Fine-tuning con MLflow**
- Modelo base: SVR lineal con valores por defecto.
- GridSearch (42 combinaciones) y RandomSearch (20) sobre kernel, C, gamma y épsilon.
- Registro del mejor modelo como `models:/svr_demanda/1`.
- Comparación base → ajuste manual → fine-tuning y ejemplos de real frente a predicción.

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

## Ejecución

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace practica2_svm.ipynb
```

Cada ejecución completa crea una nueva versión de `svr_demanda` en el registro.

## Interfaz de MLflow

MLflow guarda el historial en la carpeta del proyecto (`mlflow.db`); no necesita cuenta. Para ver las ejecuciones:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Luego se abre http://127.0.0.1:5000 y se entra al experimento `practica2_svr`. Ahí aparecen las ejecuciones `modelo_base`, `ajuste_manual`, `GridSearch`, `RandomSearch` (con un candidato por cada combinación) y `mejor_modelo`. Con **Compare** se grafican los parámetros frente al RMSE, y en **Models** está `svr_demanda` con sus versiones.

## Pendiente

- Random Forest y la explicación de predicciones todavía no se han visto en clase; la versión anterior con SHAP está guardada en `_archivo`.
- Ejercicio de árboles de decisión (Information Gain) de la guía.
