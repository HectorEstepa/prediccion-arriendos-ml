# Correspondencia con el HTML principal original

El proyecto recupera los análisis de `super.html` usando Plotly y datos calculados desde el CSV. La información que antes estaba simulada se calcula de nuevo; no se reutilizan números escritos manualmente como resultados experimentales.

| Contenido original | Ubicación en la versión actual |
| --- | --- |
| Métricas y cambio de modelo | Comparación de modelos: MAE de validación y prueba, tabla con RMSE y R²; selector del gráfico para los cinco estimadores. |
| Valores reales frente a predichos | Predicciones y valores reales; 581 anuncios de prueba para cada modelo. |
| Gráfico de residuos | Debajo de la dispersión; mismo modelo y datos, convención real menos estimado. |
| Impacto de variables numéricas de RL | Interpretación de los modelos → Regresión lineal; coeficientes convertidos a unidades originales. |
| Impacto de comuna de RL | Coeficientes de todas las comunas respecto de una referencia explícita. |
| Top 10 comunas caras y baratas | Comparación de comunas; mediana observada del CSV filtrado. El hover muestra el número de anuncios. |
| Comparador de comunas | Formulario con tres comunas y características comunes; precio estimado por los cinco modelos. Los efectos ajustados de RL siguen disponibles en su gráfico de coeficientes. |
| Distribución de precios por comuna | Diagramas de caja de las tres comunas elegidas, con precios reales del CSV. |
| Importancia numérica de RF | Interpretación de los modelos → Random Forest. |
| Top 10 comunas importantes de RF | Importancia de las columnas de comuna en el bosque. |
| Matriz de confusión KNN | Clasificación y vecinos de KNN; conteos exclusivamente de prueba. |
| ROC KNN | Misma sección; probabilidades reales del clasificador y AUC. |
| Impacto de características KNN | Importancia por permutación: seleccionar KNN en el selector de modelos. También disponible para RL, RF y RN. |
| Proyección KNN y vecinos | Superficie y gastos de entrenamiento + validación, clases reales; al estimar se añaden la propiedad y sus cinco vecinos calculados con todas las variables. |
| Rendimiento según K | Clasificación y vecinos de KNN; evaluación en validación con distintos K y umbral calculado solo en entrenamiento. K=5 se mantiene fijado. |
| Explicación del algoritmo KNN | Texto explicativo en la sección KNN y descripción del modelo junto al estimador. |
| Simulador y clasificador de propiedades | Estimador conectado al servidor, con precio del modelo elegido y clasificación binaria independiente. |
| Información de la RN, presente en los otros HTML | Arquitectura entrenada, curva de pérdida, descripción de parámetros y métricas comparables con los demás modelos. |

La comparación de comunas mantiene todas las características excepto la comuna; no identifica un efecto causal. Las importancias por permutación se calculan en prueba después de seleccionar el modelo y se muestran como diagnóstico, sin usarlas para volver a seleccionar parámetros. Los precios de compraventa siguen fuera del alcance del conjunto.
