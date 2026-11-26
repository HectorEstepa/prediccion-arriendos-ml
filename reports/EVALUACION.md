# Evaluación reproducible

Modelo seleccionado por validación: **Red neuronal**.

Los modelos finales se ajustaron con entrenamiento + validación; estas métricas corresponden a prueba.

| Modelo | MAE validación (CLP) | MAE prueba (CLP) | RMSE prueba (CLP) | R² prueba |
| --- | ---: | ---: | ---: | ---: |
| Referencia: mediana | 150267 | 150772 | 270884 | -0.0840 |
| Regresión lineal | 63546 | 73021 | 130779 | 0.7473 |
| K vecinos cercanos | 63911 | 70197 | 122851 | 0.7770 |
| Random Forest | 62250 | 71220 | 126013 | 0.7654 |
| Red neuronal | 61726 | 68154 | 132647 | 0.7401 |

Filas: {'train': 1788, 'validation': 469, 'test': 581}. Grupos: {'train': 1507, 'validation': 377, 'test': 472}.

El MAE es un promedio de errores, no un intervalo de confianza individual.
Los resultados describen este conjunto histórico; falta fecha y fuente verificable del scraping.
