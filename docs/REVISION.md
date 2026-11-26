# Revisión del proyecto original

La carpeta original se conserva sin modificaciones. Esta versión reorganiza y corrige su implementación a partir del CSV y los archivos entregados. No utiliza los pickle originales: las tuberías se entrenan desde cero.

| Hallazgo comprobado | Corrección |
| --- | --- |
| RL y RF entrenaban y evaluaban con todos los datos. | Entrenamiento, validación y prueba por grupos; métricas finales fuera de entrenamiento. |
| KNN reservaba prueba, pero calculaba métricas de clasificación sobre el conjunto completo. | Métricas exclusivamente en prueba; clasificación separada de regresión. |
| KNN combinaba gastos en pesos con otras variables sin escalar. | Escalado numérico dentro de la tubería. |
| La RN ajustaba una red sobre precios en pesos sin transformar el objetivo. | Escalado del precio con TransformedTargetRegressor. |
| La opción de ensamble de RN calculaba pesos usando R² de prueba. | Se retira esa opción; selección por MAE de validación. |
| Las cuatro carpetas contenían el mismo CSV. | Una única copia local y una carga de datos común. |
| El CSV tenía 168 duplicados exactos y precios desde $1.000. | Eliminación de duplicados y auditoría de filtros; no se adivinan unidades. |
| `super.html` mostraba gráficos con Math.random y valores escritos manualmente. | Gráfico a partir de anuncios reales de prueba y métricas exportadas. |
| Las predicciones RF/KNN de la página eran fórmulas simuladas. | API conectada a modelos entrenados y preprocesamiento persistido. |
| Había múltiples HTML divergentes, sin un punto de entrada claro. | Una aplicación web responsiva con cuatro modelos y referencia. |
| Scripts dependían del directorio actual y pedían input al ejecutarse. | Rutas desde la raíz del paquete y CLI no interactiva. |
| Faltaban instrucciones, dependencias y pruebas. | README, dependencias fijadas, pruebas con datos sintéticos y CI. |

La interfaz conserva el objetivo educativo de comparar modelos, pero se reconstruye para evitar presentar simulaciones como resultados experimentales. Las versiones originales siguen disponibles en la carpeta de origen del usuario.

## Límites pendientes

- No existe el código del scraping, según confirmación del autor. Tampoco hay fuente, fecha ni licencia verificables de los datos.
- No se puede confirmar si los precios se normalizaron correctamente desde CLP/UF ni reconstruir los anuncios originales.
- Las agrupaciones por dirección son una aproximación conservadora, no identificadores confiables de propiedad.
- No se dispone de etiquetas de compraventa, tipo de propiedad, fecha de publicación o estado de disponibilidad.
- Las métricas no aseguran desempeño actual, causalidad ni generalización fuera del dominio observado.

Estos límites se documentan en la aplicación y el README. No se crean fuentes, fechas ni resultados para completar lo que falta.
