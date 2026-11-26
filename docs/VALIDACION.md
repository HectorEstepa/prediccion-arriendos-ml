# Comprobaciones realizadas

## Datos y entrenamiento

- Entrenamiento completo a partir del CSV recibido con Python 3.13 y scikit-learn 1.7.2.
- 3.247 filas originales; 168 duplicados exactos eliminados; 241 filas adicionales fuera del alcance o inválidas; 2.838 filas utilizadas.
- Partición: 1.788 entrenamiento, 469 validación y 581 prueba; sin grupos compartidos.
- Selección por MAE de validación: red neuronal.
- MAE de prueba del modelo seleccionado: 68.154,28 CLP; R² de prueba: 0,7401.
- Los detalles de los cinco estimadores están en `reports/evaluacion.json` y `reports/EVALUACION.md`.

## Pruebas automatizadas

`python -m pytest -q`: **6 pruebas aprobadas**. Verifican limpieza, esquema, partición reproducible sin grupos compartidos, selección por validación, métricas reales, predicciones HTTP iguales a la tubería y manejo de entradas inválidas o falta de modelos. Usan datos sintéticos.

Se observó una advertencia de deprecación en la integración instalada de Starlette con httpx TestClient; no produjo fallos. El entrenamiento también emitió una advertencia del entorno al detectar núcleos físicos; utilizó los núcleos lógicos disponibles.

`node --check web/app.js`: sintaxis válida.

## Navegador

Se abrió la aplicación en Edge sin interfaz gráfica mediante Playwright y se comprobó:

- Carga de las cinco filas de métricas y los 581 puntos reales del gráfico.
- Predicción mediante los cuatro algoritmos y la referencia; sin errores JavaScript.
- Predicciones idénticas al repetir la misma solicitud.
- Invalidación del precio mostrado cuando se modifican los campos.
- Manejo de errores de red/HTTP y bloqueo de entradas cuando faltan modelos.
- Ausencia de desbordamiento horizontal de la página en anchos de 320, 390, 768 y 1.440 píxeles. La tabla conserva desplazamiento horizontal propio cuando se necesita.
- Capturas de escritorio y móvil revisadas visualmente: `vista-escritorio.png` y `vista-movil.png`.

## Revisión del diseño visual

La interfaz se reorganizó como un informe de resultados, con fondo blanco, tipografía sencilla, líneas de separación y navegación por secciones. Los gráficos siguen un estilo editorial similar a Datawrapper, implementado localmente sin cuentas externas.

Se comprobaron las cinco barras del MAE, el cambio independiente de modelo en el gráfico de dispersión y la adaptación de los ejes al ancho de pantalla. Las descargas CSV se contrastaron con el reporte: cinco modelos y 581 predicciones del modelo elegido, sin direcciones ni títulos de anuncios. Se repitieron las comprobaciones de estimación, errores de red y los cuatro anchos de pantalla tras el rediseño.

El servidor usado para la comprobación fue local. No se publicó el proyecto ni se enviaron datos a GitHub.

## Recuperación de los análisis y Plotly

La versión actual reemplaza los gráficos locales anteriores por Plotly 4.1.1, distribuido junto al proyecto. Se verificaron **18 gráficos**, los cuatro modelos y la referencia, y los análisis recuperados del HTML principal. La tabla de correspondencia está en `CONTENIDO-HTML.md`.

Las seis pruebas de Python volvieron a aprobarse, ampliadas para comprobar las medianas y cantidades por comuna, la suma de importancias del bosque, la equivalencia de los coeficientes lineales en unidades originales con la tubería, los vecinos ordenados y las comparaciones entre comunas con cada modelo. Las comunas repetidas o desconocidas se rechazan en la API.

En Edge se comprobó que cada dispersión contiene las predicciones del reporte, que los residuos usan real menos estimado, que el comparador coincide con `/api/compare` y que los vecinos de una consulta son cinco. Se verificaron cambios de campos, errores de red, descargas CSV y los anchos 320, 390, 768 y 1.440 px. No hubo errores JavaScript. Las capturas actuales son `vista-escritorio.png`, `vista-movil.png`, `comparacion-comunas.png`, `comunas-movil.png` y `analisis-modelos.png`.

## Organización visual y navegación

Se actualizaron la jerarquía tipográfica, las superficies, los márgenes, los controles y las agrupaciones de gráficos. Se conservan los 18 gráficos y las comparaciones de datos. Se repitió la comprobación completa del navegador en los cuatro tamaños, sin errores JavaScript ni desbordamiento horizontal de la página.

La navegación fija señala la sección actual con `aria-current`. Se comprobó que los enlaces a comunas, interpretación y Random Forest dejan el encabezado de destino visible debajo de la barra fija, y que la navegación móvil puede desplazarse horizontalmente. Las vistas generales actuales están en `vista-general.png` y `vista-general-movil.png`.

## Preparación de la descarga pública

El entrenamiento se repitió con `data/propiedades.csv`, sin títulos ni direcciones. Conservó las 3.247 filas iniciales, 168 duplicados exactos y la partición de 1.788/469/581 filas; las métricas se mantuvieron. Se añadieron los 17 scripts, HTML, JSON y figuras originales, comprobando que la copia local es idéntica byte a byte. Los CSV originales completos y modelos pickle no se incluyen.

`python -m pytest -q`: **7 pruebas aprobadas**, incluida la equivalencia de auditoría y particiones entre datos originales y públicos, y la ausencia de títulos y direcciones en la copia pública.
