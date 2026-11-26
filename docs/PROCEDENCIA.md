# Procedencia del conjunto

- Archivo recibido: `DatosRevisadosLimpios.csv`, repetido en KNN, RF, RL y RN.
- Filas originales: 3.247.
- SHA-256: `0601c35204c9673ee1505a4be7d9a580f5e07c77f0c2ba3a0cb215394014828d`.
- Origen informado: scraping de anuncios de arriendo en un trabajo universitario.
- Código de obtención: no disponible, confirmado por el autor durante la revisión.
- Sitio, fecha de extracción, términos de uso y licencia: no disponibles.
- Moneda y periodicidad: CLP mensuales según las etiquetas y el objetivo de los scripts originales; sin verificación externa de cada anuncio.
- Dirección y nombre del anuncio: no se usan como predictores ni se exportan al reporte público. La dirección se utiliza localmente para agrupar la partición.

El CSV original completo queda excluido por `.gitignore`. La copia publicada en `data/propiedades.csv` elimina títulos y direcciones e introduce `Grupo` y `Registro`, identificadores sustitutos que conservan las particiones y los duplicados exactos. El script de conversión es `scripts/preparar_datos_publicos.py`. Se comprobó que ambas copias producen la misma auditoría y los mismos índices de entrenamiento, validación y prueba.

El hash exacto de la copia pública usada en el entrenamiento se genera automáticamente en `reports/evaluacion.json`; ese reporte es la referencia para comprobar una reproducción. No es el mismo hash del CSV privado original indicado arriba.
