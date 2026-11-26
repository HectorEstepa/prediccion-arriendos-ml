# Versión original del trabajo

Esta carpeta conserva sin modificar los scripts, HTML, JSON de configuración y figuras recibidos. No contiene el CSV con nombres y direcciones ni los modelos pickle originales. La copia apta para publicar está en `../data/propiedades.csv` y conserva las variables necesarias para probar los algoritmos.

Para ver la página original, abre `super.html` en un navegador. Los HTML de cada modelo están en sus respectivas carpetas. Algunos resultados y gráficos originales son simulaciones o métricas calculadas con datos de entrenamiento, tal como se documenta en `../docs/REVISION.md`. Para consultar la evaluación corregida y los gráficos calculados con datos reales, utiliza la aplicación principal.

Para preparar los datos que esperan los scripts, ejecuta desde la raíz del repositorio:

```powershell
python scripts/preparar_original.py
```

Después, por ejemplo:

```powershell
cd original/RL
python RL.py
```

RL, RF y KNN usan pandas, numpy y scikit-learn; RF también requiere matplotlib. RN requiere además polars, matplotlib y networkx. Estas dependencias adicionales se pueden instalar con `python -m pip install -r original/requirements.txt`. Los scripts originales tienen interacción por consola y algunos abren gráficos. Se conservan como material histórico y no forman parte de las pruebas automatizadas del paquete principal.
