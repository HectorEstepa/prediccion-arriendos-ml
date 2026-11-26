"""Prepara el CSV público con el nombre esperado por los scripts originales."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
for model in ["KNN", "RF", "RL", "RN"]:
    destination = root / "original" / model / "DatosRevisadosLimpios.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "data/propiedades.csv", destination)
print("Datos de prueba preparados para las cuatro carpetas originales.")
