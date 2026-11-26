"""Quita títulos y direcciones conservando grupos, orden y duplicados exactos.

Uso: python scripts/preparar_datos_publicos.py --source ruta/al/original.csv
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arriendos.data import ROOT, normalize


def prepare(source, output):
    df = pd.read_csv(source, encoding="utf-8")
    df.columns = df.columns.str.strip()
    keys = df.Comuna.map(normalize) + "|" + df.Direccion.map(normalize)
    mapping = {key: f"grupo{i:05d}" for i, key in enumerate(sorted(keys.unique()), 1)}
    public = df[["Precio ($)", "Gastos_Comunes ($)", "Comuna", "Metros (m²)",
                 "Habitaciones", "Baños"]].copy()
    public["Grupo"] = keys.map(mapping)
    # Distingue anuncios con iguales características sin conservar sus títulos.
    records, _ = pd.factorize(pd.MultiIndex.from_frame(df), sort=False)
    public["Registro"] = [f"registro{i + 1:05d}" for i in records]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(output, index=False, encoding="utf-8")
    print(f"Guardadas {len(public)} filas sin títulos ni direcciones en {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/propiedades.csv")
    args = parser.parse_args()
    prepare(args.source, args.output)
