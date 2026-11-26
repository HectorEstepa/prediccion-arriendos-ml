"""Carga, validación y partición de anuncios sin compartir direcciones."""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["metros", "gastos_comunes", "habitaciones", "banos"]
FEATURES = NUMERIC + ["comuna"]
TARGET = "precio"
RENAME = {
    "Precio ($)": TARGET, "Gastos_Comunes ($)": "gastos_comunes",
    "Metros (m²)": "metros", "Habitaciones": "habitaciones",
    "Baños": "banos", "Comuna": "comuna", "Direccion": "direccion",
}
# Alcance declarado antes de evaluar; coincide con el filtro de la RN original.
PRICE_RANGE = (100_000, 2_000_000)
AREA_RANGE = (15, 200)


def normalize(value):
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)
    group_column = "Grupo" if "Grupo" in df.columns else "direccion"
    required = FEATURES + [TARGET, group_column]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    audit = {"raw_rows": len(df)}
    # La identidad de la fila original cuenta antes de convertir columnas.
    unique = df.drop_duplicates().copy()
    audit["exact_duplicates_removed"] = len(df) - len(unique)
    df = unique
    for column in NUMERIC + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["comuna", group_column]:
        df[column] = df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    finite = np.isfinite(df[NUMERIC + [TARGET]]).all(axis=1)
    valid = (
        finite & df[TARGET].between(*PRICE_RANGE) & df.metros.between(*AREA_RANGE)
        & df.gastos_comunes.ge(0) & df.habitaciones.ge(0) & df.banos.ge(1)
        & df.habitaciones.mod(1).eq(0) & df.banos.mod(1).eq(0)
        & df.comuna.notna() & df[group_column].notna()
        & df.comuna.ne("") & df[group_column].ne("")
    ).fillna(False)
    audit["out_of_scope_or_invalid_removed"] = int((~valid).sum())
    df = df.loc[valid].copy().reset_index(drop=True)
    df["group"] = df.comuna.map(normalize) + "|" + df[group_column].map(normalize)
    audit.update(clean_rows=len(df), address_groups=int(df.group.nunique()),
                 price_range_clp=list(PRICE_RANGE), area_range_m2=list(AREA_RANGE))
    if len(df) < 30 or df.group.nunique() < 10:
        raise ValueError("Se requieren al menos 30 filas válidas y 10 direcciones distintas.")
    return df, audit


def split_data(df, seed=42):
    """20% de grupos para prueba y 20% del resto para validación."""
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                              random_state=seed).split(df, groups=df.group))
    train_local, validation_local = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=seed + 1
    ).split(df.iloc[development], groups=df.iloc[development].group))
    return development[train_local], development[validation_local], test
