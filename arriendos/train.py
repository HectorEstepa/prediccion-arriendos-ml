"""Entrenamiento reproducible: python -m arriendos.train."""
import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, roc_curve, precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, NUMERIC, ROOT, TARGET, load_data, split_data
from .diagnostics import commune_summary, explain_models, k_analysis

LABELS = {"baseline": "Referencia: mediana", "lineal": "Regresión lineal",
          "knn": "K vecinos cercanos", "random_forest": "Random Forest",
          "red_neuronal": "Red neuronal"}


def preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("comuna", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), ["comuna"]),
    ])


def models(seed=42):
    estimators = {
        "baseline": DummyRegressor(strategy="median"),
        "lineal": LinearRegression(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               random_state=seed, n_jobs=-1),
        "red_neuronal": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600,
                                   early_stopping=True, n_iter_no_change=25,
                                   random_state=seed), transformer=StandardScaler()),
    }
    return {name: Pipeline([("preprocess", preprocessor()), ("model", estimator)])
            for name, estimator in estimators.items()}


def metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def fit_recording_warnings(model, x, y):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return sorted(set(str(item.message) for item in recorded))


def train(data_path, artifact_dir, report_dir, seed=42):
    df, audit = load_data(data_path)
    train_idx, val_idx, test_idx = split_data(df, seed)
    x, y = df[FEATURES], df[TARGET]
    candidates = models(seed)
    validation, notes = {}, {}
    for name, model in candidates.items():
        print(f"Validando: {LABELS[name]}", flush=True)
        notes[name] = fit_recording_warnings(model, x.iloc[train_idx], y.iloc[train_idx])
        validation[name] = metrics(y.iloc[val_idx], model.predict(x.iloc[val_idx]))
    # El conjunto de prueba no interviene en la selección.
    selected = min(validation, key=lambda name: validation[name]["mae"])
    development = np.concatenate([train_idx, val_idx])
    fitted, test_metrics, predictions = {}, {}, {}
    for name, candidate in candidates.items():
        model = clone(candidate)
        notes[name] += fit_recording_warnings(model, x.iloc[development], y.iloc[development])
        prediction = model.predict(x.iloc[test_idx])
        if not np.isfinite(prediction).all():
            raise ValueError(f"El modelo {name} produjo predicciones no finitas.")
        test_metrics[name] = metrics(y.iloc[test_idx], prediction)
        predictions[name] = prediction
        fitted[name] = model

    # Experimento de clasificación separado de la estimación numérica.
    threshold = float(y.iloc[development].mean())
    classifier = Pipeline([("preprocess", preprocessor()),
                           ("model", KNeighborsClassifier(n_neighbors=5, weights="distance"))])
    classifier.fit(x.iloc[development], (y.iloc[development] >= threshold).astype(int))
    true_class = (y.iloc[test_idx] >= threshold).astype(int)
    predicted_class = classifier.predict(x.iloc[test_idx])
    probability = classifier.predict_proba(x.iloc[test_idx])[:, 1]
    classification = {
        "threshold_clp": threshold, "labels": ["Bajo la media", "Desde la media"],
        "accuracy": float(accuracy_score(true_class, predicted_class)),
        "precision": float(precision_score(true_class, predicted_class, zero_division=0)),
        "recall": float(recall_score(true_class, predicted_class, zero_division=0)),
        "f1": float(f1_score(true_class, predicted_class, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_class, predicted_class, labels=[0, 1]).tolist(),
    }
    if true_class.nunique() == 2:
        fpr, tpr, _ = roc_curve(true_class, probability)
        classification.update(auc=float(roc_auc_score(true_class, probability)),
                              roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()})
    # Solo comunas observadas durante el ajuste final pueden solicitarse a la API.
    comunas = sorted(x.iloc[development].comuna.unique().tolist())
    numeric_ranges = {col: {"min": float(x.iloc[development][col].min()),
                            "max": float(x.iloc[development][col].max())} for col in NUMERIC}
    report = {
        "schema_version": 2, "currency": "CLP", "target": "arriendo_mensual",
        "seed": seed, "source_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "data_audit": audit,
        "split": {"method": "GroupShuffleSplit por comuna y dirección normalizadas",
                  "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
                  "groups": {part: int(df.iloc[idx].group.nunique()) for part, idx in
                             [("train", train_idx), ("validation", val_idx), ("test", test_idx)]}},
        "selection": {"criterion": "Menor MAE de validación; prueba reservada", "model": selected},
        "models": {name: {"label": LABELS[name], "validation": validation[name],
                          "test": test_metrics[name], "warnings": sorted(set(notes[name]))}
                   for name in candidates},
        "classification_knn": classification,
        "comunas": comunas, "numeric_ranges": numeric_ranges,
        "unseen_test_comunas": sorted(set(x.iloc[test_idx].comuna) - set(comunas)),
        "holdout": {"actual": y.iloc[test_idx].tolist(),
                    "predicted": {key: value.tolist() for key, value in predictions.items()}},
        "diagnostics": {
            "models": explain_models(fitted, x.iloc[test_idx], y.iloc[test_idx], seed),
            "communes": commune_summary(df),
            "reference_property": {col: float(x.iloc[development][col].median()) for col in NUMERIC},
            "k_analysis": k_analysis(preprocessor, x.iloc[train_idx], y.iloc[train_idx],
                                     x.iloc[val_idx], y.iloc[val_idx]),
            "knn_projection": {
                "metros": x.iloc[development].metros.tolist(),
                "gastos_comunes": x.iloc[development].gastos_comunes.tolist(),
                "clase": (y.iloc[development] >= threshold).astype(int).tolist(),
            },
        },
    }
    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "classifier": classifier, "report": report,
                 "development_features": x.iloc[development].reset_index(drop=True),
                 "development_prices": y.iloc[development].to_numpy()},
                artifact_dir / "modelos.joblib", compress=3)
    # No se exportan títulos, direcciones ni identificadores de anuncios.
    (report_dir / "evaluacion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = ["# Evaluación reproducible", "", f"Modelo seleccionado por validación: **{LABELS[selected]}**.",
            "", "Los modelos finales se ajustaron con entrenamiento + validación; estas métricas corresponden a prueba.",
            "", "| Modelo | MAE validación (CLP) | MAE prueba (CLP) | RMSE prueba (CLP) | R² prueba |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name in candidates:
        m = test_metrics[name]
        rows.append(f"| {LABELS[name]} | {validation[name]['mae']:.0f} | {m['mae']:.0f} | {m['rmse']:.0f} | {m['r2']:.4f} |")
    rows += ["", f"Filas: {report['split']['rows']}. Grupos: {report['split']['groups']}.",
             "", "El MAE es un promedio de errores, no un intervalo de confianza individual.",
             "Los resultados describen este conjunto histórico; falta fecha y fuente verificable del scraping."]
    (report_dir / "EVALUACION.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Seleccionado: {LABELS[selected]}; MAE de prueba: ${test_metrics[selected]['mae']:,.0f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/propiedades.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.artifacts, args.reports, args.seed)


if __name__ == "__main__":
    main()
