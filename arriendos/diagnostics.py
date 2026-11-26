"""Datos verificables para los análisis de la interfaz; sin direcciones."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .data import FEATURES, NUMERIC, TARGET


def explain_models(fitted, x_test, y_test, seed):
    result = {}
    for name, pipeline in fitted.items():
        model = pipeline.named_steps["model"]
        item = {}
        if name != "baseline":
            importance = permutation_importance(pipeline, x_test, y_test,
                                                scoring="neg_mean_absolute_error",
                                                n_repeats=5, random_state=seed, n_jobs=1)
            item["permutation"] = [
                {"feature": feature, "mean": float(mean), "std": float(std)}
                for feature, mean, std in zip(FEATURES, importance.importances_mean,
                                              importance.importances_std)
            ]
        prep = pipeline.named_steps["preprocess"]
        if name == "lineal":
            scaler = prep.named_transformers_["numeric"]
            encoder = prep.named_transformers_["comuna"]
            coefficients = model.coef_[:len(NUMERIC)] / scaler.scale_
            item.update(
                numeric_coefficients=dict(zip(NUMERIC, coefficients.tolist())),
                reference_comuna=str(encoder.categories_[0][0]),
                comuna_coefficients=[{"comuna": str(c), "coefficient": float(v)}
                                     for c, v in zip(encoder.categories_[0][1:], model.coef_[len(NUMERIC):])],
                intercept_clp=float(model.intercept_ - np.dot(coefficients, scaler.mean_)),
            )
        if name == "random_forest":
            encoder = prep.named_transformers_["comuna"]
            item["numeric_importance"] = dict(zip(NUMERIC, model.feature_importances_[:len(NUMERIC)].tolist()))
            item["comuna_importance"] = [
                {"comuna": str(c), "importance": float(v)} for c, v in
                zip(encoder.categories_[0][1:], model.feature_importances_[len(NUMERIC):])
            ]
            item["reference_comuna"] = str(encoder.categories_[0][0])
            item["trees"] = model.n_estimators
        if name == "red_neuronal":
            network = model.regressor_
            item.update(layers=[network.coefs_[0].shape[0]] + [layer.shape[1] for layer in network.coefs_],
                        loss_curve=[float(v) for v in network.loss_curve_],
                        iterations=int(network.n_iter_))
        result[name] = item
    return result


def commune_summary(df):
    rows = []
    for comuna, group in df.groupby("comuna", sort=True):
        prices = group[TARGET]
        rows.append({"comuna": str(comuna), "count": len(group),
                     "median": float(prices.median()), "mean": float(prices.mean()),
                     "prices": prices.tolist()})
    return sorted(rows, key=lambda row: row["median"], reverse=True)


def k_analysis(preprocess_factory, x_train, y_train, x_val, y_val):
    threshold = float(y_train.mean())
    rows = []
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        if k > len(x_train):
            continue
        classifier = Pipeline([("preprocess", preprocess_factory()),
                               ("model", KNeighborsClassifier(n_neighbors=k, weights="distance"))])
        classifier.fit(x_train, (y_train >= threshold).astype(int))
        actual = (y_val >= threshold).astype(int)
        predicted = classifier.predict(x_val)
        rows.append({"k": k, "accuracy": float(accuracy_score(actual, predicted)),
                     "f1": float(f1_score(actual, predicted, zero_division=0))})
    return {"threshold_clp": threshold, "rows": rows, "used_k": 5,
            "note": "Diagnóstico sobre validación; no cambia K=5 ni selecciona el modelo con prueba."}
