import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error

from arriendos.api import create_app
from arriendos.data import FEATURES, load_data, split_data
from arriendos.train import train
from scripts.preparar_datos_publicos import prepare


@pytest.fixture
def csv_data(tmp_path):
    rng = np.random.default_rng(8)
    rows = []
    for i in range(150):
        area = int(rng.integers(25, 95))
        rows.append({"Nombre_Arriendo": f"Anuncio {i}", "Precio ($)": 180000 + area * 4000,
                     "Gastos_Comunes ($)": area * 1100, "Comuna": ["Santiago", "Ñuñoa"][i % 2],
                     "Direccion": f"Calle {i // 2}", "Metros (m²)": area,
                     "Habitaciones": 2, "Baños": 1})
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cleaning_and_audit(csv_data):
    original = pd.read_csv(csv_data)
    invalid = original.iloc[[0]].copy()
    invalid["Precio ($)"] = 1000
    missing = original.iloc[[1]].copy()
    missing["Direccion"] = np.nan
    fractional = original.iloc[[2]].copy()
    fractional["Baños"] = 1.5
    pd.concat([original, original.iloc[[0]], invalid, missing, fractional]).to_csv(csv_data, index=False)
    df, audit = load_data(csv_data)
    assert len(df) == len(original)
    assert audit["exact_duplicates_removed"] == 1
    assert audit["out_of_scope_or_invalid_removed"] == 3


def test_missing_schema_fails(csv_data):
    df = pd.read_csv(csv_data).drop(columns="Comuna")
    df.to_csv(csv_data, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_data(csv_data)


def test_group_split_is_disjoint_and_repeatable(csv_data):
    df, _ = load_data(csv_data)
    # Variantes de escritura de una misma dirección deben agruparse.
    df.loc[1, "group"] = df.loc[0, "group"]
    parts = split_data(df)
    sets = [set(df.iloc[idx].group) for idx in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sorted(np.concatenate(parts)) == list(range(len(df)))
    for a, b in zip(parts, split_data(df)):
        np.testing.assert_array_equal(a, b)


def test_public_data_preserves_audit_and_split_without_addresses(csv_data, tmp_path):
    public_path = tmp_path / "public.csv"
    prepare(csv_data, public_path)
    raw_public = pd.read_csv(public_path)
    assert "Direccion" not in raw_public.columns
    assert "Nombre_Arriendo" not in raw_public.columns
    assert "Calle" not in public_path.read_text(encoding="utf-8")
    original, original_audit = load_data(csv_data)
    public, public_audit = load_data(public_path)
    assert original_audit == public_audit
    for left, right in zip(split_data(original), split_data(public)):
        np.testing.assert_array_equal(left, right)


@pytest.fixture
def trained(csv_data, tmp_path):
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    report = train(csv_data, artifacts, reports)
    return artifacts / "modelos.joblib", report


def test_report_contains_real_holdout_and_validation_selection(trained):
    _, report = trained
    winner = min(report["models"], key=lambda key: report["models"][key]["validation"]["mae"])
    assert report["selection"]["model"] == winner
    actual = report["holdout"]["actual"]
    assert len(actual) == report["split"]["rows"]["test"]
    for name, model in report["models"].items():
        predicted = report["holdout"]["predicted"][name]
        assert model["test"]["mae"] == pytest.approx(mean_absolute_error(actual, predicted))
    assert sum(map(sum, report["classification_knn"]["confusion_matrix"])) == len(actual)
    assert "Calle" not in json.dumps(report)
    assert "Direccion" not in json.dumps(report)
    communes = report["diagnostics"]["communes"]
    assert sum(row["count"] for row in communes) == report["data_audit"]["clean_rows"]
    assert all(row["median"] == np.median(row["prices"]) for row in communes)
    forest = report["diagnostics"]["models"]["random_forest"]
    assert sum(forest["numeric_importance"].values()) + sum(
        row["importance"] for row in forest["comuna_importance"]) == pytest.approx(1)


def test_api_predicts_using_saved_pipeline(trained):
    path, report = trained
    payload = {"metros": 50, "gastos_comunes": 55000, "habitaciones": 2,
               "banos": 1, "comuna": "Santiago"}
    bundle = joblib.load(path)
    with TestClient(create_app(path)) as client:
        assert client.get("/api/health").json()["models_ready"]
        assert client.get("/api/metadata").json()["selection"] == report["selection"]
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        for model_name in report["models"]:
            response = client.post("/api/predict", json={**payload, "modelo": model_name})
            assert response.status_code == 200, response.text
            expected = bundle["models"][model_name].predict(pd.DataFrame([payload])[FEATURES])[0]
            assert response.json()["precio_clp"] == int(np.rint(expected))
            assert len(response.json()["vecinos_knn"]) == 5
            assert [n["distancia"] for n in response.json()["vecinos_knn"]] == sorted(
                n["distancia"] for n in response.json()["vecinos_knn"])
        for change in [{"metros": -1}, {"metros": 201}, {"gastos_comunes": -1},
                       {"habitaciones": 1.5}, {"banos": 0}, {"comuna": "Desconocida"},
                       {"modelo": "inventado"}, {"extra": "campo"}, {"metros": "NaN"}]:
            response = client.post("/api/predict", json={**payload, **change})
            assert response.status_code == 422, (change, response.text)
        response = client.post("/api/predict", json={**payload, "gastos_comunes": 2_000_000,
                                                      "modelo": "random_forest"})
        assert response.status_code == 200
        assert response.json()["advertencias"]
        communes = report["comunas"][:2]
        response = client.post("/api/compare", json={"propiedad": payload, "comunas": communes})
        assert response.status_code == 200
        frame = pd.DataFrame([{**payload, "comuna": c} for c in communes])[FEATURES]
        for name, model in bundle["models"].items():
            expected = model.predict(frame)
            assert response.json()["predicciones"][name] == np.rint(expected).astype(int).tolist()
        assert client.post("/api/compare", json={"propiedad": payload, "comunas": [communes[0]] * 2}).status_code == 422
        assert client.post("/api/compare", json={"propiedad": payload, "comunas": [communes[0], "Inventada"]}).status_code == 422
        # La fórmula exportada en unidades originales debe reproducir la tubería.
        linear = report["diagnostics"]["models"]["lineal"]
        value = linear["intercept_clp"] + sum(payload[key] * coef for key, coef in linear["numeric_coefficients"].items())
        value += next((row["coefficient"] for row in linear["comuna_coefficients"] if row["comuna"] == payload["comuna"]), 0)
        assert value == pytest.approx(bundle["models"]["lineal"].predict(pd.DataFrame([payload])[FEATURES])[0])


def test_missing_model_has_actionable_error(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert not client.get("/api/health").json()["models_ready"]
        response = client.get("/api/metadata")
        assert response.status_code == 503
        assert "arriendos.train" in response.json()["detail"]
