"""API local y archivos estáticos de la aplicación."""
from contextlib import asynccontextmanager
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .data import AREA_RANGE, FEATURES, NUMERIC, ROOT


class Property(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    metros: float = Field(ge=AREA_RANGE[0], le=AREA_RANGE[1])
    gastos_comunes: float = Field(ge=0, le=10_000_000)
    habitaciones: int = Field(strict=True, ge=0, le=20)
    banos: int = Field(strict=True, ge=1, le=20)
    comuna: str = Field(min_length=1, max_length=100)
    modelo: str | None = None


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    propiedad: Property
    comunas: list[str] = Field(min_length=2, max_length=3)


def create_app(artifact_path=None):
    path = Path(artifact_path or os.environ.get("ARRIENDOS_MODEL", ROOT / "artifacts/modelos.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        # Solo cargar el artefacto local generado por el entrenamiento del proyecto.
        app.state.bundle = joblib.load(path) if path.is_file() else None
        yield

    app = FastAPI(title="Arriendos · API", lifespan=lifespan)

    def bundle():
        value = app.state.bundle
        if value is None:
            raise HTTPException(503, "Falta entrenar los modelos: python -m arriendos.train")
        return value

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models_ready": app.state.bundle is not None}

    @app.get("/api/metadata")
    def metadata():
        return bundle()["report"]

    @app.post("/api/predict")
    def predict(prop: Property):
        stored = bundle()
        report = stored["report"]
        model_name = prop.modelo or report["selection"]["model"]
        if model_name not in stored["models"]:
            raise HTTPException(422, "Modelo desconocido.")
        if prop.comuna not in report["comunas"]:
            raise HTTPException(422, "Comuna no observada en el entrenamiento.")
        frame = pd.DataFrame([prop.model_dump()])[FEATURES]
        price = float(stored["models"][model_name].predict(frame)[0])
        if not math.isfinite(price) or price <= 0:
            raise HTTPException(422, "El modelo no produjo un precio válido para esta combinación.")
        messages = []
        for column in NUMERIC:
            limits = report["numeric_ranges"][column]
            if not limits["min"] <= getattr(prop, column) <= limits["max"]:
                messages.append(f"{column}: fuera del rango observado en entrenamiento.")
        classification = int(stored["classifier"].predict(frame)[0])
        threshold = report["classification_knn"]["threshold_clp"]
        if price < 100_000 or price > 2_000_000:
            messages.append("El precio estimado queda fuera del rango objetivo del estudio.")
        knn = stored["models"]["knn"]
        distances, indexes = knn.named_steps["model"].kneighbors(knn.named_steps["preprocess"].transform(frame))
        neighbors = []
        for distance, index in zip(distances[0], indexes[0]):
            values = stored["development_features"].iloc[int(index)].to_dict()
            neighbors.append({**values, "precio_clp": float(stored["development_prices"][int(index)]),
                              "distancia": float(distance)})
        return {"precio_clp": int(np.rint(price)), "modelo": model_name,
                "mae_prueba_clp": report["models"][model_name]["test"]["mae"],
                "clasificacion_knn": report["classification_knn"]["labels"][classification],
                "umbral_clasificacion_clp": threshold, "advertencias": messages,
                "vecinos_knn": neighbors}

    @app.post("/api/compare")
    def compare(request: Comparison):
        stored = bundle()
        if len(set(request.comunas)) != len(request.comunas):
            raise HTTPException(422, "Elige comunas distintas para comparar.")
        if any(c not in stored["report"]["comunas"] for c in request.comunas):
            raise HTTPException(422, "Comuna no observada en el entrenamiento.")
        values = request.propiedad.model_dump()
        frame = pd.DataFrame([{**values, "comuna": c} for c in request.comunas])[FEATURES]
        predictions = {}
        for name, model in stored["models"].items():
            predicted = model.predict(frame)
            predictions[name] = [int(np.rint(v)) if math.isfinite(v) and v > 0 else None for v in predicted]
        return {"comunas": request.comunas, "predicciones": predictions,
                "propiedad": {col: values[col] for col in NUMERIC}}

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "web/index.html")

    return app


app = create_app()
