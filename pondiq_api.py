"""
PondIQ Model API — FastAPI server wrapping the trained XGBoost classifier.

Start:  uv run pondiq_api.py   (or python pondiq_api.py)
Routes:
  GET  /health              — liveness check
  POST /predict             — single prediction from 6 raw water-quality features
  POST /predict/batch       — batch predictions

The model was trained on station2_labelled.csv with:
  Features:  DO, PH, AMMONIA(mg/l), TEMP, NITRATE(PPM), TURBIDITY
  Classes:   0 = Prime Feed  |  1 = Reduce Feed  |  2 = Halt Feeding
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("pondiq-api")

# ── Paths ────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "src" / "models"

MODEL_PATH = MODEL_DIR / "feed_classifier.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.pkl"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.pkl"

# ── Load model artifacts once at startup ─────────────────────────────
_model = joblib.load(MODEL_PATH)
FEATURES: List[str] = joblib.load(FEATURE_LIST_PATH)
CLASS_NAMES: List[str] = joblib.load(CLASS_NAMES_PATH)

log.info("Model loaded: %s", MODEL_PATH)
log.info("Features (%d): %s", len(FEATURES), FEATURES)
log.info("Classes: %s", CLASS_NAMES)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PondIQ Feed Classifier API",
    version="1.0.0",
    description="Predicts optimal feeding action from 6 water-quality parameters.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────────────


class WaterQualityInput(BaseModel):
    """Raw sensor readings — no feature engineering required."""
    do: float = Field(..., ge=0, le=20, description="Dissolved Oxygen (mg/L)")
    ph: float = Field(..., ge=0, le=14, description="pH")
    ammonia: float = Field(..., ge=0, le=10, description="Ammonia (mg/L)")
    temperature: float = Field(..., ge=0, le=50,
                               description="Temperature (°C)")
    nitrate: float = Field(..., ge=0, le=500, description="Nitrate (PPM)")
    turbidity: float = Field(..., ge=0, le=200, description="Turbidity (NTU)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "do": 6.5,
                "ph": 7.2,
                "ammonia": 0.05,
                "temperature": 27.0,
                "nitrate": 20.0,
                "turbidity": 25.0,
            }
        }
    }


class PredictionResult(BaseModel):
    predicted_class: int = Field(
        description="0=Prime Feed, 1=Reduce Feed, 2=Halt Feeding")
    label: str = Field(description="Human-readable class label")
    confidence: float = Field(
        description="Probability of the predicted class (0–1)")
    probabilities: dict[str, float] = Field(
        description="Per-class probabilities")
    warning_flags: list[str] = Field(
        default_factory=list, description="Any hard-override warnings triggered")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": 0,
                "label": "Prime Feed",
                "confidence": 0.97,
                "probabilities": {"Prime Feed": 0.97, "Reduce Feed": 0.02, "Halt Feeding": 0.01},
                "warning_flags": [],
            }
        }
    }


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]


# ── Helpers ──────────────────────────────────────────────────────────

def _check_hard_overrides(row: dict[str, float]) -> list[str]:
    """Return warnings for readings that are in fish-cessation territory.
    These thresholds match the label-generation logic in
    01_data_cleaning_and_label_generation.ipynb (Layer 1 overrides).
    """
    flags: list[str] = []
    if row["DO"] < 3.0:
        flags.append(
            f"DO critically low ({row['DO']:.1f} mg/L) — feeding cessation zone")
    if row["PH"] < 6.0:
        flags.append(
            f"pH dangerously low ({row['PH']:.1f}) — feeding cessation zone")
    if row["PH"] > 9.0:
        flags.append(
            f"pH dangerously high ({row['PH']:.1f}) — feeding cessation zone")
    if row["AMMONIA(mg/l)"] > 0.25:
        flags.append(
            f"Ammonia toxic ({row['AMMONIA(mg/l)']:.3f} mg/L) — feeding cessation zone")
    if row["TEMP"] < 15.0:
        flags.append(
            f"Temperature too low ({row['TEMP']:.1f} °C) — fish stop feeding")
    if row["TEMP"] > 33.0:
        flags.append(
            f"Temperature too high ({row['TEMP']:.1f} °C) — fish stop feeding")
    return flags


def _input_to_dataframe(inp: WaterQualityInput) -> pd.DataFrame:
    """Map the API input schema to the exact column order the model expects."""
    return pd.DataFrame(
        [
            {
                "DO": inp.do,
                "PH": inp.ph,
                "AMMONIA(mg/l)": inp.ammonia,
                "TEMP": inp.temperature,
                "NITRATE(PPM)": inp.nitrate,
                "TURBIDITY": inp.turbidity,
            }
        ]
    )


def _predict_single(inp: WaterQualityInput) -> PredictionResult:
    df = _input_to_dataframe(inp)
    row_dict = df.iloc[0].to_dict()

    probs = _model.predict_proba(df[FEATURES])[0]        # shape (3,)
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return PredictionResult(
        predicted_class=pred_class,
        label=CLASS_NAMES[pred_class],
        confidence=round(confidence, 4),
        probabilities={
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        warning_flags=_check_hard_overrides(row_dict),
    )


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_features": FEATURES, "classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResult)
def predict(inp: WaterQualityInput):
    """Classify a single pond reading into Prime / Reduce / Halt."""
    return _predict_single(inp)


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(inputs: List[WaterQualityInput]):
    """Classify multiple readings at once (e.g. a day of 20-min samples)."""
    if not inputs:
        raise HTTPException(status_code=400, detail="Empty input list")
    return BatchPredictionResult(predictions=[_predict_single(inp) for inp in inputs])


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    log.info("Starting PondIQ API on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
