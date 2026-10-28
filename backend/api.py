
from pathlib import Path

import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "saved_models" / "optimized_xgboost.pkl"

# Load the optimized XGBoost model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Predictive Maintenance API",
    description="Optimized XGBoost-based industrial machine failure prediction",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://predictive-maintenance-ann.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MachineInput(BaseModel):
    air_temperature: float = Field(gt=0, le=400)
    process_temperature: float = Field(gt=0, le=500)
    rotational_speed: int = Field(gt=0, le=10000)
    torque: float = Field(ge=0, le=1000)
    tool_wear: int = Field(ge=0, le=10000)
    machine_type: str = Field(pattern="^[LMHlmh]$")


@app.get("/")
def home():
    return {
        "message": "Optimized XGBoost Predictive Maintenance API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model": "Optimized XGBoost"}


@app.post("/predict")
def predict(data: MachineInput):
    machine_type = data.machine_type.upper()

    sample = pd.DataFrame(
        [
            {
                "Air temperature [K]": data.air_temperature,
                "Process temperature [K]": data.process_temperature,
                "Rotational speed [rpm]": data.rotational_speed,
                "Torque [Nm]": data.torque,
                "Tool wear [min]": data.tool_wear,
                "Type_L": int(machine_type == "L"),
                "Type_M": int(machine_type == "M"),
                "Type_H": int(machine_type == "H"),
            }
        ],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_L",
            "Type_M",
            "Type_H",
        ],
    )

    probability = float(model.predict_proba(sample.to_numpy())[0][1])

    return {
        "failure_probability": probability,
        "failure_probability_percent": round(probability * 100, 2),
        "prediction": (
            "MACHINE FAILURE"
            if probability >= 0.5
            else "NO MACHINE FAILURE"
        ),
        "machine_type": machine_type,
        "model": "Optimized XGBoost",
    }
