from pathlib import Path
import pickle

import numpy as np
import pandas as pd

FEATURES = [
    "study_hours",
    "attendance",
    "previous_grade",
    "assignments_completed",
    "sleep_hours",
    "extracurricular_hours",
]
TARGET = "final_grade"


class GradeModel:
    """Small dependency-light regression model used by the Streamlit app."""

    def __init__(self, coefficients: np.ndarray, intercept: float):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        values = data[FEATURES].to_numpy(dtype=float)
        return values @ self.coefficients + self.intercept


def train_model(data: pd.DataFrame) -> GradeModel:
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = data[FEATURES].to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float)
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    fitted = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
    return GradeModel(fitted[1:], fitted[0])


def save_model(model: GradeModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str | Path) -> GradeModel:
    with open(path, "rb") as file:
        return pickle.load(file)


def predict_grade(model: GradeModel, values: dict) -> float:
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return max(0.0, min(100.0, prediction))
