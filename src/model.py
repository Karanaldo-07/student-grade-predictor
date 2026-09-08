from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "study_hours",
    "attendance",
    "previous_grade",
    "assignments_completed",
    "sleep_hours",
    "extracurricular_hours",
]
TARGET = "final_grade"


def build_pipeline() -> Pipeline:
    numeric_features = FEATURES
    preprocessor = ColumnTransformer(
        [("num", "passthrough", numeric_features)],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model(data: pd.DataFrame) -> Pipeline:
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    pipeline = build_pipeline()
    pipeline.fit(data[FEATURES], data[TARGET])
    return pipeline


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_grade(model: Pipeline, values: dict) -> float:
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return max(0.0, min(100.0, prediction))
