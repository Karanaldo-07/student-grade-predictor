import pandas as pd
import pytest

from src.model import FEATURES, TARGET, predict_grade, train_model


def sample_data():
    return pd.DataFrame([
        [2, 70, 55, 60, 6, 8, 55],
        [4, 80, 65, 75, 7, 6, 68],
        [6, 90, 78, 90, 8, 3, 84],
        [8, 95, 90, 98, 8, 2, 95],
        [3, 75, 60, 70, 6, 7, 62],
        [5, 85, 72, 85, 7, 4, 77],
    ], columns=FEATURES + [TARGET])


def test_model_trains_and_predicts():
    model = train_model(sample_data())
    value = predict_grade(model, {
        "study_hours": 5,
        "attendance": 85,
        "previous_grade": 72,
        "assignments_completed": 85,
        "sleep_hours": 7,
        "extracurricular_hours": 4,
    })
    assert 0 <= value <= 100


def test_missing_column_is_rejected():
    data = sample_data().drop(columns=["attendance"])
    with pytest.raises(ValueError, match="attendance"):
        train_model(data)
