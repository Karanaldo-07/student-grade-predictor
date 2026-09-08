from pathlib import Path

import numpy as np
import pandas as pd

from src.model import FEATURES, TARGET, save_model, train_model

DATA_PATH = Path("data/student_grades.csv")
MODEL_PATH = Path("models/student_grade_model.joblib")


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(data))
    test_size = max(1, int(round(len(data) * 0.2)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_df = data.iloc[train_indices]
    test_df = data.iloc[test_indices]

    model = train_model(train_df)
    predictions = model.predict(test_df[FEATURES])
    actual = test_df[TARGET].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(actual - predictions)))
    ss_res = float(np.sum((actual - predictions) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0

    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    save_model(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
