from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.model import FEATURES, TARGET, save_model, train_model

DATA_PATH = Path("data/student_grades.csv")
MODEL_PATH = Path("models/student_grade_model.joblib")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    model = train_model(train_df)
    predictions = model.predict(test_df[FEATURES])

    mae = mean_absolute_error(test_df[TARGET], predictions)
    r2 = r2_score(test_df[TARGET], predictions)
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    save_model(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
