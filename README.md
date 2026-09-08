# Student Grade Predictor

A lightweight machine-learning web app that estimates a student's final grade from academic and study-related inputs.

## Live demo

https://student-grade-predictor-josdsyjd4czsfszk5ezb7e.streamlit.app

## Features

- Interactive Streamlit prediction interface.
- Six student-performance inputs: study time, attendance, previous grade, assignments, sleep, and extracurricular activity.
- Dependency-light NumPy/pandas regression model for reliable free hosting.
- Predictions are bounded to the 0–100% range.
- Performance categories: Excellent, Good, Needs improvement, and At risk.
- Reproducible training script and automated tests with GitHub Actions.

## Project structure

```text
student-grade-predictor/
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
├── data/
│   └── student_grades.csv
├── src/
│   └── model.py
├── tests/
│   └── test_model.py
└── .github/workflows/
    └── test.yml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Train the model

```bash
python train.py
```

The included dataset is synthetic demo data created for educational purposes. Model performance on this dataset should not be interpreted as evidence of real-world academic prediction accuracy.

## Disclaimer

This project is an educational demonstration. Predictions are estimates and should not be used as the sole basis for academic decisions.
