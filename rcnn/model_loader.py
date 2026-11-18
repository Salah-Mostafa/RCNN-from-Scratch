import joblib
from pathlib import Path


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "rcnn_model.joblib"

    model = joblib.load(model_path)

    svc = model["svc"]
    regressors = model["regressors"]

    print("Model loaded successfully.")

    return svc, regressors
