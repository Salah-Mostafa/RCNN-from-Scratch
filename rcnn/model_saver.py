import joblib
from pathlib import Path


def save_model(svc, regressors, save_path="models/rcnn_model.joblib"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "svc": svc,
        "regressors": regressors
    }

    joblib.dump(model_data, save_path)
    
    print(f"Model saved successfully at: {save_path}")
