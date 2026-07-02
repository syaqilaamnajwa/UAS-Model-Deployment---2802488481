import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best_credit_score_model.pkl")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict) and "instances" in data:
            return pd.DataFrame(data["instances"])

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            return pd.DataFrame(data)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    classes = model.classes_

    results = []

    for prediction, probability in zip(predictions, probabilities):
        results.append({
            "prediction": str(prediction),
            "probability": {
                str(label): float(prob)
                for label, prob in zip(classes, probability)
            }
        })

    if len(results) == 1:
        return results[0]

    return results


def output_fn(prediction, accept):
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction), "application/json"

    raise ValueError(f"Unsupported accept type: {accept}")
