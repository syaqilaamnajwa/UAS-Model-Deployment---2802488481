import os
import joblib
import pandas as pd
from preprocessing import DataPreprocessor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_B.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_credit_score_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "credit_test_cases_final.csv")


def build_test_cases():
    model = joblib.load(MODEL_PATH)

    prep = DataPreprocessor(DATA_PATH)
    prep.load_data()
    df_clean = prep.clean_data()

    X = df_clean.drop(columns=["Credit_Score"])
    y = df_clean["Credit_Score"]

    predictions = model.predict(X)

    result = X.copy()
    result["Expected_Result"] = y
    result["Prediction_Result"] = predictions

    selected_cases = []

    for label in ["Good", "Poor", "Standard"]:
        sample = result[
            (result["Expected_Result"] == label) &
            (result["Prediction_Result"] == label)
        ].head(1)

        if len(sample) == 0:
            sample = result[result["Prediction_Result"] == label].head(1)

        row = sample.iloc[0].copy()
        selected_cases.append(row)

        print("\n==============================")
        print(label)
        print("==============================")
        print(row.to_dict())
        print("Expected Result:", row["Expected_Result"])
        print("Prediction Result:", row["Prediction_Result"])
        print("Status:", "Passed" if row["Expected_Result"] == row["Prediction_Result"] else "Failed")

    test_cases = pd.DataFrame(selected_cases)
    test_cases = test_cases.drop(columns=["Prediction_Result"])
    test_cases.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTest cases saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_test_cases()