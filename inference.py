import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_credit_score_model.pkl")


class CreditScoreInference:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, data):
        input_df = pd.DataFrame([data])
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]

        probability_df = pd.DataFrame({
            "Credit_Score": self.model.classes_,
            "Probability": probability
        })

        return prediction, probability_df


if __name__ == "__main__":
    inference = CreditScoreInference()

    sample = {
        "Month": "January",
        "Age": 42.0,
        "Occupation": "Teacher",
        "Annual_Income": 19214.965,
        "Monthly_Inhand_Salary": 1730.247083,
        "Num_Bank_Accounts": 0.0,
        "Num_Credit_Card": 4.0,
        "Interest_Rate": 11.0,
        "Num_of_Loan": 0.0,
        "Type_of_Loan": "Unknown",
        "Delay_from_due_date": 10,
        "Num_of_Delayed_Payment": 10.0,
        "Changed_Credit_Limit": 4.18,
        "Num_Credit_Inquiries": 0.0,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 498.81,
        "Credit_Utilization_Ratio": 37.600265,
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 0.0,
        "Amount_invested_monthly": 217.780472,
        "Payment_Behaviour": "Unknown",
        "Monthly_Balance": 245.244236,
        "Credit_History_Age_Months": 345.0
    }

    prediction, probability_df = inference.predict(sample)

    print("Prediction:", prediction)
    print("\nPrediction Probability:")
    print(probability_df)