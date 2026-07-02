import json
import boto3
import pandas as pd


ENDPOINT_NAME = "credit-score-endpoint-1783004703"
REGION_NAME = "us-east-1"


class SageMakerCreditScoreInference:
    def __init__(self, endpoint_name=ENDPOINT_NAME, region_name=REGION_NAME):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client("sagemaker-runtime", region_name=region_name)

    def predict(self, data):
        payload = json.dumps(data)

        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload
        )

        result = json.loads(response["Body"].read().decode("utf-8"))

        prediction = result["prediction"]

        probability_df = pd.DataFrame(
            list(result["probability"].items()),
            columns=["Credit Score", "Probability"]
        )

        return prediction, probability_df


if __name__ == "__main__":
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

    inference = SageMakerCreditScoreInference()
    prediction, probability_df = inference.predict(sample)

    print("Prediction:", prediction)
    print("\nPrediction Probability:")
    print(probability_df)
