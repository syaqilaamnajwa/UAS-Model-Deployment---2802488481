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
