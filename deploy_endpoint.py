import os
import json
import tarfile
import time
import boto3
import sagemaker

from sagemaker.sklearn.model import SKLearnModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "best_credit_score_model.pkl")
INFERENCE_SCRIPT = os.path.join(BASE_DIR, "inference_sagemaker.py")
MODEL_TAR_FILE = os.path.join(BASE_DIR, "model.tar.gz")

PROJECT_PREFIX = "credit-score-prediction"
ENDPOINT_NAME = f"credit-score-endpoint-{int(time.time())}"


def create_model_tar():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    if not os.path.exists(INFERENCE_SCRIPT):
        raise FileNotFoundError(f"Inference script not found: {INFERENCE_SCRIPT}")

    with tarfile.open(MODEL_TAR_FILE, "w:gz") as tar:
        tar.add(MODEL_FILE, arcname="best_credit_score_model.pkl")

    print("Model artifact created successfully.")
    print(f"Model tar file: {MODEL_TAR_FILE}")

    return MODEL_TAR_FILE


def deploy_model(model_artifact):
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sagemaker_session.default_bucket()

    s3_model_path = sagemaker_session.upload_data(
        path=model_artifact,
        bucket=bucket,
        key_prefix=f"{PROJECT_PREFIX}/model"
    )

    print("Model artifact uploaded to S3.")
    print(f"S3 model path: {s3_model_path}")

    sklearn_model = SKLearnModel(
        model_data=s3_model_path,
        role=role,
        entry_point="inference_sagemaker.py",
        framework_version="1.4-2",
        py_version="py3",
        name=f"{PROJECT_PREFIX}-model-{int(time.time())}"
    )

    print("Deploying model to SageMaker endpoint...")
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )

    print("Endpoint deployment completed.")
    print(f"Endpoint name: {ENDPOINT_NAME}")

    with open(os.path.join(BASE_DIR, "endpoint_name.txt"), "w") as file:
        file.write(ENDPOINT_NAME)

    return predictor


def test_endpoint():
    runtime = boto3.client("sagemaker-runtime")

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

    payload = json.dumps(sample)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload
    )

    result = response["Body"].read().decode("utf-8")

    print("\nEndpoint test result:")
    print(result)


if __name__ == "__main__":
    model_artifact = create_model_tar()
    deploy_model(model_artifact)
    test_endpoint()
    print("\nDeployment process finished successfully.")