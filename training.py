import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            "Tuned Model (Extra Trees)": ExtraTreesClassifier(
                n_estimators=200,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features=None,
                max_depth=50,
                criterion="entropy",
                bootstrap=False,
                random_state=42,
                n_jobs=-1
            )
        }

    def train_and_log_models(self, X_train, X_test, y_train, y_test):
        results = []
        best_pipeline = None
        best_model_name = None
        best_f1 = 0

        mlflow_db_path = os.path.join(BASE_DIR, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        mlflow.set_experiment("Credit Score Classification")

        for model_name, model in self.models.items():
            print(f"Training: {model_name}")

            with mlflow.start_run(run_name=model_name):
                pipeline = Pipeline(steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                mlflow.log_param("model_name", model_name)

                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                report = classification_report(y_test, y_pred)
                report_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_classification_report.txt"
                report_path = os.path.join(BASE_DIR, report_filename)

                with open(report_path, "w") as file:
                    file.write(report)

                mlflow.log_artifact(report_path)
                os.remove(report_path)

                mlflow.sklearn.log_model(pipeline, "model")

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                })

                if f1 > best_f1:
                    best_f1 = f1
                    best_pipeline = pipeline
                    best_model_name = model_name

        results_df = pd.DataFrame(results)

        comparison_path = os.path.join(BASE_DIR, "model_comparison.csv")
        results_df.to_csv(comparison_path, index=False)

        return results_df, best_pipeline, best_model_name