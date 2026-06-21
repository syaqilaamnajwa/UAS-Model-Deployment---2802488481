import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.numeric_cols = None
        self.categorical_cols = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def clean_data(self):
        if self.df is None:
            self.load_data()

        df_clean = self.df.copy()

        drop_cols = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]
        df_clean = df_clean.drop(columns=drop_cols, errors="ignore")

        num_str_cols = [
            "Age",
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in num_str_cols:
            df_clean[col] = df_clean[col].astype(str).str.replace("_", "", regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        years = df_clean["Credit_History_Age"].str.extract(r"(\d+)\s+Years").astype(float)
        months = df_clean["Credit_History_Age"].str.extract(r"(\d+)\s+Months").astype(float)

        df_clean["Credit_History_Age_Months"] = (years[0] * 12) + months[0]
        df_clean = df_clean.drop(columns=["Credit_History_Age"])

        numeric_missing_cols = [
            "Monthly_Inhand_Salary",
            "Credit_History_Age_Months",
            "Num_of_Delayed_Payment",
            "Amount_invested_monthly",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Monthly_Balance"
        ]

        for col in numeric_missing_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        df_clean["Type_of_Loan"] = df_clean["Type_of_Loan"].fillna("Unknown")

        df_clean["Occupation"] = df_clean["Occupation"].replace("_______", "Unknown")
        df_clean["Credit_Mix"] = df_clean["Credit_Mix"].replace("_", "Unknown")
        df_clean["Payment_Behaviour"] = df_clean["Payment_Behaviour"].replace("!@9#%8", "Unknown")
        df_clean["Payment_of_Min_Amount"] = df_clean["Payment_of_Min_Amount"].replace("NM", "Unknown")

        df_clean.loc[(df_clean["Age"] < 0) | (df_clean["Age"] > 100), "Age"] = np.nan
        df_clean.loc[(df_clean["Num_Bank_Accounts"] < 0) | (df_clean["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
        df_clean.loc[(df_clean["Num_Credit_Card"] < 0) | (df_clean["Num_Credit_Card"] > 20), "Num_Credit_Card"] = np.nan
        df_clean.loc[(df_clean["Interest_Rate"] < 0) | (df_clean["Interest_Rate"] > 100), "Interest_Rate"] = np.nan
        df_clean.loc[(df_clean["Num_of_Loan"] < 0) | (df_clean["Num_of_Loan"] > 20), "Num_of_Loan"] = np.nan
        df_clean.loc[(df_clean["Num_of_Delayed_Payment"] < 0) | (df_clean["Num_of_Delayed_Payment"] > 100), "Num_of_Delayed_Payment"] = np.nan
        df_clean.loc[df_clean["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan

        invalid_cols = [
            "Age",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Monthly_Balance"
        ]

        for col in invalid_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        self.df_clean = df_clean
        return self.df_clean

    def split_features_target(self):
        if self.df_clean is None:
            self.clean_data()

        X = self.df_clean.drop(columns=["Credit_Score"])
        y = self.df_clean["Credit_Score"]

        self.numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        return X, y

    def build_preprocessor(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ]
        )

        return preprocessor


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        }

    def train_and_log_models(self, X_train, X_test, y_train, y_test):
        import mlflow
        import mlflow.sklearn

        results = []
        best_pipeline = None
        best_model_name = None
        best_f1 = 0

        mlflow_db = os.path.join(BASE_DIR, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
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

                report = classification_report(y_test, y_pred)

                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                report_path = os.path.join(BASE_DIR, f"{model_name.replace(' ', '_')}_report.txt")
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


class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        return metrics


if __name__ == "__main__":
    DATA_PATH = os.path.join(BASE_DIR, "data_B.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "best_credit_score_model.pkl")

    print("Loading data...")
    preprocessor_obj = DataPreprocessor(DATA_PATH)
    preprocessor_obj.load_data()

    print("Cleaning data...")
    preprocessor_obj.clean_data()

    print("Splitting features and target...")
    X, y = preprocessor_obj.split_features_target()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = preprocessor_obj.build_preprocessor()

    trainer = ModelTrainer(preprocessor)

    print("\nTraining models with MLflow logging...")
    results_df, best_pipeline, best_model_name = trainer.train_and_log_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nModel Comparison:")
    print(results_df)

    print("\nBest Model:", best_model_name)

    evaluator = ModelEvaluator()
    final_metrics = evaluator.evaluate(best_pipeline, X_test, y_test)

    print("\nFinal Metrics:")
    print(final_metrics)

    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nBest model saved as: {MODEL_PATH}")
    print("Pipeline finished successfully.")