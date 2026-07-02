import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace("_", "", regex=False)
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if "Credit_History_Age" in df_clean.columns:
            years = df_clean["Credit_History_Age"].astype(str).str.extract(r"(\d+)\s+Years").astype(float)
            months = df_clean["Credit_History_Age"].astype(str).str.extract(r"(\d+)\s+Months").astype(float)

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
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        if "Type_of_Loan" in df_clean.columns:
            df_clean["Type_of_Loan"] = df_clean["Type_of_Loan"].fillna("Unknown")

        if "Occupation" in df_clean.columns:
            df_clean["Occupation"] = df_clean["Occupation"].replace("_______", "Unknown")

        if "Credit_Mix" in df_clean.columns:
            df_clean["Credit_Mix"] = df_clean["Credit_Mix"].replace("_", "Unknown")

        if "Payment_Behaviour" in df_clean.columns:
            df_clean["Payment_Behaviour"] = df_clean["Payment_Behaviour"].replace("!@9#%8", "Unknown")

        if "Payment_of_Min_Amount" in df_clean.columns:
            df_clean["Payment_of_Min_Amount"] = df_clean["Payment_of_Min_Amount"].replace("NM", "Unknown")

        invalid_rules = {
            "Age": (0, 100),
            "Num_Bank_Accounts": (0, 20),
            "Num_Credit_Card": (0, 20),
            "Interest_Rate": (0, 100),
            "Num_of_Loan": (0, 20),
            "Num_of_Delayed_Payment": (0, 100)
        }

        for col, (min_val, max_val) in invalid_rules.items():
            if col in df_clean.columns:
                df_clean.loc[(df_clean[col] < min_val) | (df_clean[col] > max_val), col] = np.nan
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        if "Monthly_Balance" in df_clean.columns:
            df_clean.loc[df_clean["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
            df_clean["Monthly_Balance"] = df_clean["Monthly_Balance"].fillna(df_clean["Monthly_Balance"].median())

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