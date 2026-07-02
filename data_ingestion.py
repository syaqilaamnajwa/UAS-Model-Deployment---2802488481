import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataIngestion:
    def __init__(self, input_file="data_B.csv"):
        self.input_file = os.path.join(BASE_DIR, input_file)
        self.output_dir = os.path.join(BASE_DIR, "data", "raw")
        self.output_file = os.path.join(self.output_dir, "data_B.csv")

    def ingest_data(self):
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Looking for file at: {self.input_file}")

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Data file not found: {self.input_file}")

        df = pd.read_csv(self.input_file)
        df.to_csv(self.output_file, index=False)

        print("Data ingestion completed successfully.")
        print(f"Ingested data saved to: {self.output_file}")
        print(f"Data shape: {df.shape}")

        return self.output_file