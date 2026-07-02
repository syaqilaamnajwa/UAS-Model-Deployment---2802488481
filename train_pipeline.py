import os
import joblib

from sklearn.model_selection import train_test_split

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    MODEL_PATH = os.path.join(BASE_DIR, "best_credit_score_model.pkl")

    print("Starting data ingestion...")
    ingestion = DataIngestion()
    DATA_PATH = ingestion.ingest_data()

    print("\nLoading data...")
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

    print("Building preprocessing pipeline...")
    preprocessor = preprocessor_obj.build_preprocessor()

    print("\nTraining models with MLflow logging...")
    trainer = ModelTrainer(preprocessor)
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
    final_metrics, final_report, final_matrix = evaluator.evaluate(
        best_pipeline,
        X_test,
        y_test
    )

    print("\nFinal Metrics:")
    print(final_metrics)

    joblib.dump(best_pipeline, MODEL_PATH, compress=("xz", 3))

    print(f"\nBest model saved as: {MODEL_PATH}")
    print("Model size:", round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2), "MB")
    print("Pipeline finished successfully.")