import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_and_clean_data, create_features, prepare_data_for_modeling, generate_sample_data
from src.eda import perform_eda
from src.baseline_models import run_baseline_pipeline
from src.utils import set_seed


def main():
    set_seed(42)

    data_path = 'data/raw/year_prediction.csv'

    if os.path.exists(data_path):
        df = load_and_clean_data(data_path)
    else:
        df = generate_sample_data(n_samples=5000)

    df = create_features(df)
    perform_eda(df)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_for_modeling(
        df, test_size=0.2, val_size=0.1, use_scaling=True
    )

    results = run_baseline_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"""
     Результаты:
       - Данные: {df.shape[0]:,} строк, {df.shape[1]} признаков
       - Лучшая baseline модель: {results['best_model_name']}
       - Val MAE: {results['val_mae']:.4f} лет
       - Train/Val/Test split: {len(X_train)} / {len(X_val)} / {len(X_test)}
    """)

    return results


if __name__ == "__main__":
    results = main()