"""
Вспомогательные функции для проекта
"""
import numpy as np
import random
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_metrics(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"Результаты: {model_name}")
    print(f"{'=' * 50}")
    print(f"MAE:  {mae:.4f} лет (в среднем ошибка на {mae:.1f} лет)")
    print(f"RMSE: {rmse:.4f} лет")
    print(f"R²:   {r2:.4f} ({r2 * 100:.1f}% дисперсии объяснено)")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def detect_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def save_figure(fig, filename, output_dir='outputs/figures'):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    return filepath