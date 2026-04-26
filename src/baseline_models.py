import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from src.utils import calculate_metrics, set_seed


class BaselineModels:

    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.results = {}
        self.training_times = {}
        set_seed(42)

    def train_all_models(self):
        print("BASELINE МОДЕЛИ")

        # 1. Linear Regression
        self._train_model(
            LinearRegression(),
            "Linear Regression",
            "Линейная регрессия"
        )

        # 2. Ridge Regression (L2 regularization)
        self._train_model(
            Ridge(alpha=1.0),
            "Ridge Regression",
            "Линейная с L2-регуляризацией"
        )

        # 3. Lasso Regression (L1 regularization)
        self._train_model(
            Lasso(alpha=0.01, max_iter=5000),
            "Lasso Regression",
            "Линейная с L1-регуляризацией"
        )

        # 4. K-Nearest Neighbors
        #self._train_model(
        #    KNeighborsRegressor(n_neighbors=10),
        #    "KNN (k=10)",
        #    "Непараметрическая модель на основе соседей"
        #)

        # 5. Decision Tree
        self._train_model(
            DecisionTreeRegressor(max_depth=10, random_state=42),
            "Decision Tree",
            "Дерево решений"
        )

        # 6. Random Forest (ансамбль деревьев)
        self._train_model(
            RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            "Random Forest",
            "Ансамбль из 50 деревьев"
        )

        return self.results

    def _train_model(self, model, name, description):
        print(f" {name}")
        print(f"   {description}")

        # Замер времени
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        self.training_times[name] = train_time

        # Предсказания
        y_pred_train = model.predict(self.X_train)
        y_pred_val = model.predict(self.X_val)

        # Метрики
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        val_mae = mean_absolute_error(self.y_val, y_pred_val)

        print(f"   Время обучения: {train_time:.2f} сек")
        print(f"   Train MAE: {train_mae:.4f} лет")
        print(f"   Val MAE:   {val_mae:.4f} лет")
        print(f"   Разница train-val: {abs(train_mae - val_mae):.4f}")

        self.results[name] = {
            'model': model,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_time': train_time,
            'predictions': y_pred_val
        }

        return model

    def compare_results(self):
        print("СРАВНЕНИЕ МОДЕЛЕЙ")

        # Создаем таблицу результатов
        results_df = []
        for name, metrics in self.results.items():
            results_df.append({
                'Model': name,
                'Val MAE': f"{metrics['val_mae']:.4f}",
                'Train MAE': f"{metrics['train_mae']:.4f}",
                'Time (s)': f"{metrics['train_time']:.2f}"
            })

        import pandas as pd
        df_results = pd.DataFrame(results_df)
        print(df_results.to_string(index=False))

        # Находим лучшую модель
        best_model = min(self.results.items(), key=lambda x: x[1]['val_mae'])
        print(f"\n ЛУЧШАЯ BASELINE МОДЕЛЬ:")
        print(f"   {best_model[0]} с MAE = {best_model[1]['val_mae']:.4f} лет")

        return best_model


def evaluate_on_test_set(best_model, X_test, y_test):
    y_pred_test = best_model.predict(X_test)
    calculate_metrics(y_test, y_pred_test, "Лучшая baseline модель")

    return y_pred_test


def run_baseline_pipeline(X_train, X_val, X_test, y_train, y_val, y_test):
    # Инициализация и обучение
    baseline = BaselineModels(X_train, X_val, y_train, y_val)
    results = baseline.train_all_models()

    # Сравнение
    best_model_name, best_model_info = baseline.compare_results()

    # Оценка на тесте
    y_pred_test = evaluate_on_test_set(best_model_info['model'], X_test, y_test)

    return {
        'best_model_name': best_model_name,
        'best_model': best_model_info['model'],
        'val_mae': best_model_info['val_mae'],
        'test_predictions': y_pred_test,
        'all_results': results
    }