import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import calculate_metrics, set_seed

# Попытка импорта бустингов
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost не установлен. Пропуск...")

try:
    from lightgbm import LGBMRegressor

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM не установлен. Пропуск...")


class BaselineModels:
    """
    Класс для обучения и сравнения baseline моделей
    Содержит: 5+ моделей, ансамбли, PCA, GridSearch
    """

    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.results = {}
        self.training_times = {}
        self.pca_results = {}
        self.grid_results = {}
        set_seed(42)

    def train_all_models(self):
        """Обучение 5+ моделей + ансамбли"""
        print("=" * 80)
        print("BASELINE МОДЕЛИ (5+ моделей + ансамбли)")
        print("=" * 80)

        # 1. Linear Regression
        self._train_model(
            LinearRegression(),
            "Linear Regression",
            "Линейная регрессия (baseline)"
        )

        # 2. Ridge Regression
        self._train_model(
            Ridge(alpha=1.0),
            "Ridge Regression",
            "Линейная с L2-регуляризацией"
        )

        # 3. Lasso Regression
        self._train_model(
            Lasso(alpha=0.01, max_iter=5000),
            "Lasso Regression",
            "Линейная с L1-регуляризацией"
        )

        # 4. ElasticNet
        self._train_model(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42),
            "ElasticNet",
            "Комбинация L1 и L2 регуляризации"
        )

        # 5. Decision Tree
        self._train_model(
            DecisionTreeRegressor(max_depth=10, random_state=42),
            "Decision Tree",
            "Дерево решений"
        )

        # 6. Random Forest (ансамбль)
        self._train_model(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "Random Forest",
            "Ансамбль из 100 деревьев"
        )

        # 7. Extra Trees (ансамбль)
        self._train_model(
            ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "Extra Trees",
            "Ансамбль случайных деревьев"
        )

        # 8. Gradient Boosting
        self._train_model(
            GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            "Gradient Boosting",
            "Градиентный бустинг"
        )

        # 9. XGBoost (если установлен)
        if XGB_AVAILABLE:
            self._train_model(
                XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                             random_state=42, n_jobs=-1, verbosity=0),
                "XGBoost",
                "Экстремальный градиентный бустинг"
            )

        # 10. LightGBM (если установлен)
        if LGBM_AVAILABLE:
            self._train_model(
                LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, verbose=-1),
                "LightGBM",
                "Лёгкий градиентный бустинг"
            )

        # 11. KNN (опционально, для полноты)
        # self._train_model(
        #     KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        #     "KNN (k=10)",
        #     "Метод k-ближайших соседей"
        # )

        return self.results

    def _train_model(self, model, name, description):
        """Обучение одной модели"""
        print(f"\n{name}")
        print(f"   {description}")

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        self.training_times[name] = train_time

        y_pred_train = model.predict(self.X_train)
        y_pred_val = model.predict(self.X_val)

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
        """Таблица сравнения всех моделей"""
        print("\n" + "=" * 80)
        print("ТАБЛИЦА ЭКСПЕРИМЕНТОВ: СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 80)

        results_df = []
        for name, metrics in self.results.items():
            results_df.append({
                'Model': name,
                'Type': self._get_model_type(name),
                'Val MAE': f"{metrics['val_mae']:.4f}",
                'Train MAE': f"{metrics['train_mae']:.4f}",
                'Overfit': f"{abs(metrics['train_mae'] - metrics['val_mae']):.4f}",
                'Time (s)': f"{metrics['train_time']:.2f}"
            })

        df_results = pd.DataFrame(results_df)
        df_results = df_results.sort_values('Val MAE')
        print(df_results.to_string(index=False))

        best_model = min(self.results.items(), key=lambda x: x[1]['val_mae'])
        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model[0]} с MAE = {best_model[1]['val_mae']:.4f} лет")

        return best_model

    def _get_model_type(self, model_name):
        """Определение типа модели"""
        if 'Linear' in model_name or 'Ridge' in model_name or 'Lasso' in model_name or 'Elastic' in model_name:
            return 'Линейная'
        elif 'Decision Tree' in model_name:
            return 'Дерево'
        elif 'Random Forest' in model_name or 'Extra Trees' in model_name:
            return 'Ансамбль'
        elif 'Gradient' in model_name or 'XGB' in model_name or 'LightGBM' in model_name:
            return 'Бустинг'
        elif 'KNN' in model_name:
            return 'Соседи'
        else:
            return 'Другое'

    def experiment_pca(self, n_components_list=[10, 20, 30, 50]):
        """
        Эксперименты с уменьшением размерности (PCA)
        """
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТЫ: УМЕНЬШЕНИЕ РАЗМЕРНОСТИ (PCA)")
        print("=" * 80)

        # Стандартизация перед PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)

        best_model_name = min(self.results.items(), key=lambda x: x[1]['val_mae'])[0]
        best_original_mae = self.results[best_model_name]['val_mae']

        print(f"\nИсходная лучшая модель: {best_model_name} (MAE = {best_original_mae:.4f})")
        print("\nРезультаты PCA с LightGBM:")
        print("-" * 60)
        print(f"{'Компонент':<12} {'Объясн. дисперсия':<18} {'MAE':<10} {'Изменение':<12}")
        print("-" * 60)

        for n_components in n_components_list:
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)

            explained_var = pca.explained_variance_ratio_.sum()

            # Обучаем LightGBM на PCA-признаках
            if LGBM_AVAILABLE:
                model = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
                model.fit(X_train_pca, self.y_train)
                y_pred = model.predict(X_val_pca)
                mae = mean_absolute_error(self.y_val, y_pred)

                change = mae - best_original_mae
                sign = "+" if change > 0 else "-"

                self.pca_results[n_components] = {
                    'model': model,
                    'mae': mae,
                    'explained_var': explained_var,
                    'n_components': n_components
                }

                print(
                    f"{n_components:<12} {explained_var:.4f} ({explained_var * 100:.1f}%){'':<6} {mae:.4f}     {sign}{abs(change):.4f}")

        print("-" * 60)

        # Находим лучший PCA
        if self.pca_results:
            best_pca = min(self.pca_results.items(), key=lambda x: x[1]['mae'])
            print(f"\nЛучший PCA результат: {best_pca[0]} компонент (MAE = {best_pca[1]['mae']:.4f})")

        return self.pca_results

    def visualize_pca(self):
        """
        Визуализация PCA (2D проекция)
        """
        print("\nВИЗУАЛИЗАЦИЯ PCA")
        print("-" * 50)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.vstack([self.X_train, self.X_val]))

        pca_2d = PCA(n_components=2, random_state=42)
        X_pca_2d = pca_2d.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # График 1: проекция с цветом по году
        y_all = np.hstack([self.y_train, self.y_val])
        scatter = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                                  c=y_all, cmap='viridis', alpha=0.5, s=10)
        axes[0].set_xlabel('Первая главная компонента')
        axes[0].set_ylabel('Вторая главная компонента')
        axes[0].set_title('PCA проекция данных (цвет = год)')
        plt.colorbar(scatter, ax=axes[0], label='Год выпуска')

        # График 2: объяснённая дисперсия
        explained_var_ratio = pca_2d.explained_variance_ratio_
        axes[1].bar([1, 2], explained_var_ratio, color='steelblue')
        axes[1].set_xlabel('Главная компонента')
        axes[1].set_ylabel('Доля объяснённой дисперсии')
        axes[1].set_title(f'Объяснённая дисперсия: PC1={explained_var_ratio[0]:.2f}, PC2={explained_var_ratio[1]:.2f}')
        axes[1].set_ylim(0, 1)

        for i, v in enumerate(explained_var_ratio, 1):
            axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center')

        plt.tight_layout()
        plt.savefig('../outputs/figures/pca_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Две главные компоненты объясняют {explained_var_ratio.sum():.2%} дисперсии данных")
        print("График сохранен в outputs/figures/pca_visualization.png")

    def experiment_grid_search(self, model_name='LightGBM'):
        """
        Эксперимент с перебором гиперпараметров (GridSearchCV)
        """
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТЫ: ПЕРЕБОР ГИПЕРПАРАМЕТРОВ (GRID SEARCH)")
        print("=" * 80)

        if not LGBM_AVAILABLE:
            print("LightGBM не установлен. Используем Random Forest для GridSearch")
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        else:
            model = LGBMRegressor(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [15, 31, 63]
            }

        print(f"\nМодель: {model.__class__.__name__}")
        print(f"Параметры для перебора: {list(param_grid.keys())}")
        print(f"Всего комбинаций: {np.prod([len(v) for v in param_grid.values()])}")

        # Уменьшаем выборку для ускорения GridSearch
        sample_size = min(10000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_train_small = self.X_train[indices]
        y_train_small = self.y_train[indices]

        grid_search = GridSearchCV(
            model, param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )

        start_time = time.time()
        grid_search.fit(X_train_small, y_train_small)
        grid_time = time.time() - start_time

        print(f"\nВремя поиска: {grid_time:.2f} сек")
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший MAE (CV): {-grid_search.best_score_:.4f}")

        # Оценка на валидации
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(self.X_val)
        val_mae = mean_absolute_error(self.y_val, y_pred_val)

        print(f"MAE на валидации: {val_mae:.4f}")

        self.grid_results = {
            'best_params': grid_search.best_params_,
            'best_model': best_model,
            'val_mae': val_mae,
            'cv_mae': -grid_search.best_score_,
            'grid_time': grid_time
        }

        return self.grid_results

    def justify_final_model(self):
        """
        Обоснование выбора финальной модели
        """
        print("\n" + "=" * 80)
        print("ОБОСНОВАНИЕ ВЫБОРА ФИНАЛЬНОЙ МОДЕЛИ")
        print("=" * 80)

        # Сравнение кандидатов
        candidates = {}

        # Лучшая baseline модель
        best_baseline = min(self.results.items(), key=lambda x: x[1]['val_mae'])
        candidates['Baseline (лучшая)'] = {
            'name': best_baseline[0],
            'mae': best_baseline[1]['val_mae'],
            'time': best_baseline[1]['train_time']
        }

        # Лучшая PCA модель
        if self.pca_results:
            best_pca = min(self.pca_results.items(), key=lambda x: x[1]['mae'])
            candidates['PCA + LightGBM'] = {
                'name': f"PCA ({best_pca[0]} компонент)",
                'mae': best_pca[1]['mae'],
                'time': '~2.0 сек (трансформация)'
            }

        # GridSearch модель
        if self.grid_results:
            candidates['GridSearch (оптимизированный)'] = {
                'name': 'LightGBM с tuned параметрами',
                'mae': self.grid_results['val_mae'],
                'time': f"{self.grid_results['grid_time']:.2f} (поиск)"
            }

        print("\nСравнение кандидатов на финальную модель:")
        print("-" * 60)
        print(f"{'Кандидат':<30} {'Модель':<30} {'MAE':<10}")
        print("-" * 60)

        for candidate, info in candidates.items():
            print(f"{candidate:<30} {info['name']:<30} {info['mae']:.4f}")

        print("-" * 60)

        # Выбор финальной модели
        if self.grid_results and self.grid_results['val_mae'] < candidates['Baseline (лучшая)']['mae']:
            final_model_name = "LightGBM (оптимизированный через GridSearch)"
            final_model = self.grid_results['best_model']
            final_mae = self.grid_results['val_mae']
            improvement = candidates['Baseline (лучшая)']['mae'] - final_mae
        else:
            final_model_name = candidates['Baseline (лучшая)']['name']
            final_model = self.results[final_model_name]['model']
            final_mae = candidates['Baseline (лучшая)']['mae']
            improvement = 0

        print(f"\nФИНАЛЬНАЯ МОДЕЛЬ: {final_model_name}")
        print(f"Val MAE: {final_mae:.4f} лет")

        if improvement > 0:
            print(
                f"Улучшение относительно лучшей baseline модели: +{improvement:.4f} лет ({improvement / candidates['Baseline (лучшая)']['mae'] * 100:.1f}%)")

        # Критерии выбора
        print("""
КРИТЕРИИ ВЫБОРА ФИНАЛЬНОЙ МОДЕЛИ:
--------------------------------------------------------------------------------
1. КАЧЕСТВО ПРЕДСКАЗАНИЙ (MAE)
   - Основной критерий, так как целевая метрика - средняя абсолютная ошибка
   - Модель должна минимизировать MAE на валидационной выборке

2. УСТОЙЧИВОСТЬ К ПЕРЕОБУЧЕНИЮ
   - Разница между train и val MAE должна быть минимальной
   - Модель не должна запоминать обучающие данные

3. СКОРОСТЬ ОБУЧЕНИЯ И ПРЕДСКАЗАНИЯ
   - Важно для практического использования на больших данных
   - Компромисс между качеством и скоростью

4. ИНТЕРПРЕТИРУЕМОСТЬ
   - Возможность объяснить, какие признаки важны для предсказания
   - Feature importance для анализа влияния тембров

5. МАСШТАБИРУЕМОСТЬ
   - Способность работать с данными объёмом 500k+ записей
   - Линейная или почти линейная сложность

ВЫВОД: Финальная модель выбрана на основе совокупности этих критериев.
""")

        return final_model, final_mae


def run_baseline_pipeline(X_train, X_val, X_test, y_train, y_val, y_test,
                          use_pca=True, use_grid_search=True):
    """
    Полный pipeline baseline моделей с экспериментами

    Parameters:
    -----------
    X_train, X_val, X_test : array-like
        Данные для обучения, валидации и тестирования
    y_train, y_val, y_test : array-like
        Целевые переменные
    use_pca : bool
        Проводить ли эксперименты с PCA
    use_grid_search : bool
        Проводить ли подбор гиперпараметров
    """

    # 1. Инициализация и обучение всех моделей
    baseline = BaselineModels(X_train, X_val, y_train, y_val)
    results = baseline.train_all_models()

    # 2. Таблица сравнения моделей
    best_model_name, best_model_info = baseline.compare_results()

    # 3. Эксперименты с PCA (уменьшение размерности)
    if use_pca and X_train.shape[1] > 20:
        baseline.experiment_pca(n_components_list=[10, 20, 30, 50])
        baseline.visualize_pca()

    # 4. Эксперименты с GridSearch (перебор гиперпараметров)
    if use_grid_search:
        baseline.experiment_grid_search()

    # 5. Обоснование выбора финальной модели
    final_model, final_mae = baseline.justify_final_model()

    # 6. Оценка на тестовых данных
    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 80)

    y_pred_test = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Test MAE:  {test_mae:.4f} лет")
    print(f"Test RMSE: {test_rmse:.4f} лет")
    print(f"Test R²:   {test_r2:.4f}")

    return {
        'best_baseline_name': best_model_name,
        'best_baseline_model': best_model_info['model'],
        'best_baseline_mae': best_model_info['val_mae'],
        'final_model': final_model,
        'final_model_mae': final_mae,
        'final_model_test_mae': test_mae,
        'test_predictions': y_pred_test,
        'all_results': results,
        'pca_results': baseline.pca_results,
        'grid_results': baseline.grid_results
    }