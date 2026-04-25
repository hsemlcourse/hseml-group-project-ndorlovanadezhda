import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import detect_outliers_iqr, set_seed


def load_and_clean_data(filepath):
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"\nЗагружено данных: {df.shape[0]:,} строк, {df.shape[1]} столбцов")

    # 1. Проверка пропусков
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"   Найдено пропусков: {len(missing_cols)} столбцов")
        df = df.dropna()
        print(f"   Новый размер: {df.shape[0]:,}")
    else:
        print("   Пропусков нет")

    # 2. Проверка дубликатов
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   Найдено дубликатов: {duplicates}")
        df = df.drop_duplicates()
        print(f"   Дубликаты удалены. Новый размер: {df.shape[0]:,}")
    else:
        print("   Дубликатов нет")

    # 3. Проверка типов
    for col in df.columns:
        if col == 'label':
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    print("   Типы данных приведены корректно")


    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'label']

    for col in numeric_cols[:5]:
        outliers = detect_outliers_iqr(df, col)
        if outliers.sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
    print(f"   Выбросы обработаны для {min(5, len(numeric_cols))} признаков")

    return df


def create_features(df):
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    original_features = len([c for c in df.columns if c != 'label'])
    print(f"\n Исходных признаков: {original_features}")

    # 1. Агрегаты для TimbreAvg
    timbre_avg_cols = [c for c in df.columns if 'TimbreAvg' in c and 'Covariance' not in c]
    if timbre_avg_cols:
        df['Avg_Mean'] = df[timbre_avg_cols].mean(axis=1)
        df['Avg_Std'] = df[timbre_avg_cols].std(axis=1)
        df['Avg_Range'] = df[timbre_avg_cols].max(axis=1) - df[timbre_avg_cols].min(axis=1)
        print(f"   Добавлены: Avg_Mean, Avg_Std, Avg_Range")

    # 2. Агрегаты для TimbreCovariance
    cov_cols = [c for c in df.columns if 'TimbreCovariance' in c]
    if cov_cols:
        df['Cov_Mean'] = df[cov_cols].mean(axis=1)
        df['Cov_Std'] = df[cov_cols].std(axis=1)
        df['Cov_Sum'] = df[cov_cols].sum(axis=1)
        print(f"   Добавлены: Cov_Mean, Cov_Std, Cov_Sum")

        # Trace (приблизительная диагональ)
        diag_indices = list(range(0, min(len(cov_cols), 80), 9))
        if diag_indices:
            diag_cols = [cov_cols[i] for i in diag_indices if i < len(cov_cols)]
            df['Cov_Trace'] = df[diag_cols].sum(axis=1)
            print(f"   Добавлен: Cov_Trace")

    new_features = len([c for c in df.columns if c != 'label']) - original_features
    print(f"\nДобавлено новых признаков: {new_features}")
    print(f"Итого признаков: {len([c for c in df.columns if c != 'label'])}")

    return df


def prepare_data_for_modeling(df, test_size=0.2, val_size=0.1, use_scaling=True):
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ")
    print("=" * 60)

    set_seed(42)

    # Разделение на признаки и целевую переменную
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols]
    y = df['label']

    print(f"\nРазмерность: X = {X.shape}, y = {y.shape}")

    # Разбиваем на train+val и test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=pd.cut(y, bins=20)  # стратификация по году
    )

    # Разбиваем train+val на train и val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        random_state=42,
        stratify=pd.cut(y_train_val, bins=20)
    )

    print(f"\nРаспределение данных:")
    print(f"   Train: {X_train.shape[0]:,} ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]:,} ({X_val.shape[0] / len(X) * 100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]:,} ({X_test.shape[0] / len(X) * 100:.1f}%)")

    print(f"\nДиапазон годов:")
    print(f"   Train: {y_train.min():.0f} - {y_train.max():.0f}")
    print(f"   Val:   {y_val.min():.0f} - {y_val.max():.0f}")
    print(f"   Test:  {y_test.min():.0f} - {y_test.max():.0f}")

    # Масштабирование
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print(f"\n Данные масштабированы (StandardScaler)")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

    return X_train, X_val, X_test, y_train, y_val, y_test, None


def generate_sample_data(n_samples=5000):
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ДЕМО-ДАННЫХ")
    print("=" * 60)

    np.random.seed(42)
    years = np.random.choice(range(1922, 2012), n_samples)

    timbre_cols = [f'TimbreAvg{i}' for i in range(1, 13)]
    cov_cols = [f'TimbreCovariance{i}' for i in range(1, 79)]

    data = {'label': years}

    # Генерация признаков с трендом
    for i, col in enumerate(timbre_cols):
        trend = (years - 1950) / 100 * np.random.uniform(0.5, 2.0)
        noise = np.random.normal(0, 10, n_samples)
        data[col] = 50 + trend + noise + i * 2

    for i, col in enumerate(cov_cols[:30]):
        trend = (years - 1950) / 100 * np.random.uniform(-1, 1)
        noise = np.random.normal(0, 5, n_samples)
        data[col] = 0 + trend + noise

    df = pd.DataFrame(data)
    print(f"Сгенерировано {n_samples} строк, {len(df.columns)} столбцов")

    return df