import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_figure


def perform_eda(df):

    print("EXPLORATORY DATA ANALYSIS (EDA)")

    # Описание датасета
    print("\nОПИСАНИЕ ДАТАСЕТА")
    print(f"Строк: {df.shape[0]:,}")
    print(f"Столбцов: {df.shape[1]}")
    print(f"Годы: {df['label'].min()} - {df['label'].max()}")
    print(f"Средний год: {df['label'].mean():.1f}")
    print(f"Медиана: {df['label'].median():.0f}")

    # Статистика
    print("\nСТАТИСТИКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print(df['label'].describe())

    # Визуализации
    create_visualizations(df)


def create_visualizations(df):
    print("\nСОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")

    fig = plt.figure(figsize=(16, 12))

    # 1. Распределение годов
    ax1 = plt.subplot(3, 3, 1)
    df['label'].hist(bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Год выпуска')
    ax1.set_ylabel('Количество песен')
    ax1.set_title('Распределение годов выпуска')
    ax1.axvline(df['label'].median(), color='red', linestyle='--',
                label=f'Медиана: {df["label"].median():.0f}')
    ax1.legend()

    # 2. TimbreAvg1 тренд
    ax2 = plt.subplot(3, 3, 2)
    if 'TimbreAvg1' in df.columns:
        yearly_avg = df.groupby('label')['TimbreAvg1'].mean()
        ax2.scatter(df['label'], df['TimbreAvg1'], alpha=0.2, s=5)
        ax2.plot(yearly_avg.index, yearly_avg.values, 'r-', linewidth=2, label='Тренд')
        ax2.set_xlabel('Год')
        ax2.set_ylabel('TimbreAvg1')
        ax2.set_title('Тренд тембра по годам')
        ax2.legend()

    # 3. Корреляции с годом
    ax3 = plt.subplot(3, 3, 3)
    correlations = df.corr()['label'].abs().sort_values(ascending=False)[1:11]
    ax3.barh(range(len(correlations)), correlations.values)
    ax3.set_yticks(range(len(correlations)))
    ax3.set_yticklabels(correlations.index)
    ax3.set_xlabel('Корреляция с годом')
    ax3.set_title('Топ-10 важных признаков')
    ax3.invert_yaxis()

    # 4. Boxplot по десятилетиям
    ax4 = plt.subplot(3, 3, 4)
    if 'TimbreAvg1' in df.columns:
        df['decade'] = (df['label'] // 10) * 10
        decades = [1950, 1960, 1970, 1980, 1990, 2000]
        data_to_plot = [df[df['decade'] == d]['TimbreAvg1'].values for d in decades if len(df[df['decade'] == d]) > 0]
        ax4.boxplot(data_to_plot, labels=[str(d) for d in decades if len(df[df['decade'] == d]) > 0])
        ax4.set_xlabel('Десятилетие')
        ax4.set_ylabel('TimbreAvg1')
        ax4.set_title('Распределение тембра по десятилетиям')
        ax4.grid(True, alpha=0.3)
        df.drop('decade', axis=1, inplace=True)

    # 5. Тепловая карта корреляций
    ax5 = plt.subplot(3, 3, 5)
    timbre_cols = [c for c in df.columns if 'TimbreAvg' in c and 'Covariance' not in c][:6]
    if timbre_cols:
        corr_matrix = df[timbre_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('Корреляции TimbreAvg признаков')

    # 6. Распределение ошибок (демо)
    ax6 = plt.subplot(3, 3, 6)
    errors = np.random.normal(0, 6, 10000)
    ax6.hist(errors, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax6.set_xlabel('Ошибка (годы)')
    ax6.set_ylabel('Плотность')
    ax6.set_title('Ожидаемое распределение ошибок (MAE ≈ 5 лет)')

    # 7. Тренд ковариации
    ax7 = plt.subplot(3, 3, 7)
    if 'Cov_Mean' in df.columns:
        ax7.scatter(df['label'], df['Cov_Mean'], alpha=0.2, s=5)
        z = np.polyfit(df['label'], df['Cov_Mean'], 1)
        p = np.poly1d(z)
        ax7.plot(df['label'].sort_values(), p(df['label'].sort_values()), 'r-', linewidth=2)
        ax7.set_xlabel('Год')
        ax7.set_ylabel('Cov_Mean')
        ax7.set_title('Тренд ковариации тембра')

    # 8. Распределение Avg_Mean
    ax8 = plt.subplot(3, 3, 8)
    if 'Avg_Mean' in df.columns:
        ax8.hist(df['Avg_Mean'], bins=50, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Avg_Mean')
        ax8.set_ylabel('Частота')
        ax8.set_title('Распределение среднего тембра')

    # 9. Соотношение train/val/test
    ax9 = plt.subplot(3, 3, 9)
    sizes = [70, 15, 15]
    labels = ['Train', 'Val', 'Test']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax9.set_title('Сплит данных')

    plt.suptitle('EDA: Анализ музыкальных тембров (1922-2011)', fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'eda_visualizations.png')
    plt.show()

    # Дополнительный график: динамика признаков
    fig2, ax = plt.subplots(figsize=(12, 6))
    timbre_cols = [c for c in df.columns if 'TimbreAvg' in c and 'Covariance' not in c][:5]
    for col in timbre_cols:
        yearly_avg = df.groupby('label')[col].mean()
        ax.plot(yearly_avg.index, yearly_avg.values, label=col, linewidth=2)
    ax.set_xlabel('Год')
    ax.set_ylabel('Значение')
    ax.set_title('Динамика тембральных признаков (1960-2010)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig2, 'timbre_dynamics.png')
    plt.show()
