ML Project — Предсказание года выпуска песни по тембральным характеристикам

**Студент:** Орлова Надежда Сергеевна

**Группа:** 238

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

**Задача:** Регрессия (предсказание непрерывного значения)

**Что предсказываем:** Год выпуска песни (от 1922 до 2011)

**Датасет:** YearPredictionMSD (Million Song Dataset)
- **Источник:** UCI Machine Learning Repository
- **Ссылка:** https://www.kaggle.com/datasets/uciml/msd-audio-features
- **Объём:** 515,345 треков, 90 признаков

**Признаки:**
- 12 тембральных средних (TimbreAvg1-12)
- 78 ковариаций тембра (TimbreCovariance1-78)

**Целевая метрика:** MAE (Mean Absolute Error) — основная метрика, так как:
- Интерпретируема: "ошибка в среднем на X лет"
- Линейный штраф за ошибку

**Дополнительные метрики:** RMSE, R²

**Почему выбран этот датасет:**
1. Задача предсказания года по аудио-фичам
2. Достаточно большой объем
3. Признаки уже извлечены
4. Есть целевая переменная


## Структура репозитория

.
├── data
│   ├── processed
│   └── raw
├── notebooks
│   ├── 01_eda_analysis.ipynb
├── outputs
│   ├── figures
│   └── results
├── src
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── baseline_models.py
│   └── utils.py
├── requirements.txt
├── main.py
└── README.md

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить полный пайплайн (EDA + Baseline модели)
python main.py

# 5. Или запустить только EDA в Jupyter
jupyter notebook notebooks/01_eda_analysis.ipynb

```

**Требования к системе**
- Python 3.8+
- 4+ GB RAM 
- 1+ GB свободного места на диске