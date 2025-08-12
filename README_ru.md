# XAI 2.0: SHAP-регуляризация для нейро-нечётких сетей ANFIS

Практическая реализация метода встроенной SHAP-регуляризации в модели Adaptive Neuro-Fuzzy Inference System (ANFIS) для повышения интерпретируемости без потери точности.

## 🎯 Основные возможности

- **Vanilla ANFIS**: Обучение стандартных нейро-нечётких моделей
- **SHAP Post-hoc анализ**: Традиционный анализ важности признаков после обучения
- **SHAP-регуляризованный ANFIS**: Встроенная регуляризация для повышения интерпретируемости
- **Универсальная поддержка**: Классификация и регрессия
- **Автоматическая визуализация**: Комплексные графики и метрики
- **Гибкая конфигурация**: YAML-конфигурации для различных датасетов

## 📁 Структура проекта

```
XAI-2.0-SHAP-regularized-ANFIS/
├── configs/                    # YAML конфигурации датасетов
│   ├── breast_cancer.yaml
│   ├── heart_disease.yaml
│   ├── diabetes.yaml
│   ├── banknote_auth.yaml
│   └── wine_quality.yaml
├── src/
│   ├── data/
│   │   └── loader.py          # Универсальный загрузчик данных
│   ├── models/
│   │   ├── anfis_manager.py   # Менеджер ANFIS моделей
│   │   └── shap_trainer.py    # SHAP-регуляризованный тренер
│   ├── analysis/
│   │   └── shap_analyzer.py   # Post-hoc SHAP анализ
│   └── visualization/
│       └── visualizer.py      # Создание графиков и метрик
├── experiments/
│   └── run_experiment.py      # Основной скрипт запуска
├── results/                   # Результаты экспериментов
└── datasets/                  # Папка для датасетов
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

**Основные зависимости:**
- `xanfis` - Библиотека нейро-нечётких систем
- `shap` - Анализ важности признаков
- `scikit-learn` - Машинное обучение
- `matplotlib`, `seaborn` - Визуализация
- `pandas`, `numpy` - Обработка данных

### 2. Подготовка датасета

Поместите ваш датасет в папку `src/datasets/` или используйте готовые конфигурации для популярных датасетов.

### 3. Запуск экспериментов

**Полный эксперимент (все методы):**
```bash
python experiments/run_experiment.py --dataset breast_cancer --experiment all --save-results
```

**Только Vanilla ANFIS:**
```bash
python experiments/run_experiment.py --dataset heart_disease --experiment vanilla
```

**Только SHAP анализ:**
```bash
python experiments/run_experiment.py --dataset diabetes --experiment shap --save-results
```

**Только SHAP-регуляризация:**
```bash
python experiments/run_experiment.py --dataset wine_quality --experiment regularized
```

## 📊 Поддерживаемые датасеты

| Датасет | Конфиг | Задача | Описание |
|---------|---------|---------|-----------|
| Wisconsin Breast Cancer | `breast_cancer` | Классификация | Диагностика рака молочной железы |
| Heart Disease | `heart_disease` | Классификация | Предсказание болезни сердца |
| Pima Indians Diabetes | `diabetes` | Классификация | Диагностика диабета |
| Banknote Authentication | `banknote_auth` | Классификация | Аутентификация банкнот |
| Red Wine Quality | `wine_quality` | Регрессия | Оценка качества вина |

## ⚙️ Создание собственной конфигурации

Создайте файл `configs/my_dataset.yaml`:

```yaml
# Конфигурация для вашего датасета
dataset:
  name: "My Dataset"
  source_type: "local"                    # "local" или "url"
  file_path: "./src/datasets/my_data.csv"
  description: "Описание вашего датасета"
  task_type: "classification"             # "classification" или "regression"

  special_preprocessing:
    type: "standard"                      # Тип обработки
    has_header: true                      # Есть ли заголовки
    target_column: "target"               # Имя целевой колонки
    target_mapping: null                  # Маппинг значений (если нужен)

preprocessing:
  test_size: 0.25                         # Размер тестовой выборки
  random_state: 42                        # Зерно случайности
  scaling: true                           # Масштабирование признаков
  stratify: true                          # Стратифицированное разделение

model:
  num_rules: 5                            # Количество нечётких правил
  mf_class: "GBell"                       # Тип функций принадлежности
  optim: "OriginalPSO"                    # Алгоритм оптимизации
  optim_params:
    epoch: 20                             # Количество эпох
    pop_size: 30                          # Размер популяции
    verbose: true
  reg_lambda: 0.1                         # Коэффициент регуляризации
  seed: 42

shap:
  gamma: 0.5                              # Коэффициент SHAP-регуляризации
  training_epochs: 25                     # Эпохи для SHAP-обучения
  batch_size: 32                          # Размер батча
  learning_rate: 0.005                    # Скорость обучения
  sample_size: 80                         # Размер выборки для SHAP

visualization:
  figure_size: [20, 16]                   # Размер графиков
  style: "seaborn-v0_8"                   # Стиль графиков
  save_plots: true                        # Сохранять графики
  plot_format: "png"                      # Формат файлов
  dpi: 300                                # Разрешение
```

Запустите эксперимент:
```bash
python experiments/run_experiment.py --dataset my_dataset --experiment all
```

## 📈 Результаты экспериментов

После запуска в папке `results/` будут созданы:

- **Графики анализа**: ROC кривые, важность признаков, матрицы ошибок
- **Сравнительные метрики**: CSV таблица с результатами всех методов
- **SHAP визуализации**: Summary plots, waterfall plots, force plots
- **Лог выполнения**: Подробная информация о процессе обучения

## 🔬 Методы исследования

### 1. Vanilla ANFIS
- Стандартное обучение нейро-нечёткой системы
- Извлечение важности признаков из коэффициентов модели
- Базовая точность для сравнения

### 2. Post-hoc SHAP анализ
- Традиционный подход с использованием библиотеки `shap`
- KernelExplainer для модельно-агностичного анализа
- Локальные и глобальные объяснения

### 3. SHAP-регуляризованный ANFIS
- Встроенная SHAP-подобная регуляризация в функцию потерь
- Одновременная оптимизация точности и интерпретируемости
- Снижение вычислительных затрат

## 🎛️ Настройка параметров

### Модель ANFIS
- `num_rules`: Количество нечётких правил (5-15 для большинства задач)
- `mf_class`: Тип функций принадлежности (`GBell`, `Gaussian`, `Sigmoid`)
- `optim`: Алгоритм оптимизации (`OriginalPSO`, `BaseGA`, `OriginalABC`)

### SHAP-регуляризация
- `gamma`: Вес SHAP-регуляризации (0.1-1.0)
- `training_epochs`: Количество эпох (15-50)
- `learning_rate`: Скорость обучения (0.001-0.01)

## 📊 Метрики оценки

### Классификация
- **Accuracy**: Общая точность
- **Precision**: Точность по положительному классу
- **Recall**: Полнота
- **F1-Score**: Гармоническое среднее точности и полноты
- **ROC AUC**: Площадь под ROC кривой

### Регрессия
- **RMSE**: Среднеквадратичная ошибка
- **MAE**: Средняя абсолютная ошибка
- **R²**: Коэффициент детерминации

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature ветку (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в ветку (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

## Авторы

    Yuri Trofimov

    Aleksey Shevchenko

    Andrei Ilin

    Alexander Lebedev

    Aleksey Averkin

## Цитирование

Если вы используете этот код в своих исследованиях, пожалуйста, процитируйте нашу работу:

@software{trofimov2025xai,
  title={XAI 2.0: Embedded SHAP Regularization as a Principle for Building Globally and Locally Interpretable Models},
  author={Trofimov, Yuri and Shevchenko, Aleksey and Ilin, Andrei and Lebedev, Alexander and Averkin, Aleksey},
  year={2025},
  version={v1.0.0},
  publisher={Zenodo},
  doi={10.5281/zenodo.16790521},
  url={https://doi.org/10.5281/zenodo.16790521},
  note={FUZZY\_XAI}
}

## Лицензия

Этот проект является частью научного исследования, опубликованного на Zenodo с DOI: 10.5281/zenodo.16790521.

Условия использования:

    ✅ Академическое и исследовательское использование приветствуется

    ✅ Коммерческое использование разрешено с указанием авторства

    ✅ Модификация и распространение с сохранением указания на оригинальную работу

    ⚠️ Обязательное цитирование при использовании в публикациях

## Отказ от ответственности

Программное обеспечение предоставляется "как есть", без каких-либо гарантий, явных или подразумеваемых. Авторы не несут ответственности за любые убытки или ущерб, возникающие в результате использования данного программного обеспечения.
🏛️ Институциональная принадлежность

Эта работа выполнена в рамках исследований в области объяснимого искусственного интеллекта (XAI 2.0) и нечётких нейронных систем.

## 🆘 Поддержка

При возникновении вопросов или проблем:

1. Проверьте [Issues](../../issues) на наличие похожих проблем
2. Создайте новый Issue с подробным описанием
3. Убедитесь, что все зависимости установлены корректно

***

**Этот проект предназначен для исследователей и практиков в области объяснимого ИИ, заинтересованных в повышении интерпретируемости моделей без потери точности.**