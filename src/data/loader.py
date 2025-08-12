"""
Универсальный загрузчик данных
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

class DataLoader:
    """Универсальный загрузчик данных по конфигурации"""

    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_and_prepare_data(self):
        """Полный цикл загрузки и подготовки данных"""
        print(f"📊 Загрузка датасета: {self.config['dataset']['name']}")

        # Загрузка сырых данных
        raw_data = self._load_raw_data()

        # Подготовка признаков и целевой переменной
        X, y, feature_names = self._prepare_features_and_target(raw_data)

        # Предобработка
        X_train, X_test, y_train, y_test, scaler = self._preprocess_data(X, y)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler,
            'raw_data': raw_data
        }

    def _load_raw_data(self):
        """Загрузка сырых данных"""
        dataset_config = self.config['dataset']
        special = dataset_config.get('special_preprocessing', {})

        if dataset_config['source_type'] == 'url':
            file_path = dataset_config['file_path']

            if special.get('type') == 'breast_cancer':
                # Специальная обработка для Breast Cancer (без заголовков)
                column_names = special['column_names']
                data = pd.read_csv(file_path, header=None, names=column_names)
            else:
                # Стандартная загрузка с URL
                data = pd.read_csv(file_path)

        elif dataset_config['source_type'] == 'local':
            file_path = dataset_config['file_path']

            # Определяем есть ли заголовки
            has_header = special.get('has_header', True)
            column_names = special.get('column_names')

            if file_path.endswith('.csv'):
                if has_header and not column_names:
                    # Стандартная загрузка с заголовками из файла
                    data = pd.read_csv(file_path)
                elif not has_header and column_names:
                    # Загрузка без заголовков с заданными именами
                    data = pd.read_csv(file_path, header=None, names=column_names)
                else:
                    # Fallback - стандартная загрузка
                    data = pd.read_csv(file_path)

            elif file_path.endswith(('.xlsx', '.xls')):
                if has_header and not column_names:
                    # Стандартная загрузка Excel с заголовками
                    data = pd.read_excel(file_path)
                elif not has_header and column_names:
                    # Загрузка Excel без заголовков с заданными именами
                    data = pd.read_excel(file_path, header=None, names=column_names)
                else:
                    # Fallback - стандартная загрузка Excel
                    data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Неподдерживаемый формат: {file_path}")
        else:
            raise ValueError(f"Неизвестный source_type: {dataset_config['source_type']}")

        print(f"✅ Загружено: {data.shape[0]} образцов, {data.shape[1]} колонок")
        print(f"📋 Колонки: {list(data.columns)}")
        return data

    def _prepare_features_and_target(self, data):
        """Подготовка X, y и названий признаков"""
        special = self.config['dataset'].get('special_preprocessing', {})
        dataset_type = special.get('type', 'standard')

        if dataset_type == 'breast_cancer':
            # Специальная обработка для Breast Cancer
            target_col = special['target_column']
            feature_start = special['feature_start_col']
            target_mapping = special['target_mapping']

            # Целевая переменная с маппингом
            y = data[target_col].map(target_mapping)

            # Признаки начиная с указанной колонки
            X = data.iloc[:, feature_start:].copy()
            feature_names = X.columns.tolist()

        elif dataset_type == 'heart_disease' or dataset_type == 'standard':
            # Обработка для Heart Disease и стандартных датасетов
            target_col = special.get('target_column', 'target')
            target_mapping = special.get('target_mapping')

            # Проверяем существует ли target колонка
            if target_col not in data.columns:
                # Если target колонка не найдена, берем последнюю колонку
                target_col = data.columns[-1]
                print(f"⚠️ Колонка '{special.get('target_column', 'target')}' не найдена. Используется '{target_col}'")

            # Целевая переменная
            if target_mapping:
                y = data[target_col].map(target_mapping)
            else:
                y = data[target_col]

            # Признаки - все колонки кроме target
            X = data.drop(columns=[target_col])
            feature_names = X.columns.tolist()

        else:
            # Полностью стандартная обработка
            # Ищем target колонку или берем последнюю
            target_candidates = ['target', 'label', 'class', 'y', 'output']
            target_col = None

            for candidate in target_candidates:
                if candidate in data.columns:
                    target_col = candidate
                    break

            if target_col is None:
                target_col = data.columns[-1]
                print(f"⚠️ Target колонка не определена автоматически. Используется последняя колонка: '{target_col}'")

            y = data[target_col]
            X = data.drop(columns=[target_col])
            feature_names = X.columns.tolist()

        # Статистика классов
        self._print_class_distribution(y, target_col)

        print(f"📊 Подготовлено:")
        print(f"   Целевая переменная: '{target_col}' ({len(y.unique())} уникальных значений)")
        print(f"   Признаков: {len(feature_names)}")
        print(f"   Названия признаков: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")

        return X, y, feature_names

    def _preprocess_data(self, X, y):
        """Предобработка данных"""
        prep_config = self.config['preprocessing']

        # Масштабирование
        if prep_config['scaling']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
            scaler = None

        # Разделение на train/test
        stratify = y if prep_config['stratify'] else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=prep_config['test_size'],
            random_state=prep_config['random_state'],
            stratify=stratify
        )

        print(f"✅ Предобработка завершена:")
        print(f"   Train: {X_train.shape[0]} объектов")
        print(f"   Test: {X_test.shape} объектов")
        print(f"   Признаков: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test, scaler

    def _print_class_distribution(self, y, target_col):
        """Вывод статистики классов"""
        print(f"\n📊 Распределение классов для '{target_col}':")
        for class_val in sorted(y.unique()):
            count = (y == class_val).sum()
            percentage = count / len(y) * 100
            print(f"   Класс {class_val}: {count} образцов ({percentage:.1f}%)")
