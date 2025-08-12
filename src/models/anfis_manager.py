"""
Менеджер ANFIS моделей
"""
import time
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score)
from xanfis import BioAnfisClassifier, BioAnfisRegressor

class ANFISManager:
    """Менеджер для обучения ANFIS моделей"""

    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.task_type = config['dataset']['task_type']  # 'classification' или 'regression'

    def create_model(self, verbose=True):
        """Создание модели ANFIS"""
        base_params = {
            'num_rules': self.model_config['num_rules'],
            'mf_class': self.model_config['mf_class'],
            'optim': self.model_config['optim'],
            'optim_params': self.model_config['optim_params'],
            'reg_lambda': self.model_config['reg_lambda'],
            'seed': self.model_config['seed'],
            'verbose': verbose
        }

        if self.task_type == 'regression':
            return BioAnfisRegressor(**base_params)
        else:
            return BioAnfisClassifier(**base_params)

    def train_vanilla_model(self, X_train, y_train, X_test, y_test):
        """Обучение стандартной ANFIS модели"""
        task_name = "Регрессия" if self.task_type == 'regression' else "Классификация"
        print(f"🔵 Обучение Vanilla ANFIS ({task_name})...")
        start_time = time.time()

        # Создание и обучение модели
        model = self.create_model(verbose=True)
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        model.fit(X_train, y_train_values)

        training_time = time.time() - start_time

        # Получение предсказаний
        y_pred = model.predict(X_test)

        # ИСПРАВЛЕНИЕ: Приводим предсказания к правильной форме
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        if self.task_type == 'regression':

            y_prob_for_metrics = y_pred
        else:
            y_prob = model.predict_proba(X_test)
            # Для бинарной классификации берем вероятности для класса 1
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob_for_metrics = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.flatten()
            else:
                y_prob_for_metrics = y_prob.flatten()

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_prob_for_metrics)

        # Важность признаков
        feature_importance = self._extract_feature_importance(model, X_train.shape[1])

        results = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob_for_metrics,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': training_time
        }

        self._print_results(results, f"Vanilla ANFIS ({task_name})")
        return results

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Вычисление метрик в зависимости от типа задачи"""
        if self.task_type == 'regression':
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_prob)
            }

    def _extract_feature_importance(self, model, n_features):
        """Извлечение важности признаков из модели"""
        try:
            coefficients = model.network.state_dict()['coeffs'].detach().cpu().numpy()
            return np.sum(np.abs(coefficients[:, :-1, 0]), axis=0)
        except Exception as e:
            print(f"⚠️ Не удалось извлечь важность признаков: {e}")
            return np.ones(n_features) / n_features

    def _print_results(self, results, model_name):
        """Вывод результатов"""
        metrics = results['metrics']
        print(f"\n✅ {model_name} обучен успешно!")

        if self.task_type == 'regression':
            print(f"   📊 RMSE: {metrics['rmse']:.4f}")
            print(f"   🎯 MAE: {metrics['mae']:.4f}")
            print(f"   📈 R²: {metrics['r2']:.4f}")
        else:
            print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"   🎯 Precision: {metrics['precision']:.4f}")
            print(f"   🎯 Recall: {metrics['recall']:.4f}")
            print(f"   🎯 F1-Score: {metrics['f1_score']:.4f}")
            print(f"   📈 ROC AUC: {metrics['roc_auc']:.4f}")

        print(f"   ⏱️ Время: {results['training_time']:.2f} сек")
