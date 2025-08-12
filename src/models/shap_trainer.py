"""
SHAP-регуляризованный тренер ANFIS
"""
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score)

class ShapAwareANFISTrainer:
    """Тренер ANFIS с SHAP-регуляризацией"""

    def __init__(self, model,config, gamma=0.5, verbose=True):
        self.model = model.network
        self.gamma = gamma
        self.verbose = verbose
        self.task_type = config['dataset']['task_type']
        self.training_time = 0

    def fit(self, X_train, y_train, epochs=25, batch_size=32, lr=0.005):
        """Обучение с SHAP-регуляризацией"""
        start_time = time.time()

        # Подготовка данных
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        training_dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        # Базовые значения для SHAP
        baseline_values = np.mean(X_train, axis=0)

        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Выбор функции потерь в зависимости от типа задачи
        if self.task_type == 'regression':
            loss_function = torch.nn.MSELoss()
        else:
            loss_function = torch.nn.BCELoss()

        # История потерь
        history = {
            'total_loss': [],
            'main_loss': [],
            'shap_loss': []
        }

        if self.verbose:
            task_name = "регрессии" if self.task_type == 'regression' else "классификации"
            print(f"🟠 Начинаю обучение ANFIS с SHAP-регуляризацией ({task_name})...")

        for epoch in range(epochs):
            epoch_losses = {'total': [], 'main': [], 'shap': []}

            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()

                # Прямой проход
                self.model.train()
                predictions = self.model(batch_X).squeeze()
                main_loss = loss_function(predictions, batch_y)

                # SHAP регуляризация
                shap_importance = self._calculate_shap_approximation(batch_X, baseline_values)
                shap_normalized = shap_importance / (np.sum(shap_importance) + 1e-8)
                target_uniform = np.ones_like(shap_normalized) / len(shap_normalized)
                shap_regularization_loss = np.mean((shap_normalized - target_uniform) ** 2)

                # Общая потеря
                total_loss = main_loss + self.gamma * shap_regularization_loss

                # Обратное распространение
                total_loss.backward()
                optimizer.step()

                # Сохранение потерь
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['main'].append(main_loss.item())
                epoch_losses['shap'].append(shap_regularization_loss)

            # Усреднение потерь по эпохе
            for loss_type in history:
                loss_key = loss_type.split('_')[0]
                history[loss_type].append(np.mean(epoch_losses[loss_key]))

            # Прогресс
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"   Эпоха {epoch + 1}/{epochs}: "
                      f"Total: {history['total_loss'][-1]:.4f}, "
                      f"Main: {history['main_loss'][-1]:.4f}, "
                      f"SHAP: {history['shap_loss'][-1]:.4f}")

        self.training_time = time.time() - start_time

        if self.verbose:
            print(f"✅ Обучение завершено за {self.training_time:.2f} сек")

        return history

    def predict(self, X_test):
        """Получение предсказаний"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
            return predictions

    def get_global_shap_importance(self, X_sample):
        """Глобальная важность признаков"""
        baseline_values = np.mean(X_sample, axis=0)
        return self._calculate_shap_approximation(X_sample, baseline_values)

    def _calculate_shap_approximation(self, X_batch, baseline):
        """Приближенные SHAP значения"""
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X_batch, torch.Tensor):
                X_tensor = torch.tensor(X_batch, dtype=torch.float32)
            else:
                X_tensor = X_batch

            original_predictions = self.model(X_tensor).squeeze().cpu().numpy()
            shap_values = []
            X_numpy = X_tensor.cpu().numpy()

            for feature_index in range(X_numpy.shape[1]):
                X_masked = X_numpy.copy()
                X_masked[:, feature_index] = baseline[feature_index]

                X_masked_tensor = torch.tensor(X_masked, dtype=torch.float32)
                masked_predictions = self.model(X_masked_tensor).squeeze().cpu().numpy()

                if np.isscalar(original_predictions) and np.isscalar(masked_predictions):
                    feature_importance = abs(original_predictions - masked_predictions)
                else:
                    feature_importance = np.mean(np.abs(original_predictions - masked_predictions))

                shap_values.append(feature_importance)

        return np.array(shap_values)
