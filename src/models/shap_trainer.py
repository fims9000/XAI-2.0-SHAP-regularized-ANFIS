"""
Тренер ANFIS с Baseline-Masked Feature Sensitivity Regularization
"""
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from utils.method_labels import METHOD_LABEL_INLINE

class ShapAwareANFISTrainer:
    """Тренер ANFIS с Baseline-Masked Feature Sensitivity Regularization"""

    def __init__(self, model,config, gamma=0.5, verbose=True):
        self.model = model.network
        self.gamma = gamma
        self.verbose = verbose
        self.task_type = config['dataset']['task_type']
        self.training_time = 0
        self.best_val_score = None
        self.best_epoch = 0
        self.patience = config.get('shap', {}).get('early_stopping_patience', 10)
        self.min_delta = config.get('shap', {}).get('early_stopping_min_delta', 0.001)

        self.method_label_inline = METHOD_LABEL_INLINE

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=25, batch_size=32, lr=0.005):
        """Обучение с Baseline-Masked Feature Sensitivity Regularization с отслеживанием метрик"""
        start_time = time.time()

        # Подготовка данных
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        training_dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        # Базовые значения для SHAP (вычисляем как среднее по обучающей выборке)
        baseline_values = np.mean(X_train, axis=0)

        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Выбор функции потерь в зависимости от типа задачи
        if self.task_type == 'regression':
            loss_function = torch.nn.MSELoss()
        else:
            loss_function = torch.nn.BCELoss()

        # История потерь и метрик
        history = {
            'total_loss': [],
            'main_loss': [],
            'shap_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'epoch_times': []
        }

        if self.verbose:
            task_name = "регрессии" if self.task_type == 'regression' else "классификации"
            print(f"[INFO] Начинаю обучение ANFIS с {self.method_label_inline} ({task_name})...")

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_losses = {'total': [], 'main': [], 'shap': []}

            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()

                # Прямой проход
                self.model.train()
                raw_predictions = self.model(batch_X).squeeze()
                
                # Для классификации применяем сигмоиду для ограничения [0,1]
                if self.task_type == 'classification':
                    predictions = torch.sigmoid(raw_predictions)
                else:
                    predictions = raw_predictions
                
                main_loss = loss_function(predictions, batch_y)

                # SHAP регуляризация (вычисляется с градиентами!)
                shap_importance = self._calculate_shap_approximation_train(batch_X, baseline_values)
                
                # Нормализация SHAP значений
                shap_sum = torch.sum(torch.abs(shap_importance)) + 1e-8
                shap_normalized = shap_importance / shap_sum
                
                # Целевое равномерное распределение
                target_uniform = torch.ones_like(shap_normalized) / len(shap_normalized)
                
                # MSE между нормализованными SHAP и равномерным распределением
                shap_regularization_loss = torch.mean((shap_normalized - target_uniform) ** 2)
                
                # Масштабирование SHAP loss для сопоставимости с main_loss
                # Используем относительное масштабирование для лучшего влияния gamma
                # SHAP loss масштабируется так, чтобы быть сопоставимым с main_loss
                if main_loss.item() > 1e-6:
                    # Масштабируем SHAP loss пропорционально main_loss
                    # Это обеспечивает, что gamma будет иметь заметное влияние
                    scale_factor = main_loss.item() / max(shap_regularization_loss.item(), 1e-8)
                    shap_loss_scaled = shap_regularization_loss * scale_factor
                else:
                    shap_loss_scaled = shap_regularization_loss

                # Общая потеря с масштабированным SHAP loss
                # Gamma теперь будет иметь более заметное влияние
                total_loss = main_loss + self.gamma * shap_loss_scaled

                # Обратное распространение
                total_loss.backward()
                optimizer.step()

                # Сохранение потерь
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['main'].append(main_loss.item())
                epoch_losses['shap'].append(shap_regularization_loss.item())

            # Усреднение потерь по эпохе
            for loss_type in ['total_loss', 'main_loss', 'shap_loss']:
                loss_key = loss_type.split('_')[0]
                history[loss_type].append(np.mean(epoch_losses[loss_key]))

            # Вычисление метрик на обучающей выборке
            self.model.eval()
            with torch.no_grad():
                train_pred = self.predict(X_train)
                train_metrics = self._calculate_metrics(y_train, train_pred)
                history['train_metrics'].append(train_metrics)

                # Метрики на валидационной выборке, если она предоставлена
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_metrics = self._calculate_metrics(y_val, val_pred)
                    history['val_metrics'].append(val_metrics)
                    
                    # Ранняя остановка на основе валидационной метрики
                    val_score = val_metrics.get('f1_score', val_metrics.get('roc_auc', val_metrics.get('accuracy', 0)))
                    if self.best_val_score is None or val_score > self.best_val_score + self.min_delta:
                        self.best_val_score = val_score
                        self.best_epoch = epoch
                        # Сохраняем лучшие веса модели
                        self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    history['val_metrics'].append(None)
                    
            # Время эпохи (записываем перед проверкой ранней остановки)
            epoch_time = time.time() - epoch_start
            history['epoch_times'].append(epoch_time)
            
            # Проверка ранней остановки
            if X_val is not None and y_val is not None and self.patience > 0:
                if epoch - self.best_epoch >= self.patience:
                    if self.verbose:
                        print(f"   [WARNING] Ранняя остановка на эпохе {epoch + 1} (лучшая эпоха: {self.best_epoch + 1})")
                    # Восстанавливаем лучшие веса
                    if hasattr(self, 'best_model_state'):
                        self.model.load_state_dict(self.best_model_state)
                    break

            # Прогресс
            if self.verbose and (epoch + 1) % 5 == 0:
                metrics_str = ""
                if self.task_type == 'regression':
                    metrics_str = f"Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}"
                    if X_val is not None and y_val is not None and history['val_metrics'][-1]:
                        val_m = history['val_metrics'][-1]
                        metrics_str += f" | Val RMSE: {val_m['rmse']:.4f}, R²: {val_m['r2']:.4f}"
                else:
                    metrics_str = f"Train Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}, ROC: {train_metrics['roc_auc']:.4f}"
                    if X_val is not None and y_val is not None and history['val_metrics'][-1]:
                        val_m = history['val_metrics'][-1]
                        gap = train_metrics['accuracy'] - val_m['accuracy']
                        metrics_str += f" | Val Acc: {val_m['accuracy']:.4f}, F1: {val_m['f1_score']:.4f}"
                        if gap > 0.05:
                            metrics_str += f" [WARNING](gap={gap:.3f})"
                
                print(f"   Эпоха {epoch + 1}/{epochs}: "
                      f"Total: {history['total_loss'][-1]:.4f}, "
                      f"Main: {history['main_loss'][-1]:.4f}, "
                      f"SHAP: {history['shap_loss'][-1]:.4f}, "
                      f"{metrics_str}, "
                      f"Время: {epoch_time:.2f}с")

        self.training_time = time.time() - start_time

        if self.verbose:
            print(f"[OK] Обучение завершено за {self.training_time:.2f} сек")

        return history

    def _calculate_metrics(self, y_true, y_pred):
        """Вычисление метрик в зависимости от типа задачи"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if self.task_type == 'regression':
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            y_pred_bin = (y_pred > 0.5).astype(int)
            return {
                'accuracy': accuracy_score(y_true, y_pred_bin),
                'precision': precision_score(y_true, y_pred_bin, zero_division=0),
                'recall': recall_score(y_true, y_pred_bin, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_bin, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
            }

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

    def _calculate_shap_approximation_train(self, X_batch, baseline):
        """Приближенные SHAP значения для обучения (с градиентами!)"""
        # X_batch уже должен быть torch.Tensor
        if not isinstance(X_batch, torch.Tensor):
            X_tensor = torch.tensor(X_batch, dtype=torch.float32, requires_grad=False)
        else:
            X_tensor = X_batch
        
        # Преобразуем baseline в тензор на том же device
        if isinstance(baseline, np.ndarray):
            # Определяем device из X_tensor
            device = X_tensor.device if hasattr(X_tensor, 'device') else 'cpu'
            baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)
        elif isinstance(baseline, torch.Tensor):
            baseline_tensor = baseline.to(X_tensor.device) if hasattr(X_tensor, 'device') else baseline
        else:
            baseline_tensor = baseline

        # Предсказания для исходных данных
        original_predictions = self.model(X_tensor).squeeze()
        
        # Вычисляем важность каждого признака
        shap_values = []
        num_features = X_tensor.shape[1]
        
        for feature_index in range(num_features):
            # Создаем маскированную версию данных
            X_masked = X_tensor.clone()
            X_masked[:, feature_index] = baseline_tensor[feature_index]
            
            # Предсказания для маскированных данных
            masked_predictions = self.model(X_masked).squeeze()
            
            # Важность признака = среднее абсолютное изменение предсказаний
            feature_importance = torch.mean(torch.abs(original_predictions - masked_predictions))
            shap_values.append(feature_importance)
        
        # Возвращаем тензор с градиентами
        return torch.stack(shap_values)
    
    def _calculate_shap_approximation(self, X_batch, baseline):
        """Приближенные SHAP значения для оценки (без градиентов)"""
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X_batch, torch.Tensor):
                X_tensor = torch.tensor(X_batch, dtype=torch.float32)
            else:
                X_tensor = X_batch

            original_predictions = self.model(X_tensor).squeeze().cpu().numpy()
            shap_values = []
            X_numpy = X_tensor.cpu().numpy()
            
            if isinstance(baseline, torch.Tensor):
                baseline_numpy = baseline.cpu().numpy()
            else:
                baseline_numpy = baseline

            for feature_index in range(X_numpy.shape[1]):
                X_masked = X_numpy.copy()
                X_masked[:, feature_index] = baseline_numpy[feature_index]

                X_masked_tensor = torch.tensor(X_masked, dtype=torch.float32)
                masked_predictions = self.model(X_masked_tensor).squeeze().cpu().numpy()

                if np.isscalar(original_predictions) and np.isscalar(masked_predictions):
                    feature_importance = abs(original_predictions - masked_predictions)
                else:
                    feature_importance = np.mean(np.abs(original_predictions - masked_predictions))

                shap_values.append(feature_importance)

        return np.array(shap_values)
