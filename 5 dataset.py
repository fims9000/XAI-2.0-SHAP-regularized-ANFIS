"""# Fifth dataset

## Предобработка

Источник и описание датасета можно найти :
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
"""

# Импорты
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Настройки для красивых графиков
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

!pip install --no-deps xanfis mealpy permetrics

from xanfis import BioAnfisClassifier
from scipy.stats import pearsonr

# === Информация о датасете ===
data = pd.read_csv("./winequality-red.csv")

data.head()

# === Подготовка признаков и целевой переменной ===
X = data.iloc[:, :-1].copy()
y = data['quality']
feature_names = X.columns.tolist()

# Масштабирование и разбиение на train/test
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"✅ Данные подготовлены:")
print(f"   Train: {X_train.shape[0]} объектов")
print(f"   Test:  {X_test.shape[0]} объектов")
print(f"   Всего признаков: {X.shape[1]}")

"""## Ванильная Анфис"""

# ===============================================================================
# Обучение стандартного ANFIS с визуализацией результатов
# ===============================================================================
import seaborn as sns
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                           roc_curve, precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score)
import pandas as pd
from xanfis import BioAnfisRegressor

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

print("🔵 Начинаю обучение стандартного ANFIS...")
start_time = time.time()

# Создание и настройка модели ANFIS
model_vanilla = BioAnfisRegressor(
    num_rules=33,                    # Количество нечётких правил
    mf_class='Sigmoid',              # Тип функций принадлежности
    optim='BaseGA',           # Алгоритм оптимизации (роевая оптимизация)
    optim_params={
        'epoch': 100,               # Количество эпох обучения
        'pop_size': 30,            # Размер популяции для PSO
        'verbose': True           # Отключить подробный вывод PSO
    },
    reg_lambda=0.1,                # Коэффициент регуляризации
    seed=42,                       # Фиксированное зерно для воспроизводимости
    verbose=True
)

# Обучение модели
print("📚 Процесс обучения...")
model_vanilla.fit(X_train, y_train)
vanilla_time = time.time() - start_time

# Получение предсказаний
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error, mean_absolute_error,r2_score)
print("🔍 Получение предсказаний...")
y_pred_vanilla = model_vanilla.predict(X_test)
model_vanilla.network.eval()
#y_prob_vanilla = model_vanilla.predict_proba(X_test)

# Вычисление метрик качества


from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

num_classes = len(np.unique(y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred_vanilla))
mae  = mean_absolute_error(y_test, y_pred_vanilla)
r2   = r2_score(y_test, y_pred_vanilla)

# Извлечение важности признаков из коэффициентов модели
model_coefficients = model_vanilla.network.state_dict()['coeffs'].detach().cpu().numpy()
feature_importance_vanilla = np.sum(np.abs(model_coefficients[:, :-1, 0]), axis=0)

# Вывод результатов обучения
print(f"\n✅ Обучение ANFIS завершено успешно!")
print(f"   📊 RMSE: {rmse:.4f}")
print(f"   🎯 MAE: {mae:.4f}")
print(f"   📈 R2: {r2:.4f}")

print(f"   ⏱️  Время обучения: {vanilla_time:.2f} секунд")

plt.scatter(y_test, y_pred_vanilla, s=40, alpha=0.8, color='deepskyblue', label="Пары", zorder=1)
plt.scatter(y_test, y_test, s=40, alpha=0.3, color='limegreen', label="y = x", zorder=2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.xlabel('Реальные')
plt.ylabel('Предсказанные')
plt.title('Scatter')
plt.legend()

# ===============================================================================
# Создание комплексной визуализации результатов
# ===============================================================================
alpha=0.8
plt.scatter(range(len(y_test)), y_test, color='green',alpha=alpha, label="Реальные", s=40)
plt.scatter(range(len(y_pred_vanilla)), y_pred_vanilla, color='red',alpha=alpha ,label="Предсказанные", marker='x', s=40)
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Scatter: Реальные и предсказанные по индексу')
plt.legend()

"""## SHAP Post-hoc"""

!pip install shap

# ===============================================================================
# ЯЧЕЙКА 5: Вычисление SHAP Post-hoc для vanilla модели
# ===============================================================================
import shap
import time

def predict_fn(data):
  model_vanilla.network.eval()
  return np.asarray(model_vanilla.predict(data)).ravel()
print("🟢 Вычисляю SHAP через библиотеку shap...")
start_time = time.time()

explainer = shap.KernelExplainer(predict_fn, X_test[:100])
shap_values = explainer.shap_values(X_test[:100])
posthoc_time = vanilla_time + time.time() - start_time  # Время расчёта
shap_posthoc = np.abs(shap_values).mean(axis=0)  # Глобальная (средняя) важность
print(f"✅ SHAP Post-hoc вычислен: размер={shap_values.shape}, время={posthoc_time:.2f} c.")

shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

shap.summary_plot(shap_values, X_test[:100],
                  feature_names=feature_names, plot_type="bar")

shap.decision_plot(explainer.expected_value, shap_values,
                   X_test[:100], feature_names=feature_names,
                   feature_display_range=slice(None))

#local
X_sample_rounded = np.round(X_test[0], 2)
shap.force_plot(explainer.expected_value,
                shap_values[0],
                X_sample_rounded,
                feature_names=feature_names,
                matplotlib=True)

shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                      base_values=explainer.expected_value,
                                      data=X_test[0],
                                      feature_names=feature_names))

"""## Обучение с Shap подкрелпением"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class ShapAwareANFISTrainer:
    """
    Обучение ANFIS с дифференцируемой SHAP-регуляризацией.
    """

    def __init__(self, model, gamma=0.15, verbose=True):
        self.model = model.network
        self.gamma = gamma
        self.verbose = verbose
        self.training_time = 0

    def _calculate_shap_approximation(self, X_batch, baseline):
        self.model.eval()
        X_tensor = X_batch if isinstance(X_batch, torch.Tensor) else torch.tensor(X_batch, dtype=torch.float32)

        original_predictions = self.model(X_tensor).squeeze()

        shap_values = []
        for feature_index in range(X_tensor.shape[1]):
            X_masked = X_tensor.clone()
            X_masked[:, feature_index] = baseline[feature_index]

            masked_predictions = self.model(X_masked).squeeze()

            feature_importance = torch.mean(torch.abs(original_predictions - masked_predictions))
            shap_values.append(feature_importance)

        return torch.stack(shap_values)

    def fit(self, X_train, y_train, epochs=25, batch_size=32, lr=0.005):
        start_time = time.time()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        baseline = torch.mean(X_tensor, dim=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        training_history = {'total_loss': [], 'main_loss': [], 'shap_loss': []}

        if self.verbose:
            print("Начинаю обучение ANFIS с SHAP-регуляризацией...")

        for epoch in range(epochs):
            epoch_total_loss = []
            epoch_main_loss = []
            epoch_shap_loss = []

            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                self.model.train()

                predictions = self.model(batch_X).squeeze()
                main_loss = loss_fn(predictions, batch_y)

                shap_importance = self._calculate_shap_approximation(batch_X, baseline)
                shap_normalized = shap_importance / (torch.sum(shap_importance) + 1e-8)
                target_uniform = torch.ones_like(shap_normalized) / shap_normalized.numel()

                shap_loss = torch.mean((shap_normalized - target_uniform) ** 2)

                total_loss = main_loss + self.gamma * shap_loss
                total_loss.backward()
                optimizer.step()

                epoch_total_loss.append(total_loss.item())
                epoch_main_loss.append(main_loss.item())
                epoch_shap_loss.append(shap_loss.item())

            training_history['total_loss'].append(np.mean(epoch_total_loss))
            training_history['main_loss'].append(np.mean(epoch_main_loss))
            training_history['shap_loss'].append(np.mean(epoch_shap_loss))

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"Эпоха {epoch+1}/{epochs}: "
                      f"Общая потеря: {training_history['total_loss'][-1]:.6f}, "
                      f"Основная потеря: {training_history['main_loss'][-1]:.6f}, "
                      f"SHAP регуляризация: {training_history['shap_loss'][-1]:.6f}")

        self.training_time = time.time() - start_time
        if self.verbose:
            print(f"Обучение завершено за {self.training_time:.2f} секунд")

        return training_history

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        return predictions

    def get_global_shap_importance(self, X_sample):
        baseline = torch.mean(torch.tensor(X_sample, dtype=torch.float32), dim=0)
        return self._calculate_shap_approximation(torch.tensor(X_sample, dtype=torch.float32), baseline).detach().cpu().numpy()

print("🟠 Обучение ANFIS с SHAP-регуляризацией...")
import time
# Создание базовой модели для инициализации
model_shapreg = BioAnfisRegressor(
    num_rules=7,                    # Количество нечётких правил
    mf_class='Sigmoid',              # Тип функций принадлежности
    optim='BaseGA',           # Алгоритм оптимизации (роевая оптимизация)
    optim_params={
        'epoch': 1,               # Количество эпох обучения
        'pop_size': 30,            # Размер популяции для PSO
        'verbose': False          # Отключить подробный вывод PSO
    },
    reg_lambda=0.1,                # Коэффициент регуляризации
    seed=42,                       # Фиксированное зерно для воспроизводимости
    verbose=True
)
model_shapreg.fit(X_train[:50], y_train.values[:50])  # мини-инициализация
trainer = ShapAwareANFISTrainer(model_shapreg, gamma=0.5, verbose=True)

start_time = time.time()
trainer.fit(X_train, y_train.values, epochs=25)
shapreg_time = time.time() - start_time

# Получение предсказаний
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error, mean_absolute_error,r2_score)
print("🔍 Получение предсказаний...")
y_pred_shapreg = model_shapreg.predict(X_test)
model_shapreg.network.eval()

# Вычисление метрик качества


from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

num_classes = len(np.unique(y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred_shapreg))
mae  = mean_absolute_error(y_test, y_pred_shapreg)
r2   = r2_score(y_test, y_pred_shapreg)

# Извлечение важности признаков из коэффициентов модели
model_coefficients = model_shapreg.network.state_dict()['coeffs'].detach().cpu().numpy()
feature_importance_shapreg = np.sum(np.abs(model_coefficients[:, :-1, 0]), axis=0)

# Вывод результатов обучения
print(f"✅ SHAP-регуляризованное ANFIS обучение:")
print(f"   📊 RMSE: {rmse:.4f}")
print(f"   🎯 MAE: {mae:.4f}")
print(f"   📈 R2: {r2:.4f}")

print(f"   ⏱️  Время обучения: {shapreg_time:.2f} секунд")