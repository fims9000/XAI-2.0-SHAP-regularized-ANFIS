"""
Модуль для создания визуализаций результатов экспериментов
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                           average_precision_score)
from sklearn.calibration import calibration_curve
from pathlib import Path

class ResultsVisualizer:
    """Класс для создания комплексной визуализации результатов"""

    def __init__(self, config, save_dir=None):
        self.config = config
        self.task_type = config['dataset']['task_type']
        self.save_dir = Path(save_dir) if save_dir else None

        # Настройка стиля графиков
        plt.style.use(config['visualization']['style'])
        sns.set_palette("husl")

    def create_comprehensive_analysis(self, results, feature_names, X_test, y_test):
        """Создание комплексной визуализации всех результатов"""
        print("Создание комплексной визуализации...")

        # Основная визуализация для Vanilla ANFIS
        if 'vanilla' in results:
            if self.task_type == 'regression':
                self._create_regression_analysis(results['vanilla'], feature_names, X_test, y_test)
            else:
                self._create_vanilla_analysis(results['vanilla'], feature_names, X_test, y_test)

        # Визуализация для SHAP regularization
        if 'regularized' in results:
            self._create_shap_regularization_analysis(
                results, feature_names, X_test, y_test
            )

        # Сравнительный анализ
        if len(results) > 1:
            self._create_comparative_analysis(results, feature_names)

        print("Визуализация создана и сохранена!")

    def _create_vanilla_analysis(self, vanilla_results, feature_names, X_test, y_test):
        """Создание анализа для классификации Vanilla ANFIS"""

        fig = plt.figure(figsize=self.config['visualization']['figure_size'])

        y_pred = np.array(vanilla_results['predictions']).flatten()
        y_prob = np.array(vanilla_results['probabilities']).flatten()
        metrics = vanilla_results['metrics']
        feature_importance = vanilla_results['feature_importance']
        training_time = vanilla_results['training_time']
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()

        # 1. Матрица ошибок
        plt.subplot(3, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Матрица ошибок\n(Confusion Matrix)', fontsize=12, fontweight='bold')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')

        # 2. ROC кривая (только для бинарной классификации)
        plt.subplot(3, 3, 2)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная классификация')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC кривая', fontsize=12, fontweight='bold')
            plt.legend(loc="lower right")
        else:
            plt.text(0.5, 0.5, f'ROC кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'ROC кривая ({n_classes} классов)', fontsize=12, fontweight='bold')

        plt.grid(True, alpha=0.3)

        # 3. Важность признаков
        plt.subplot(3, 3, 3)
        top_n = min(15, len(feature_importance))
        sorted_indices = np.argsort(feature_importance)[::-1][:top_n]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        bars = plt.bar(range(len(sorted_importance)), sorted_importance, color=colors)
        plt.title(f'Топ-{top_n} важных признаков\n(ANFIS)', fontsize=12, fontweight='bold')
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.xticks(range(len(sorted_names)), [name[:8] for name in sorted_names],
                  rotation=45, ha='right')

        # 4. Распределение предсказаний (только для бинарной классификации)
        plt.subplot(3, 3, 4)
        if n_classes == 2:
            plt.hist(y_prob[y_test_values == 0], bins=20, alpha=0.7, label='Класс 0',
                    color='lightblue', density=True)
            plt.hist(y_prob[y_test_values == 1], bins=20, alpha=0.7, label='Класс 1',
                    color='lightcoral', density=True)
        else:
            plt.hist(y_prob, bins=20, alpha=0.7, color='lightgreen', density=True)

        plt.xlabel('Предсказанная вероятность')
        plt.ylabel('Плотность')
        plt.title('Распределение вероятностей\nпо классам', fontsize=12, fontweight='bold')
        if n_classes == 2:
            plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Precision-Recall кривая (только для бинарной классификации)
        plt.subplot(3, 3, 5)
        if n_classes == 2:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            plt.plot(recall_curve, precision_curve, color='purple', lw=2,
                    label=f'PR (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall кривая', fontsize=12, fontweight='bold')
            plt.legend()
        else:
            plt.text(0.5, 0.5, f'PR кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Precision-Recall ({n_classes} классов)', fontsize=12, fontweight='bold')

        plt.grid(True, alpha=0.3)

        # 6. Калибровочная кривая (только для бинарной классификации)
        plt.subplot(3, 3, 6)
        if n_classes == 2:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
            plt.plot([0, 1], [0, 1], 'k--', label='Идеально калиброванная')
            plt.plot(mean_predicted_value, fraction_of_positives, 's-', color='red', label='ANFIS')
            plt.xlabel('Средняя предсказанная вероятность')
            plt.ylabel('Доля положительных')
            plt.title('Калибровочная кривая', fontsize=12, fontweight='bold')
            plt.legend()
        else:
            plt.text(0.5, 0.5, f'Калибровка не поддерживается\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Калибровка ({n_classes} классов)', fontsize=12, fontweight='bold')

        plt.grid(True, alpha=0.3)

        # 7. Ранжированная важность признаков (горизонтальная)
        plt.subplot(3, 3, 7)
        top_n_detailed = min(10, len(feature_importance))
        sorted_indices_detailed = np.argsort(feature_importance)[::-1][:top_n_detailed]
        sorted_importance_detailed = feature_importance[sorted_indices_detailed]
        sorted_names_detailed = [feature_names[i] for i in sorted_indices_detailed]

        colors_sorted = plt.cm.plasma(np.linspace(0, 1, top_n_detailed))
        plt.barh(range(top_n_detailed), sorted_importance_detailed, color=colors_sorted)
        plt.yticks(range(top_n_detailed), [name[:15] for name in sorted_names_detailed])
        plt.xlabel('Важность признака')
        plt.title(f'Топ-{top_n_detailed} важных признаков', fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # 8. Распределение ошибок
        plt.subplot(3, 3, 8)
        if n_classes == 2:
            errors = y_prob - y_test_values
        else:
            errors = y_pred - y_test_values

        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Ошибка предсказания')
        plt.ylabel('Частота')
        plt.title('Распределение ошибок\nпредсказания', fontsize=12, fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)

        # 9. Сводная таблица метрик
        plt.subplot(3, 3, 9)
        plt.axis('off')

        metrics_text = f"""
РЕЗУЛЬТАТЫ VANILLA ANFIS

Основные метрики:
• Accuracy: {metrics['accuracy']:.4f}
• Precision: {metrics['precision']:.4f}  
• Recall: {metrics['recall']:.4f}
• F1-Score: {metrics['f1_score']:.4f}
• ROC AUC: {metrics['roc_auc']:.4f}

Матрица ошибок:
• TN: {cm[0,0]}, FP: {cm[0,1]}
• FN: {cm[1,0]}, TP: {cm[1,1]}

Время обучения: {training_time:.2f} сек
"""

        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=3.0)
        plt.suptitle('Результаты обучения Vanilla ANFIS',
                    fontsize=16, fontweight='bold', y=0.98)

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'vanilla_anfis_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.show()

    def _create_regression_analysis(self, vanilla_results, feature_names, X_test, y_test):
        """Создание анализа для регрессионной задачи"""

        fig = plt.figure(figsize=self.config['visualization']['figure_size'])

        y_pred = np.array(vanilla_results['predictions']).flatten()
        metrics = vanilla_results['metrics']
        feature_importance = vanilla_results['feature_importance']
        training_time = vanilla_results['training_time']
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()

        # 1. Scatter plot: реальные vs предсказанные
        plt.subplot(3, 3, 1)
        plt.scatter(y_test_values, y_pred, alpha=0.6, color='deepskyblue')
        plt.plot([y_test_values.min(), y_test_values.max()], [y_test_values.min(), y_test_values.max()], 'r--', lw=2)
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Реальные vs Предсказанные', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 2. Остатки (residuals)
        plt.subplot(3, 3, 2)
        residuals = y_test_values - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title('График остатков', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 3. Важность признаков
        plt.subplot(3, 3, 3)
        top_n = min(15, len(feature_importance))
        sorted_indices = np.argsort(feature_importance)[::-1][:top_n]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        bars = plt.bar(range(len(sorted_importance)), sorted_importance, color=colors)
        plt.title(f'Топ-{top_n} важных признаков', fontsize=12, fontweight='bold')
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.xticks(range(len(sorted_names)), [name[:8] for name in sorted_names],
                   rotation=45, ha='right')

        # 4. Гистограмма остатков
        plt.subplot(3, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Остатки')
        plt.ylabel('Частота')
        plt.title('Распределение остатков', fontsize=12, fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)

        # 5. Q-Q plot для проверки нормальности остатков
        plt.subplot(3, 3, 5)
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot остатков', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
        except ImportError:
            plt.text(0.5, 0.5, 'scipy не установлен\nдля Q-Q plot',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Q-Q Plot остатков', fontsize=12, fontweight='bold')

        # 6. Временной ряд предсказаний
        plt.subplot(3, 3, 6)
        indices = np.arange(len(y_test_values))
        plt.plot(indices, y_test_values, color='green', alpha=0.8, label="Реальные", linewidth=2)
        plt.plot(indices, y_pred, color='red', alpha=0.8, label="Предсказанные", linewidth=2)
        plt.xlabel('Индекс образца')
        plt.ylabel('Значение')
        plt.title('Временной ряд', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 7. Ранжированная важность признаков (горизонтальная)
        plt.subplot(3, 3, 7)
        top_n_detailed = min(10, len(feature_importance))
        sorted_indices_detailed = np.argsort(feature_importance)[::-1][:top_n_detailed]
        sorted_importance_detailed = feature_importance[sorted_indices_detailed]
        sorted_names_detailed = [feature_names[i] for i in sorted_indices_detailed]

        colors_sorted = plt.cm.plasma(np.linspace(0, 1, top_n_detailed))
        plt.barh(range(top_n_detailed), sorted_importance_detailed, color=colors_sorted)
        plt.yticks(range(top_n_detailed), [name[:15] for name in sorted_names_detailed])
        plt.xlabel('Важность признака')
        plt.title(f'Топ-{top_n_detailed} важных признаков', fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # 8. Распределение предсказаний vs реальных
        plt.subplot(3, 3, 8)
        plt.hist(y_test_values, bins=30, alpha=0.7, color='lightblue',
                 label='Реальные', density=True, edgecolor='black')
        plt.hist(y_pred, bins=30, alpha=0.7, color='lightcoral',
                 label='Предсказанные', density=True, edgecolor='black')
        plt.xlabel('Значение')
        plt.ylabel('Плотность')
        plt.title('Распределение значений', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 9. Сводная таблица метрик
        plt.subplot(3, 3, 9)
        plt.axis('off')

        metrics_text = f"""
РЕЗУЛЬТАТЫ РЕГРЕССИИ ANFIS

Основные метрики:
• RMSE: {metrics['rmse']:.4f}
• MAE: {metrics['mae']:.4f}
• R²: {metrics['r2']:.4f}

Диапазон значений:
• Min реальных: {y_test_values.min():.2f}
• Max реальных: {y_test_values.max():.2f}
• Min предсказанных: {y_pred.min():.2f}
• Max предсказанных: {y_pred.max():.2f}

Статистика остатков:
• Mean остатков: {residuals.mean():.4f}
• Std остатков: {residuals.std():.4f}

Время обучения: {training_time:.2f} сек
"""

        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=3.0)
        plt.suptitle('Результаты регрессии Vanilla ANFIS',
                     fontsize=16, fontweight='bold', y=0.98)

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'vanilla_anfis_regression_analysis.png',
                        dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.show()

    def _create_shap_regularization_analysis(self, results, feature_names, X_test, y_test):
        """Анализ SHAP регуляризации в сравнении с Vanilla"""

        if 'vanilla' not in results or 'regularized' not in results:
            return

        vanilla_results = results['vanilla']
        shapreg_results = results['regularized']

        # Определяем тип задачи и вызываем соответствующий метод
        if self.task_type == 'regression':
            self._create_regression_shap_analysis(
                vanilla_results, shapreg_results, feature_names, X_test, y_test
            )
        else:
            self._create_classification_shap_analysis(
                vanilla_results, shapreg_results, feature_names, X_test, y_test
            )

    def _create_regression_shap_analysis(self, vanilla_results, shapreg_results, feature_names, X_test, y_test):
        """SHAP анализ для регрессии"""

        fig = plt.figure(figsize=self.config['visualization']['figure_size'])

        y_pred_vanilla = np.array(vanilla_results['predictions']).flatten()
        y_pred_shapreg = np.array(shapreg_results['predictions']).flatten()
        feature_imp_vanilla = vanilla_results['feature_importance']
        feature_imp_shapreg = shapreg_results['feature_importance']
        history = shapreg_results['history']

        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()

        # 1. Динамика лоссов
        plt.subplot(2, 3, 1)
        plt.plot(history['total_loss'], label='Total Loss', linewidth=2, color='red')
        plt.plot(history['main_loss'], label='Main Loss (MSE)', linewidth=2, color='blue')
        plt.plot(history['shap_loss'], label='SHAP Loss', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Динамика лоссов\n(SHAP-регуляризация)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Сравнение предсказаний - Scatter plot
        plt.subplot(2, 3, 2)
        plt.scatter(y_test_values, y_pred_vanilla, alpha=0.6, color='blue', label='Vanilla ANFIS', s=20)
        plt.scatter(y_test_values, y_pred_shapreg, alpha=0.6, color='darkgreen', label='SHAP-reg ANFIS', s=20)
        plt.plot([y_test_values.min(), y_test_values.max()], [y_test_values.min(), y_test_values.max()], 'r--', lw=2)
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Сравнение предсказаний', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Сравнение важности признаков
        plt.subplot(2, 3, 3)
        top_n = min(10, len(feature_names))
        x_pos = np.arange(top_n)
        width = 0.35

        # Нормализация для сравнения
        vanilla_norm = feature_imp_vanilla / np.max(feature_imp_vanilla)
        shapreg_norm = feature_imp_shapreg / np.max(feature_imp_shapreg)

        plt.bar(x_pos - width/2, vanilla_norm[:top_n], width,
               label='Vanilla ANFIS', color='lightblue', alpha=0.8)
        plt.bar(x_pos + width/2, shapreg_norm[:top_n], width,
               label='SHAP-регуляризация', color='lightgreen', alpha=0.8)
        plt.title(f'Сравнение важности\nпризнаков (топ-{top_n})', fontsize=12, fontweight='bold')
        plt.xlabel('Признаки')
        plt.ylabel('Нормализованная важность')
        plt.xticks(x_pos, [feature_names[i][:8] for i in range(top_n)], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Сравнение метрик
        plt.subplot(2, 3, 4)
        metrics_names = ['RMSE', 'MAE', 'R²']
        vanilla_scores = [vanilla_results['metrics'][k] for k in ['rmse', 'mae', 'r2']]
        shapreg_scores = [shapreg_results['metrics'][k] for k in ['rmse', 'mae', 'r2']]

        x = np.arange(len(metrics_names))
        plt.bar(x - 0.2, vanilla_scores, 0.4, label='Vanilla ANFIS', alpha=0.7)
        plt.bar(x + 0.2, shapreg_scores, 0.4, label='SHAP-регуляризация', alpha=0.7)
        plt.xlabel('Метрики')
        plt.ylabel('Значения')
        plt.title('Сравнение метрик\nмоделей', fontsize=12, fontweight='bold')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Корреляция предсказаний
        plt.subplot(2, 3, 5)
        correlation = np.corrcoef(y_pred_vanilla, y_pred_shapreg)[0, 1]
        plt.scatter(y_pred_vanilla, y_pred_shapreg, alpha=0.6, c='purple')
        min_val = min(y_pred_vanilla.min(), y_pred_shapreg.min())
        max_val = max(y_pred_vanilla.max(), y_pred_shapreg.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Vanilla ANFIS предсказания')
        plt.ylabel('SHAP-регуляризованные предсказания')
        plt.title(f'Корреляция предсказаний\n(r = {correlation:.3f})', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 6. Сводная таблица
        plt.subplot(2, 3, 6)
        plt.axis('off')

        comparison_text = f"""
СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕГРЕССИИ

Vanilla ANFIS:
• RMSE: {vanilla_results['metrics']['rmse']:.4f}
• MAE: {vanilla_results['metrics']['mae']:.4f}
• R²: {vanilla_results['metrics']['r2']:.4f}

SHAP-регуляризация:
• RMSE: {shapreg_results['metrics']['rmse']:.4f}
• MAE: {shapreg_results['metrics']['mae']:.4f}
• R²: {shapreg_results['metrics']['r2']:.4f}

Улучшения:
• ΔRMSE: {shapreg_results['metrics']['rmse'] - vanilla_results['metrics']['rmse']:+.4f}
• ΔMAE: {shapreg_results['metrics']['mae'] - vanilla_results['metrics']['mae']:+.4f}
• ΔR²: {shapreg_results['metrics']['r2'] - vanilla_results['metrics']['r2']:+.4f}

Корреляция предсказаний: {correlation:.3f}
"""

        plt.text(0.1, 0.9, comparison_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=3.0)
        plt.suptitle('Сравнительный анализ регрессии ANFIS с SHAP-регуляризацией',
                    fontsize=16, fontweight='bold', y=0.98)

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'shap_regularization_regression_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.show()

    def _create_classification_shap_analysis(self, vanilla_results, shapreg_results, feature_names, X_test, y_test):
        """SHAP анализ для классификации"""

        fig = plt.figure(figsize=self.config['visualization']['figure_size'])

        y_prob_vanilla = np.array(vanilla_results['probabilities']).flatten()
        y_prob_shapreg = np.array(shapreg_results['probabilities']).flatten()
        feature_imp_vanilla = vanilla_results['feature_importance']
        feature_imp_shapreg = shapreg_results['feature_importance']
        history = shapreg_results['history']

        n_classes = len(np.unique(y_test))

        # 1. Динамика лоссов
        plt.subplot(2, 3, 1)
        plt.plot(history['total_loss'], label='Total Loss', linewidth=2, color='red')
        plt.plot(history['main_loss'], label='Main Loss (BCE)', linewidth=2, color='blue')
        plt.plot(history['shap_loss'], label='SHAP Loss', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Динамика лоссов\n(SHAP-регуляризация)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. ROC кривые сравнение (только для бинарной классификации)
        plt.subplot(2, 3, 2)

        if n_classes == 2:
            fpr_vanilla, tpr_vanilla, _ = roc_curve(y_test, y_prob_vanilla)
            fpr_shapreg, tpr_shapreg, _ = roc_curve(y_test, y_prob_shapreg)

            plt.plot(fpr_vanilla, tpr_vanilla, color='blue', lw=2,
                    label=f'Vanilla AUC = {vanilla_results["metrics"]["roc_auc"]:.3f}')
            plt.plot(fpr_shapreg, tpr_shapreg, color='darkgreen', lw=2,
                    label=f'SHAP-reg AUC = {shapreg_results["metrics"]["roc_auc"]:.3f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Сравнение ROC кривых', fontsize=12, fontweight='bold')
            plt.legend(loc="lower right")
        else:
            plt.text(0.5, 0.5, f'ROC кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROC кривые (многоклассовая)', fontsize=12, fontweight='bold')

        plt.grid(True, alpha=0.3)

        # 3. Сравнение важности признаков
        plt.subplot(2, 3, 3)
        top_n = min(10, len(feature_names))
        x_pos = np.arange(top_n)
        width = 0.35

        # Нормализация для сравнения
        vanilla_norm = feature_imp_vanilla / np.max(feature_imp_vanilla)
        shapreg_norm = feature_imp_shapreg / np.max(feature_imp_shapreg)

        plt.bar(x_pos - width/2, vanilla_norm[:top_n], width,
               label='Vanilla ANFIS', color='lightblue', alpha=0.8)
        plt.bar(x_pos + width/2, shapreg_norm[:top_n], width,
               label='SHAP-регуляризация', color='lightgreen', alpha=0.8)
        plt.title(f'Сравнение важности\nпризнаков (топ-{top_n})', fontsize=12, fontweight='bold')
        plt.xlabel('Признаки')
        plt.ylabel('Нормализованная важность')
        plt.xticks(x_pos, [feature_names[i][:8] for i in range(top_n)], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Сравнение метрик
        plt.subplot(2, 3, 4)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        vanilla_scores = [vanilla_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        shapreg_scores = [shapreg_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]

        x = np.arange(len(metrics_names))
        plt.bar(x - 0.2, vanilla_scores, 0.4, label='Vanilla ANFIS', alpha=0.7)
        plt.bar(x + 0.2, shapreg_scores, 0.4, label='SHAP-регуляризация', alpha=0.7)
        plt.xlabel('Метрики')
        plt.ylabel('Значения')
        plt.title('Сравнение метрик\nмоделей', fontsize=12, fontweight='bold')
        plt.xticks(x, metrics_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Корреляция предсказаний
        plt.subplot(2, 3, 5)
        correlation = np.corrcoef(y_prob_vanilla, y_prob_shapreg)[0, 1]
        plt.scatter(y_prob_vanilla, y_prob_shapreg, alpha=0.6, c='purple')
        min_val = min(y_prob_vanilla.min(), y_prob_shapreg.min())
        max_val = max(y_prob_vanilla.max(), y_prob_shapreg.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Vanilla ANFIS предсказания')
        plt.ylabel('SHAP-регуляризованные предсказания')
        plt.title(f'Корреляция предсказаний\n(r = {correlation:.3f})', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 6. Сводная таблица
        plt.subplot(2, 3, 6)
        plt.axis('off')

        comparison_text = f"""
СРАВНИТЕЛЬНЫЙ АНАЛИЗ

Vanilla ANFIS:
• Accuracy: {vanilla_results['metrics']['accuracy']:.4f}
• ROC AUC: {vanilla_results['metrics']['roc_auc']:.4f}
• F1-Score: {vanilla_results['metrics']['f1_score']:.4f}

SHAP-регуляризация:
• Accuracy: {shapreg_results['metrics']['accuracy']:.4f}
• ROC AUC: {shapreg_results['metrics']['roc_auc']:.4f}
• F1-Score: {shapreg_results['metrics']['f1_score']:.4f}

Улучшения:
• ΔAccuracy: {shapreg_results['metrics']['accuracy'] - vanilla_results['metrics']['accuracy']:+.4f}
• ΔAUC: {shapreg_results['metrics']['roc_auc'] - vanilla_results['metrics']['roc_auc']:+.4f}
• ΔF1: {shapreg_results['metrics']['f1_score'] - vanilla_results['metrics']['f1_score']:+.4f}

Корреляция предсказаний: {correlation:.3f}
"""

        plt.text(0.1, 0.9, comparison_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=3.0)
        plt.suptitle('Сравнительный анализ ANFIS с SHAP-регуляризацией',
                    fontsize=16, fontweight='bold', y=0.98)

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'shap_regularization_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.show()

    def _create_comparative_analysis(self, results, feature_names):
        """Создание сравнительного анализа всех методов"""

        # Создание сводной таблицы результатов
        comparison_data = []
        for method_name, method_results in results.items():
            if 'metrics' in method_results:
                row = {'Method': method_name.title()}
                row.update(method_results['metrics'])
                row['Training_Time'] = method_results.get('training_time', 0)
                comparison_data.append(row)

        # Сохранение таблицы в CSV
        if self.save_dir and comparison_data:
            import pandas as pd
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_csv(self.save_dir / 'comparison_results.csv', index=False)
            print(f"Сравнительная таблица сохранена: {self.save_dir / 'comparison_results.csv'}")
