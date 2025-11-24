"""
Модуль для создания визуализаций результатов экспериментов
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from pathlib import Path

from utils.method_labels import (
    METHOD_ALIAS_RU,
    METHOD_LABEL_COMPACT,
    METHOD_LABEL_INLINE,
    METHOD_MODEL_LABEL,
    METHOD_MODEL_LABEL_RU,
    METHOD_NAME_EN,
)

class ResultsVisualizer:
    """Класс для создания комплексной визуализации результатов"""

    def __init__(self, config, save_dir=None):
        self.config = config
        self.task_type = config['dataset']['task_type']
        self.save_dir = Path(save_dir) if save_dir else None
        self.dpi = config['visualization'].get('dpi', 300)  # Высокое качество по умолчанию
        self.save_individual_plots = True  # Сохранять каждый график отдельно
        self.method_name_en = METHOD_NAME_EN
        self.method_alias = METHOD_ALIAS_RU
        self.method_label_inline = METHOD_LABEL_INLINE
        self.method_label_compact = METHOD_LABEL_COMPACT
        self.method_model_label = METHOD_MODEL_LABEL
        self.method_model_label_ru = METHOD_MODEL_LABEL_RU

        # Настройка строгого стиля для военной презентации
        plt.style.use(config['visualization']['style'])
        sns.set_palette("muted")
        
        # Настройка параметров matplotlib для профессионального вида
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'grid.alpha': 0.3,
            'lines.linewidth': 2.0,
            'patch.linewidth': 1.2
        })
    
    def _save_individual_plot(self, fig, ax, filename, title=None):
        """Сохраняет отдельный subplot как отдельный PNG файл с высоким качеством"""
        if not self.save_dir or not self.save_individual_plots:
            return
        
        # Создаем новую фигуру для отдельного графика
        fig_individual = plt.figure(figsize=(10, 8))
        ax_individual = fig_individual.add_subplot(111)
        
        # Копируем содержимое subplot'а
        # Получаем все элементы из исходного axes
        for child in ax.get_children():
            if hasattr(child, 'get_data'):
                # Линии
                data = child.get_data()
                if len(data) == 2:
                    ax_individual.plot(data[0], data[1], 
                                     color=child.get_color() if hasattr(child, 'get_color') else 'blue',
                                     linewidth=child.get_linewidth() if hasattr(child, 'get_linewidth') else 2,
                                     label=child.get_label() if hasattr(child, 'get_label') else '',
                                     linestyle=child.get_linestyle() if hasattr(child, 'get_linestyle') else '-')
        
        # Копируем заголовки и метки
        ax_individual.set_title(ax.get_title() if title is None else title, 
                               fontsize=14, fontweight='bold', pad=15)
        ax_individual.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='bold')
        ax_individual.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='bold')
        
        # Копируем легенду если есть
        if ax.get_legend():
            ax_individual.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Копируем сетку
        ax_individual.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Сохранение с высоким DPI
        save_path = self.save_dir / filename
        fig_individual.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
        plt.close(fig_individual)

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
        """Создание улучшенного анализа для классификации Vanilla ANFIS"""

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(22, 14))

        y_pred = np.array(vanilla_results['predictions']).flatten()
        y_prob = np.array(vanilla_results['probabilities']).flatten()
        metrics = vanilla_results['metrics']
        feature_importance = vanilla_results['feature_importance']
        training_time = vanilla_results['training_time']
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()

        # Анализ переобучения на основе метрик
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        overfitting_warning = ""
        if precision >= 0.99 and recall >= 0.99:
            overfitting_warning = "[WARNING] Precision и Recall близки к 1.0 - возможное переобучение!"
        elif precision >= 0.99:
            overfitting_warning = "[INFO] Precision = 1.0 - все предсказанные положительные классы правильные"

        # 1. Матрица ошибок (улучшенная)
        plt.subplot(3, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                   cbar_kws={'shrink': 0.8}, linewidths=2, linecolor='black',
                   annot_kws={'size': 12, 'weight': 'bold'})
        plt.title('Матрица ошибок\n(Confusion Matrix)', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Предсказанный класс', fontsize=11, fontweight='bold')
        plt.ylabel('Истинный класс', fontsize=11, fontweight='bold')
        plt.tick_params(labelsize=10)

        # 2. ROC кривая (улучшенная)
        plt.subplot(3, 3, 2)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, color='darkorange', lw=3, alpha=0.8,
                    label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7, 
                    label='Случайная классификация')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            plt.title('ROC кривая', fontsize=13, fontweight='bold', pad=10)
            plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tick_params(labelsize=9)
        else:
            plt.text(0.5, 0.5, f'ROC кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
            plt.title(f'ROC кривая ({n_classes} классов)', fontsize=13, fontweight='bold', pad=10)

        # 3. Важность признаков (улучшенная)
        plt.subplot(3, 3, 3)
        top_n = min(15, len(feature_importance))
        sorted_indices = np.argsort(feature_importance)[::-1][:top_n]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        bars = plt.bar(range(len(sorted_importance)), sorted_importance, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1)
        plt.title(f'Топ-{top_n} важных признаков\n(ANFIS)', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Признаки', fontsize=11, fontweight='bold')
        plt.ylabel('Важность', fontsize=11, fontweight='bold')
        plt.xticks(range(len(sorted_names)), [name[:10] for name in sorted_names],
                  rotation=45, ha='right', fontsize=8)
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tick_params(labelsize=9)

        # 4. Распределение предсказаний (улучшенное)
        plt.subplot(3, 3, 4)
        if n_classes == 2:
            plt.hist(y_prob[y_test_values == 0], bins=20, alpha=0.7, label='Класс 0',
                    color='#3498db', density=True, edgecolor='navy', linewidth=1)
            plt.hist(y_prob[y_test_values == 1], bins=20, alpha=0.7, label='Класс 1',
                    color='#e74c3c', density=True, edgecolor='darkred', linewidth=1)
        else:
            plt.hist(y_prob, bins=20, alpha=0.7, color='lightgreen', density=True, edgecolor='darkgreen', linewidth=1)

        plt.xlabel('Предсказанная вероятность', fontsize=11, fontweight='bold')
        plt.ylabel('Плотность', fontsize=11, fontweight='bold')
        plt.title('Распределение вероятностей\nпо классам', fontsize=13, fontweight='bold', pad=10)
        if n_classes == 2:
            plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=9)

        # 5. Precision-Recall кривая (улучшенная)
        plt.subplot(3, 3, 5)
        if n_classes == 2:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            plt.plot(recall_curve, precision_curve, color='purple', lw=2.5, alpha=0.8,
                    label=f'PR (AP = {avg_precision:.4f})')
            plt.xlabel('Recall', fontsize=11, fontweight='bold')
            plt.ylabel('Precision', fontsize=11, fontweight='bold')
            plt.title('Precision-Recall кривая', fontsize=13, fontweight='bold', pad=10)
            plt.legend(loc='best', fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tick_params(labelsize=9)
        else:
            plt.text(0.5, 0.5, f'PR кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
            plt.title(f'Precision-Recall ({n_classes} классов)', fontsize=13, fontweight='bold', pad=10)

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

        # 9. Сводная таблица метрик (улучшенная)
        plt.subplot(3, 3, 9)
        plt.axis('off')

        metrics_text = f"""РЕЗУЛЬТАТЫ VANILLA ANFIS
========================================

Основные метрики:
  Accuracy:  {metrics['accuracy']:.4f}
  Precision: {metrics['precision']:.4f}  
  Recall:    {metrics['recall']:.4f}
  F1-Score:  {metrics['f1_score']:.4f}
  ROC AUC:   {metrics['roc_auc']:.4f}

Матрица ошибок:
  TN: {cm[0,0]:4d}, FP: {cm[0,1]:4d}
  FN: {cm[1,0]:4d}, TP: {cm[1,1]:4d}

Время обучения: {training_time:.2f} сек

{overfitting_warning}
"""

        # Цвет фона зависит от наличия предупреждения
        bg_color = 'lightyellow' if overfitting_warning else 'lightblue'
        edge_color = 'orange' if overfitting_warning else 'navy'

        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9, 
                         edgecolor=edge_color, linewidth=2))

        plt.tight_layout(pad=3.0)
        title = 'Результаты обучения Vanilla ANFIS'
        if overfitting_warning:
            title += ' [WARNING]'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'vanilla_anfis_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

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
        plt.close()

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
        plt.title(
            f'Динамика лоссов\n({self.method_label_inline})',
            fontsize=12,
            fontweight='bold'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Сравнение предсказаний - Scatter plot
        plt.subplot(2, 3, 2)
        plt.scatter(y_test_values, y_pred_vanilla, alpha=0.6, color='blue', label='Vanilla ANFIS', s=20)
        plt.scatter(
            y_test_values,
            y_pred_shapreg,
            alpha=0.6,
            color='darkgreen',
            label=self.method_model_label_ru,
            s=20
        )
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
        plt.bar(
            x_pos + width/2,
            shapreg_norm[:top_n],
            width,
            label=self.method_label_inline,
            color='lightgreen',
            alpha=0.8
        )
        plt.title(f'Сравнение важности\nSHAP-признаков (топ-{top_n})', fontsize=12, fontweight='bold')
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
        plt.bar(x + 0.2, shapreg_scores, 0.4, label=self.method_label_inline, alpha=0.7)
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
        plt.ylabel(f'Предсказания после {self.method_label_inline}')
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

{self.method_label_inline}:
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
        plt.suptitle(
            f'Сравнительный анализ регрессии: Vanilla ANFIS vs {self.method_model_label_ru}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'shap_regularization_regression_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def _create_classification_shap_analysis(self, vanilla_results, shapreg_results, feature_names, X_test, y_test):
        """SHAP анализ для классификации (улучшенный)"""

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(22, 14))

        y_prob_vanilla = np.array(vanilla_results['probabilities']).flatten()
        y_prob_shapreg = np.array(shapreg_results['probabilities']).flatten()
        feature_imp_vanilla = vanilla_results['feature_importance']
        feature_imp_shapreg = shapreg_results['feature_importance']
        history = shapreg_results['history']

        n_classes = len(np.unique(y_test))
        epochs = range(1, len(history['total_loss']) + 1)

        # 1. Динамика лоссов (улучшенная)
        plt.subplot(3, 3, 1)
        plt.plot(epochs, history['total_loss'], label='Total Loss', linewidth=2.5, color='red', alpha=0.8)
        plt.plot(epochs, history['main_loss'], label='Main Loss (BCE)', linewidth=2, color='blue', alpha=0.7, linestyle='--')
        plt.plot(epochs, history['shap_loss'], label='SHAP Loss', linewidth=2, color='green', alpha=0.7, linestyle=':')
        plt.xlabel('Epoch', fontsize=11, fontweight='bold')
        plt.ylabel('Loss', fontsize=11, fontweight='bold')
        plt.title(
            f'Динамика потерь\n({self.method_label_inline})',
            fontsize=13,
            fontweight='bold',
            pad=10
        )
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=9)

        # 2. ROC кривые сравнение (улучшенные)
        plt.subplot(3, 3, 2)

        if n_classes == 2:
            fpr_vanilla, tpr_vanilla, _ = roc_curve(y_test, y_prob_vanilla)
            fpr_shapreg, tpr_shapreg, _ = roc_curve(y_test, y_prob_shapreg)

            plt.plot(fpr_vanilla, tpr_vanilla, color='blue', lw=3, alpha=0.8,
                    label=f'Vanilla AUC = {vanilla_results["metrics"]["roc_auc"]:.4f}')
            plt.plot(
                fpr_shapreg,
                tpr_shapreg,
                color='darkgreen',
                lw=3,
                alpha=0.8,
                label=f'{self.method_label_inline} AUC = {shapreg_results["metrics"]["roc_auc"]:.4f}'
            )
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            plt.title('Сравнение ROC кривых', fontsize=13, fontweight='bold', pad=10)
            plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tick_params(labelsize=9)
        else:
            plt.text(0.5, 0.5, f'ROC кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
            plt.title('ROC кривые (многоклассовая)', fontsize=13, fontweight='bold', pad=10)

        # 3. Сравнение важности признаков (улучшенное)
        plt.subplot(3, 3, 3)
        top_n = min(12, len(feature_names))
        sorted_idx = np.argsort(feature_imp_vanilla)[::-1][:top_n]
        x_pos = np.arange(top_n)
        width = 0.35

        # Нормализация для сравнения
        vanilla_norm = feature_imp_vanilla / (np.sum(feature_imp_vanilla) + 1e-8)
        shapreg_norm = feature_imp_shapreg / (np.sum(feature_imp_shapreg) + 1e-8)

        plt.bar(x_pos - width/2, vanilla_norm[sorted_idx], width,
               label='Vanilla ANFIS', color='#3498db', alpha=0.8, edgecolor='navy', linewidth=1)
        plt.bar(
            x_pos + width/2,
            shapreg_norm[sorted_idx],
            width,
            label=self.method_label_inline,
            color='#2ecc71',
            alpha=0.8,
            edgecolor='darkgreen',
            linewidth=1
        )
        plt.title(f'Сравнение важности SHAP-признаков\n(топ-{top_n})', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Признаки', fontsize=11, fontweight='bold')
        plt.ylabel('Нормализованная важность', fontsize=11, fontweight='bold')
        plt.xticks(x_pos, [feature_names[i][:10] for i in sorted_idx], rotation=45, ha='right', fontsize=8)
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tick_params(labelsize=9)

        # 4. Сравнение метрик (улучшенное)
        plt.subplot(3, 3, 4)
        metrics_names = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
        vanilla_scores = [vanilla_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        shapreg_scores = [shapreg_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]

        x = np.arange(len(metrics_names))
        width_bar = 0.35
        bars1 = plt.bar(x - width_bar/2, vanilla_scores, width_bar, label='Vanilla ANFIS', 
                       color='#3498db', alpha=0.8, edgecolor='navy', linewidth=1.5)
        bars2 = plt.bar(
            x + width_bar/2,
            shapreg_scores,
            width_bar,
            label=self.method_label_inline,
            color='#2ecc71',
            alpha=0.8,
            edgecolor='darkgreen',
            linewidth=1.5
        )
        
        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)
        
        plt.xlabel('Метрики', fontsize=11, fontweight='bold')
        plt.ylabel('Значения', fontsize=11, fontweight='bold')
        plt.title('Сравнение метрик моделей', fontsize=13, fontweight='bold', pad=10)
        plt.xticks(x, metrics_names, fontsize=10)
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.ylim([0, 1.1])
        plt.tick_params(labelsize=9)

        # 5. Корреляция предсказаний (улучшенная)
        plt.subplot(3, 3, 5)
        correlation = np.corrcoef(y_prob_vanilla, y_prob_shapreg)[0, 1]
        plt.scatter(y_prob_vanilla, y_prob_shapreg, alpha=0.6, c='purple', s=30, edgecolors='black', linewidth=0.5)
        min_val = min(y_prob_vanilla.min(), y_prob_shapreg.min())
        max_val = max(y_prob_vanilla.max(), y_prob_shapreg.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Идеальная корреляция')
        plt.xlabel('Vanilla ANFIS предсказания', fontsize=11, fontweight='bold')
        plt.ylabel(
            f'Предсказания после\n{self.method_label_inline}',
            fontsize=11,
            fontweight='bold'
        )
        plt.title(f'Корреляция предсказаний\n(r = {correlation:.4f})', fontsize=13, fontweight='bold', pad=10)
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=9)

        # 6. Precision-Recall кривые
        plt.subplot(3, 3, 6)
        if n_classes == 2:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            prec_vanilla, rec_vanilla, _ = precision_recall_curve(y_test, y_prob_vanilla)
            prec_shapreg, rec_shapreg, _ = precision_recall_curve(y_test, y_prob_shapreg)
            ap_vanilla = average_precision_score(y_test, y_prob_vanilla)
            ap_shapreg = average_precision_score(y_test, y_prob_shapreg)
            
            plt.plot(rec_vanilla, prec_vanilla, color='blue', lw=2.5, alpha=0.8,
                    label=f'Vanilla AP = {ap_vanilla:.4f}')
            plt.plot(
                rec_shapreg,
                prec_shapreg,
                color='darkgreen',
                lw=2.5,
                alpha=0.8,
                label=f'{self.method_label_inline} AP = {ap_shapreg:.4f}'
            )
            plt.xlabel('Recall', fontsize=11, fontweight='bold')
            plt.ylabel('Precision', fontsize=11, fontweight='bold')
            plt.title('Precision-Recall кривые', fontsize=13, fontweight='bold', pad=10)
            plt.legend(loc='best', fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tick_params(labelsize=9)
        else:
            plt.text(0.5, 0.5, f'PR кривые не поддерживаются\nдля {n_classes} классов',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
            plt.title('Precision-Recall кривые', fontsize=13, fontweight='bold', pad=10)

        # 7. Анализ переобучения (если есть валидационные метрики)
        plt.subplot(3, 3, 7)
        if history.get('val_metrics') and any(m is not None for m in history['val_metrics']):
            # Для классификации используем accuracy, для регрессии - r2 или rmse
            if self.task_type == 'classification':
                train_acc = [m.get('accuracy', 0) for m in history['train_metrics'] if m]
                val_acc = [m.get('accuracy', 0) for m in history['val_metrics'] if m is not None]
            else:
                # Для регрессии используем R² или RMSE
                train_acc = [m.get('r2', -m.get('rmse', 0)) for m in history['train_metrics'] if m]
                val_acc = [m.get('r2', -m.get('rmse', 0)) for m in history['val_metrics'] if m is not None]
            if len(train_acc) == len(val_acc) and len(train_acc) > 1:
                epochs_plot = range(1, len(train_acc) + 1)
                plt.plot(epochs_plot, train_acc, 'b-', linewidth=2.5, marker='o', markersize=4,
                        label='Train Accuracy', alpha=0.8)
                plt.plot(epochs_plot, val_acc, 'r--', linewidth=2.5, marker='s', markersize=4,
                        label='Val Accuracy', alpha=0.8)
                gap = train_acc[-1] - val_acc[-1]
                if gap > 0.05:
                    plt.fill_between(epochs_plot, train_acc, val_acc, alpha=0.3, color='red', 
                                   label=f'Overfitting gap={gap:.3f}')
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
                plt.title('Train vs Val Accuracy\n(Анализ переобучения)', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)
                plt.ylim([0, 1.05])
            else:
                plt.text(0.5, 0.5, 'Недостаточно данных\nдля анализа', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
                plt.title('Анализ переобучения', fontsize=13, fontweight='bold', pad=10)
        else:
            plt.text(0.5, 0.5, 'Валидационные данные\nнедоступны', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
            plt.title('Анализ переобучения', fontsize=13, fontweight='bold', pad=10)

        # 8. Сводная таблица (улучшенная)
        plt.subplot(3, 3, 8)
        plt.axis('off')

        comparison_text = f"""СРАВНИТЕЛЬНЫЙ АНАЛИЗ
========================================

[VANILLA] ANFIS:
  Accuracy:  {vanilla_results['metrics']['accuracy']:.4f}
  Precision: {vanilla_results['metrics']['precision']:.4f}
  Recall:    {vanilla_results['metrics']['recall']:.4f}
  F1-Score:  {vanilla_results['metrics']['f1_score']:.4f}
  ROC AUC:   {vanilla_results['metrics']['roc_auc']:.4f}

[{self.method_name_en}] {self.method_label_inline}:
  Accuracy:  {shapreg_results['metrics']['accuracy']:.4f}
  Precision: {shapreg_results['metrics']['precision']:.4f}
  Recall:    {shapreg_results['metrics']['recall']:.4f}
  F1-Score:  {shapreg_results['metrics']['f1_score']:.4f}
  ROC AUC:   {shapreg_results['metrics']['roc_auc']:.4f}

ИЗМЕНЕНИЯ:
  Delta Accuracy:  {shapreg_results['metrics']['accuracy'] - vanilla_results['metrics']['accuracy']:+.4f}
  Delta Precision: {shapreg_results['metrics']['precision'] - vanilla_results['metrics']['precision']:+.4f}
  Delta Recall:    {shapreg_results['metrics']['recall'] - vanilla_results['metrics']['recall']:+.4f}
  Delta F1:        {shapreg_results['metrics']['f1_score'] - vanilla_results['metrics']['f1_score']:+.4f}
  Delta AUC:       {shapreg_results['metrics']['roc_auc'] - vanilla_results['metrics']['roc_auc']:+.4f}

Корреляция: {correlation:.4f}
"""

        plt.text(0.05, 0.95, comparison_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='teal', linewidth=2))

        # 9. Время обучения
        plt.subplot(3, 3, 9)
        plt.axis('off')
        time_text = f"""ВРЕМЯ ОБУЧЕНИЯ
========================================

Vanilla ANFIS:
  {vanilla_results.get('training_time', 0):.2f} сек

{self.method_label_inline}:
  {shapreg_results.get('training_time', 0):.2f} сек

Ускорение: {vanilla_results.get('training_time', 1) / shapreg_results.get('training_time', 1):.2f}x

Эпох обучения:
  {len(history['total_loss'])} эпох
"""
        plt.text(0.05, 0.95, time_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=2))

        plt.tight_layout(pad=3.0)
        plt.suptitle(
            f'Сравнительный анализ: Vanilla ANFIS vs {self.method_model_label_ru}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )

        # Сохранение
        if self.save_dir:
            plt.savefig(self.save_dir / 'shap_regularization_analysis.png',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

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

    def create_training_plots(self, history, save_dir=None):
        """Создание улучшенных графиков обучения (loss, метрики, время) с анализом переобучения"""
        if not history or 'total_loss' not in history:
            print("[WARNING] История обучения недоступна для визуализации")
            return

        # Используем более красивую тему
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(22, 14))

        epochs = range(1, len(history['total_loss']) + 1)
        has_val = history.get('val_metrics') and any(m is not None for m in history['val_metrics'])

        # 1. График потерь (улучшенный)
        plt.subplot(3, 3, 1)
        plt.plot(epochs, history['total_loss'], 'r-', label='Total Loss', linewidth=2.5, alpha=0.8)
        plt.plot(epochs, history['main_loss'], 'b--', label='Main Loss', linewidth=2, alpha=0.7)
        plt.plot(epochs, history['shap_loss'], 'g:', label='SHAP Loss', linewidth=2, alpha=0.7)
        plt.xlabel('Epoch', fontsize=11, fontweight='bold')
        plt.ylabel('Loss', fontsize=11, fontweight='bold')
        plt.title('Динамика потерь во время обучения', fontsize=13, fontweight='bold', pad=10)
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=9)

        # 2. График времени обучения (улучшенный)
        plt.subplot(3, 3, 2)
        if 'epoch_times' in history and history['epoch_times']:
            # Исправление: используем правильное количество эпох для времени
            epoch_times = history['epoch_times']
            epochs_for_time = range(1, len(epoch_times) + 1)
            plt.plot(epochs_for_time, epoch_times, 'purple', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
            avg_time = np.mean(epoch_times)
            plt.axhline(y=avg_time, color='r', linestyle='--', linewidth=2, 
                       label=f'Среднее: {avg_time:.2f}с', alpha=0.7)
            plt.xlabel('Epoch', fontsize=11, fontweight='bold')
            plt.ylabel('Время (сек)', fontsize=11, fontweight='bold')
            plt.title('Время обучения на эпоху', fontsize=13, fontweight='bold', pad=10)
            plt.legend(loc='best', fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tick_params(labelsize=9)

        # 3-5. Графики метрик обучения с сравнением train/val
        if history.get('train_metrics'):
            train_metrics = history['train_metrics']
            val_metrics = history.get('val_metrics', [])
            
            if self.task_type == 'classification':
                # Accuracy с train/val сравнением
                plt.subplot(3, 3, 3)
                train_acc = [m.get('accuracy', 0) for m in train_metrics if m]
                val_acc = [m.get('accuracy', 0) for m in val_metrics if m is not None] if has_val else []
                if train_acc:
                    plt.plot(epochs[:len(train_acc)], train_acc, 'b-', linewidth=2.5, marker='o', 
                            markersize=4, label='Train', alpha=0.8)
                    if val_acc:
                        plt.plot(epochs[:len(val_acc)], val_acc, 'r--', linewidth=2.5, marker='s', 
                                markersize=4, label='Validation', alpha=0.8)
                        # Анализ переобучения
                        if len(train_acc) == len(val_acc) and len(train_acc) > 1:
                            gap = train_acc[-1] - val_acc[-1]
                            if gap > 0.05:
                                plt.text(0.5, 0.05, f'[WARNING] Переобучение: gap={gap:.3f}', 
                                        transform=plt.gca().transAxes, fontsize=9, 
                                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
                plt.title('Accuracy: Train vs Validation', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)
                plt.ylim([0, 1.05])

                # Precision и Recall
                plt.subplot(3, 3, 4)
                train_prec = [m['precision'] for m in train_metrics if m]
                train_rec = [m['recall'] for m in train_metrics if m]
                val_prec = [m['precision'] for m in val_metrics if m is not None] if has_val else []
                val_rec = [m['recall'] for m in val_metrics if m is not None] if has_val else []
                if train_prec and train_rec:
                    plt.plot(epochs[:len(train_prec)], train_prec, 'g-', linewidth=2, marker='o', 
                            markersize=3, label='Precision (Train)', alpha=0.7)
                    plt.plot(epochs[:len(train_rec)], train_rec, 'b-', linewidth=2, marker='s', 
                            markersize=3, label='Recall (Train)', alpha=0.7)
                    if val_prec and val_rec:
                        plt.plot(epochs[:len(val_prec)], val_prec, 'g--', linewidth=2, marker='^', 
                                markersize=3, label='Precision (Val)', alpha=0.7)
                        plt.plot(epochs[:len(val_rec)], val_rec, 'b--', linewidth=2, marker='v', 
                                markersize=3, label='Recall (Val)', alpha=0.7)
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('Score', fontsize=11, fontweight='bold')
                plt.title('Precision & Recall', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)
                plt.ylim([0, 1.05])

                # F1-Score и ROC-AUC
                plt.subplot(3, 3, 5)
                train_f1 = [m['f1_score'] for m in train_metrics if m]
                train_roc = [m['roc_auc'] for m in train_metrics if m]
                val_f1 = [m['f1_score'] for m in val_metrics if m is not None] if has_val else []
                val_roc = [m['roc_auc'] for m in val_metrics if m is not None] if has_val else []
                if train_f1 and train_roc:
                    plt.plot(epochs[:len(train_f1)], train_f1, 'purple', linewidth=2, marker='o', 
                            markersize=3, label='F1 (Train)', alpha=0.7)
                    plt.plot(epochs[:len(train_roc)], train_roc, 'orange', linewidth=2, marker='s', 
                            markersize=3, label='ROC-AUC (Train)', alpha=0.7)
                    if val_f1 and val_roc:
                        plt.plot(epochs[:len(val_f1)], val_f1, 'purple', linewidth=2, marker='^', 
                                markersize=3, label='F1 (Val)', linestyle='--', alpha=0.7)
                        plt.plot(epochs[:len(val_roc)], val_roc, 'orange', linewidth=2, marker='v', 
                                markersize=3, label='ROC-AUC (Val)', linestyle='--', alpha=0.7)
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('Score', fontsize=11, fontweight='bold')
                plt.title('F1-Score & ROC-AUC', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)
                plt.ylim([0, 1.05])

            else:  # regression
                # RMSE с train/val
                plt.subplot(3, 3, 3)
                train_rmse = [m['rmse'] for m in train_metrics if m]
                val_rmse = [m['rmse'] for m in val_metrics if m is not None] if has_val else []
                if train_rmse:
                    plt.plot(epochs[:len(train_rmse)], train_rmse, 'r-', linewidth=2.5, marker='o', 
                            markersize=4, label='Train', alpha=0.8)
                    if val_rmse:
                        plt.plot(epochs[:len(val_rmse)], val_rmse, 'r--', linewidth=2.5, marker='s', 
                                markersize=4, label='Validation', alpha=0.8)
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('RMSE', fontsize=11, fontweight='bold')
                plt.title('RMSE: Train vs Validation', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)

                # R² с train/val
                plt.subplot(3, 3, 4)
                train_r2 = [m['r2'] for m in train_metrics if m]
                val_r2 = [m['r2'] for m in val_metrics if m is not None] if has_val else []
                if train_r2:
                    plt.plot(epochs[:len(train_r2)], train_r2, 'b-', linewidth=2.5, marker='s', 
                            markersize=4, label='Train', alpha=0.8)
                    if val_r2:
                        plt.plot(epochs[:len(val_r2)], val_r2, 'b--', linewidth=2.5, marker='^', 
                                markersize=4, label='Validation', alpha=0.8)
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('R²', fontsize=11, fontweight='bold')
                plt.title('R²: Train vs Validation', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)

                # MAE с train/val
                plt.subplot(3, 3, 5)
                train_mae = [m['mae'] for m in train_metrics if m]
                val_mae = [m['mae'] for m in val_metrics if m is not None] if has_val else []
                if train_mae:
                    plt.plot(epochs[:len(train_mae)], train_mae, 'g-', linewidth=2.5, marker='^', 
                            markersize=4, label='Train', alpha=0.8)
                    if val_mae:
                        plt.plot(epochs[:len(val_mae)], val_mae, 'g--', linewidth=2.5, marker='v', 
                                markersize=4, label='Validation', alpha=0.8)
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('MAE', fontsize=11, fontweight='bold')
                plt.title('MAE: Train vs Validation', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tick_params(labelsize=9)

        # 6. Анализ переобучения (новый график)
        plt.subplot(3, 3, 6)
        if has_val and train_metrics and val_metrics:
            # Для классификации используем accuracy, для регрессии - r2 или rmse
            if self.task_type == 'classification':
                train_acc = [m.get('accuracy', 0) for m in train_metrics if m]
                val_acc = [m.get('accuracy', 0) for m in val_metrics if m is not None]
            else:
                # Для регрессии используем R² или RMSE
                train_acc = [m.get('r2', -m.get('rmse', 0)) for m in train_metrics if m]
                val_acc = [m.get('r2', -m.get('rmse', 0)) for m in val_metrics if m is not None]
            if len(train_acc) == len(val_acc) and len(train_acc) > 1:
                overfitting_gap = [t - v for t, v in zip(train_acc, val_acc)]
                colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gap]
                plt.bar(epochs[:len(overfitting_gap)], overfitting_gap, color=colors, alpha=0.6, edgecolor='black')
                plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
                plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Критический порог')
                plt.xlabel('Epoch', fontsize=11, fontweight='bold')
                plt.ylabel('Gap (Train - Val)', fontsize=11, fontweight='bold')
                plt.title('Анализ переобучения', fontsize=13, fontweight='bold', pad=10)
                plt.legend(loc='best', fontsize=9, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--', axis='y')
                plt.tick_params(labelsize=9)
            else:
                plt.text(0.5, 0.5, 'Недостаточно данных\nдля анализа', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title('Анализ переобучения', fontsize=13, fontweight='bold', pad=10)
        else:
            plt.text(0.5, 0.5, 'Валидационные данные\nнедоступны', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Анализ переобучения', fontsize=13, fontweight='bold', pad=10)

        # 7. Сводная статистика (улучшенная)
        plt.subplot(3, 3, 7)
        plt.axis('off')
        
        stats_text = "[INFO] СТАТИСТИКА ОБУЧЕНИЯ\n\n"
        stats_text += f"Всего эпох: {len(history['total_loss'])}\n"
        if 'epoch_times' in history and history['epoch_times']:
            total_time = sum(history['epoch_times'])
            avg_time = np.mean(history['epoch_times'])
            stats_text += f"Общее время: {total_time:.2f} сек\n"
            stats_text += f"Среднее/эпоха: {avg_time:.2f} сек\n"
        
        stats_text += f"\n[INFO] Финальные потери:\n"
        stats_text += f"Total: {history['total_loss'][-1]:.4f}\n"
        stats_text += f"Main: {history['main_loss'][-1]:.4f}\n"
        stats_text += f"SHAP: {history['shap_loss'][-1]:.4f}\n"
        
        if history.get('train_metrics') and history['train_metrics'][-1]:
            final_train = history['train_metrics'][-1]
            final_val = history['val_metrics'][-1] if has_val and history['val_metrics'][-1] else None
            stats_text += f"\n[INFO] Финальные метрики:\n"
            if self.task_type == 'classification':
                stats_text += f"Train Accuracy: {final_train.get('accuracy', 0):.4f}\n"
                if final_val:
                    stats_text += f"Val Accuracy: {final_val.get('accuracy', 0):.4f}\n"
                    gap = final_train.get('accuracy', 0) - final_val.get('accuracy', 0)
                    stats_text += f"Gap: {gap:.4f}\n"
                    if gap > 0.05:
                        stats_text += f"[WARNING] Переобучение!\n"
                stats_text += f"Train F1: {final_train.get('f1_score', 0):.4f}\n"
                stats_text += f"Train ROC-AUC: {final_train.get('roc_auc', 0):.4f}\n"
            else:
                stats_text += f"Train RMSE: {final_train.get('rmse', 0):.4f}\n"
                stats_text += f"Train MAE: {final_train.get('mae', 0):.4f}\n"
                stats_text += f"Train R²: {final_train.get('r2', 0):.4f}\n"
                if final_val:
                    stats_text += f"Val RMSE: {final_val.get('rmse', 0):.4f}\n"
                    stats_text += f"Val MAE: {final_val.get('mae', 0):.4f}\n"
                    stats_text += f"Val R²: {final_val.get('r2', 0):.4f}\n"
                    gap = final_train.get('r2', 0) - final_val.get('r2', 0)
                    stats_text += f"Gap: {gap:.4f}\n"
                    if gap > 0.05:
                        stats_text += f"[WARNING] Переобучение!\n"

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='navy', linewidth=2))

        # 8. График потерь (детальный)
        plt.subplot(3, 3, 8)
        plt.plot(epochs, history['main_loss'], 'b-', linewidth=2, label='Main Loss', alpha=0.7)
        plt.plot(epochs, [h * self.config.get('shap', {}).get('gamma', 0.5) for h in history['shap_loss']], 
                'g-', linewidth=2, label=f'SHAP Loss × γ', alpha=0.7)
        plt.xlabel('Epoch', fontsize=11, fontweight='bold')
        plt.ylabel('Loss', fontsize=11, fontweight='bold')
        plt.title('Детализация потерь', fontsize=13, fontweight='bold', pad=10)
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=9)

        # 9. Информация о переобучении
        plt.subplot(3, 3, 9)
        plt.axis('off')
        overfitting_info = "[INFO] АНАЛИЗ ПЕРЕОБУЧЕНИЯ\n\n"
        if has_val and train_metrics and val_metrics:
            # Для классификации используем accuracy, для регрессии - r2 или rmse
            if self.task_type == 'classification':
                train_acc = [m.get('accuracy', 0) for m in train_metrics if m]
                val_acc = [m.get('accuracy', 0) for m in val_metrics if m is not None]
            else:
                # Для регрессии используем R² или RMSE
                train_acc = [m.get('r2', -m.get('rmse', 0)) for m in train_metrics if m]
                val_acc = [m.get('r2', -m.get('rmse', 0)) for m in val_metrics if m is not None]
            if len(train_acc) == len(val_acc) and len(train_acc) > 0:
                final_gap = train_acc[-1] - val_acc[-1]
                max_gap = max([t - v for t, v in zip(train_acc, val_acc)])
                overfitting_info += f"Финальный gap: {final_gap:.4f}\n"
                overfitting_info += f"Максимальный gap: {max_gap:.4f}\n\n"
                if final_gap > 0.1:
                    overfitting_info += "[WARNING] СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ\n"
                    overfitting_info += "Рекомендации:\n"
                    overfitting_info += "• Уменьшить γ\n"
                    overfitting_info += "• Увеличить регуляризацию\n"
                    overfitting_info += "• Ранняя остановка\n"
                elif final_gap > 0.05:
                    overfitting_info += "[WARNING] УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ\n"
                    overfitting_info += "Рекомендации:\n"
                    overfitting_info += "• Использовать раннюю остановку\n"
                    overfitting_info += "• Увеличить dropout\n"
                else:
                    overfitting_info += "[OK] Переобучения нет\n"
                    overfitting_info += "Модель хорошо обобщается\n"
        else:
            overfitting_info += "[WARNING] Валидационные данные\nнедоступны\n"
            overfitting_info += "Невозможно оценить\nпереобучение"

        plt.text(0.05, 0.95, overfitting_info, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))

        plt.tight_layout(pad=3.0)
        plt.suptitle(
            f'Графики обучения {self.method_model_label_ru}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )

        # Сохранение
        if save_dir:
            save_path = Path(save_dir) / 'training_plots.png'
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            print(f"[INFO] Графики обучения сохранены: {save_path}")
        plt.close()

    def create_rules_comparison(self, rules_comparison, save_dir=None):
        """Визуализация сравнения правил ANFIS"""
        fig = plt.figure(figsize=(18, 10))

        # 1. Сходство правил
        plt.subplot(2, 3, 1)
        similarities = rules_comparison['rule_similarities']
        plt.bar(range(len(similarities)), similarities, color='skyblue', alpha=0.7)
        plt.xlabel('Номер правила', fontsize=12)
        plt.ylabel('Сходство', fontsize=12)
        plt.title('Сходство правил между моделями', fontsize=14, fontweight='bold')
        plt.axhline(y=rules_comparison['average_similarity'], color='r', 
                   linestyle='--', label=f'Среднее: {rules_comparison["average_similarity"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 2. Различия коэффициентов
        plt.subplot(2, 3, 2)
        coeff_diffs = rules_comparison['coefficient_differences']
        plt.bar(range(len(coeff_diffs)), coeff_diffs, color='lightcoral', alpha=0.7)
        plt.xlabel('Номер правила', fontsize=12)
        plt.ylabel('Различие коэффициентов', fontsize=12)
        plt.title('Различия коэффициентов правил', fontsize=14, fontweight='bold')
        plt.axhline(y=rules_comparison['average_coeff_diff'], color='r', 
                   linestyle='--', label=f'Среднее: {rules_comparison["average_coeff_diff"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 3. Различия функций принадлежности
        plt.subplot(2, 3, 3)
        mf_diffs = rules_comparison['mf_differences']
        plt.bar(range(len(mf_diffs)), mf_diffs, color='lightgreen', alpha=0.7)
        plt.xlabel('Номер правила', fontsize=12)
        plt.ylabel('Различие функций принадлежности', fontsize=12)
        plt.title('Различия функций принадлежности', fontsize=14, fontweight='bold')
        plt.axhline(y=rules_comparison['average_mf_diff'], color='r', 
                   linestyle='--', label=f'Среднее: {rules_comparison["average_mf_diff"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 4. Сводная статистика
        plt.subplot(2, 3, 4)
        plt.axis('off')
        
        stats_text = "СРАВНЕНИЕ ПРАВИЛ\n\n"
        stats_text += f"Количество правил:\n"
        for model_name, num_rules in rules_comparison['num_rules'].items():
            stats_text += f"  {model_name}: {num_rules}\n"
        
        stats_text += f"\nСреднее сходство: {rules_comparison['average_similarity']:.4f}\n"
        stats_text += f"Среднее различие коэффициентов: {rules_comparison['average_coeff_diff']:.4f}\n"
        stats_text += f"Среднее различие ФП: {rules_comparison['average_mf_diff']:.4f}\n"

        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 5. Распределение сходства
        plt.subplot(2, 3, 5)
        plt.hist(similarities, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Сходство', fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.title('Распределение сходства правил', fontsize=14, fontweight='bold')
        plt.axvline(x=rules_comparison['average_similarity'], color='r', 
                   linestyle='--', label=f'Среднее: {rules_comparison["average_similarity"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 6. Комбинированный график различий
        plt.subplot(2, 3, 6)
        x = range(len(coeff_diffs))
        width = 0.35
        plt.bar([i - width/2 for i in x], coeff_diffs, width, 
               label='Коэффициенты', color='lightcoral', alpha=0.7)
        plt.bar([i + width/2 for i in x], mf_diffs, width, 
               label='Функции принадлежности', color='lightgreen', alpha=0.7)
        plt.xlabel('Номер правила', fontsize=12)
        plt.ylabel('Различие', fontsize=12)
        plt.title('Сравнение различий', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(pad=3.0)
        plt.suptitle(
            f'Сравнение правил: Vanilla ANFIS vs {self.method_model_label_ru}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Сохранение
        if save_dir:
            save_path = Path(save_dir) / 'rules_comparison.png'
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            print(f"[INFO] График сравнения правил сохранен: {save_path}")
        plt.close()

    def create_shap_comparison(self, posthoc_shap, embedded_shap, feature_names, save_dir=None):
        """
        Сравнение post-hoc SHAP и встроенного метода Baseline-Masked Feature Sensitivity Regularization.
        
        Метод сопоставляет два подхода к оценке важности признаков. Post-hoc SHAP применяется 
        к уже обученной vanilla модели после завершения обучения, тогда как Embedded SHAP 
        представляет собой встроенную важность признаков, вычисляемую в процессе обучения 
        модели с Baseline-Masked Feature Sensitivity Regularization и интегрированную в функцию потерь. 
        График изменения рангов показывает, как изменилась относительная важность признаков при переходе 
        от post-hoc анализа к встроенной оценке чувствительности признаков, что позволяет оценить 
        согласованность локальных объяснений между двумя методами и выявить признаки, получившие приоритет 
        в процессе обучения с регуляризацией чувствительности признаков.
        """
        fig = plt.figure(figsize=(20, 12))

        # Нормализация для сравнения
        posthoc_norm = posthoc_shap / (np.sum(np.abs(posthoc_shap)) + 1e-8)
        embedded_norm = embedded_shap / (np.sum(np.abs(embedded_shap)) + 1e-8)

        top_n = min(15, len(feature_names))
        sorted_idx = np.argsort(np.abs(posthoc_shap))[::-1][:top_n]

        # 1. Сравнение важности признаков (bar plot)
        plt.subplot(2, 3, 1)
        x_pos = np.arange(top_n)
        width = 0.35
        plt.bar(x_pos - width/2, posthoc_norm[sorted_idx], width, 
               label='Post-hoc SHAP', color='lightblue', alpha=0.8)
        plt.bar(x_pos + width/2, embedded_norm[sorted_idx], width, 
               label='Embedded SHAP', color='lightgreen', alpha=0.8)
        plt.xlabel('Признаки', fontsize=12)
        plt.ylabel('Нормализованная важность', fontsize=12)
        plt.title(f'Сравнение важности SHAP-признаков (топ-{top_n})', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, [feature_names[i][:10] for i in sorted_idx], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 2. Scatter plot сравнения
        plt.subplot(2, 3, 2)
        plt.scatter(posthoc_norm, embedded_norm, alpha=0.6, s=50)
        min_val = min(posthoc_norm.min(), embedded_norm.min())
        max_val = max(posthoc_norm.max(), embedded_norm.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        correlation = np.corrcoef(posthoc_norm, embedded_norm)[0, 1]
        plt.xlabel('Post-hoc SHAP (нормализовано)', fontsize=12)
        plt.ylabel('Embedded SHAP (нормализовано)', fontsize=12)
        plt.title(f'Корреляция важностей (r={correlation:.3f})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 3. Различия важности
        plt.subplot(2, 3, 3)
        differences = np.abs(posthoc_norm - embedded_norm)
        sorted_diff_idx = np.argsort(differences)[::-1][:top_n]
        plt.barh(range(top_n), differences[sorted_diff_idx], color='orange', alpha=0.7)
        plt.yticks(range(top_n), [feature_names[i][:15] for i in sorted_diff_idx])
        plt.xlabel('Абсолютное различие', fontsize=12)
        plt.title('Топ различий в важности', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 4. Распределение важностей
        plt.subplot(2, 3, 4)
        plt.hist(posthoc_norm, bins=20, alpha=0.6, label='Post-hoc SHAP', color='lightblue', density=True)
        plt.hist(embedded_norm, bins=20, alpha=0.6, label='Embedded SHAP', color='lightgreen', density=True)
        plt.xlabel('Нормализованная важность', fontsize=12)
        plt.ylabel('Плотность', fontsize=12)
        plt.title('Распределение важностей', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Ранжирование признаков
        # Этот график сравнивает ранги признаков между двумя методами оценки важности. 
        # Post-hoc SHAP применяется к уже обученной vanilla модели после завершения обучения, 
        # а Embedded SHAP представляет встроенную важность из модели с Baseline-Masked Feature Sensitivity Regularization,
        # вычисляемую в процессе обучения. Ранг признака определяется его позицией при сортировке 
        # по убыванию важности, где ранг 0 соответствует самому важному признаку.
        plt.subplot(2, 3, 5)
        
        # Вычисляем ранги признаков для каждого метода. Ранги представляют собой индексы признаков, 
        # отсортированные по убыванию абсолютной важности. Первый элемент имеет ранг 0 и является 
        # самым важным, последний элемент имеет наивысший ранг и является наименее важным.
        posthoc_ranks = np.argsort(np.abs(posthoc_shap))[::-1]
        embedded_ranks = np.argsort(np.abs(embedded_shap))[::-1]
        
        # Для топ признаков, отобранных по важности в post-hoc методе, вычисляем разницу в рангах 
        # между двумя методами. Разница вычисляется как posthoc_rank - embedded_rank. Положительная 
        # разница означает, что признак имеет более высокий ранг в post-hoc методе по сравнению 
        # с embedded методом, то есть стал менее важным в процессе обучения с регуляризацией. 
        # Отрицательная разница означает, что признак более важен в embedded методе и получил 
        # приоритет в процессе обучения. Нулевая разница означает, что ранг признака не изменился.
        rank_diffs = []
        top_features = []
        for i in range(min(top_n, len(feature_names))):
            feat_idx = sorted_idx[i]
            # Находим позицию признака в массиве рангов для каждого метода
            posthoc_rank = np.where(posthoc_ranks == feat_idx)[0][0]
            embedded_rank = np.where(embedded_ranks == feat_idx)[0][0]
            # Вычисляем изменение ранга. Положительное значение означает, что признак стал менее 
            # важным в embedded методе, отрицательное значение означает повышение важности.
            rank_diffs.append(posthoc_rank - embedded_rank)
            top_features.append(feature_names[feat_idx][:10])
        
        # Цветовая кодировка помогает визуально различить направление изменения ранга. Красный цвет 
        # используется для признаков, которые стали менее важными в embedded методе, зеленый цвет 
        # для признаков, которые стали более важными, и серый цвет для признаков без изменения ранга.
        colors = ['red' if d > 0 else 'green' if d < 0 else 'gray' for d in rank_diffs]
        plt.barh(range(len(rank_diffs)), rank_diffs, color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Разница в ранге (Post-hoc - Embedded)', fontsize=12)
        plt.title('Изменение рангов признаков', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='x')

        # 6. Сводная статистика
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        stats_text = "СРАВНЕНИЕ SHAP МЕТОДОВ\n\n"
        stats_text += f"Корреляция: {correlation:.4f}\n"
        stats_text += f"Среднее абсолютное различие: {np.mean(differences):.4f}\n"
        stats_text += f"Максимальное различие: {np.max(differences):.4f}\n"
        stats_text += f"Медианное различие: {np.median(differences):.4f}\n\n"
        
        stats_text += "Топ-5 признаков (Post-hoc):\n"
        for i, idx in enumerate(sorted_idx[:5]):
            stats_text += f"  {i+1}. {feature_names[idx][:20]}: {posthoc_norm[idx]:.4f}\n"
        
        stats_text += "\nТоп-5 признаков (Embedded):\n"
        embedded_sorted = np.argsort(np.abs(embedded_shap))[::-1][:5]
        for i, idx in enumerate(embedded_sorted):
            stats_text += f"  {i+1}. {feature_names[idx][:20]}: {embedded_norm[idx]:.4f}\n"

        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

        plt.tight_layout(pad=3.0)
        plt.suptitle(
            f'Сравнение Post-hoc SHAP и встроенного метода {self.method_label_inline}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Сохранение
        if save_dir:
            save_path = Path(save_dir) / 'shap_methods_comparison.png'
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            print(f"[INFO] График сравнения SHAP методов сохранен: {save_path}")
        plt.close()
