"""
Модуль для создания отдельных графиков высокого качества
Каждый график сохраняется в отдельный PNG файл с DPI 300
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from utils.method_labels import (
    METHOD_ALIAS_RU,
    METHOD_LABEL_INLINE,
    METHOD_MODEL_LABEL_RU,
    METHOD_NAME_EN,
)


class IndividualPlotCreator:
    """Класс для создания отдельных графиков высокого качества"""
    
    def __init__(self, config, save_dir=None):
        self.config = config
        self.task_type = config['dataset']['task_type']
        self.save_dir = Path(save_dir) if save_dir else None
        self.dpi = config['visualization'].get('dpi', 300)
        self.method_name_en = METHOD_NAME_EN
        self.method_alias = METHOD_ALIAS_RU
        self.method_label_inline = METHOD_LABEL_INLINE
        self.method_model_label_ru = METHOD_MODEL_LABEL_RU
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("muted")
        
        # Настройка параметров для высокого качества
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 13,
            'axes.titlesize': 15,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 17,
            'figure.titleweight': 'bold',
            'axes.linewidth': 1.5,
            'grid.linewidth': 1.0,
            'grid.alpha': 0.4,
            'lines.linewidth': 2.5,
            'patch.linewidth': 1.5,
        })
    
    def save_plot(self, fig, filename, **kwargs):
        """Сохраняет график с высоким качеством"""
        if not self.save_dir:
            return
        
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', **kwargs)
        plt.close(fig)
        print(f"[INFO] График сохранен: {save_path}")
    
    def create_confusion_matrix(self, y_test, y_pred, filename='vanilla_confusion_matrix.png'):
        """Матрица ошибок"""
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   cbar_kws={'shrink': 0.8}, linewidths=2, linecolor='black',
                   annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
        ax.set_title('Матрица ошибок (Confusion Matrix)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Предсказанный класс', fontsize=13, fontweight='bold')
        ax.set_ylabel('Истинный класс', fontsize=13, fontweight='bold')
        self.save_plot(fig, filename)
    
    def create_roc_curve(self, y_test, y_prob, metrics, filename='vanilla_roc_curve.png'):
        """ROC кривая"""
        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, color='darkorange', lw=3, alpha=0.8,
                   label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7,
                   label='Случайная классификация')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
            ax.set_title('ROC кривая', fontsize=16, fontweight='bold', pad=15)
            ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'ROC кривые не поддерживаются\nдля {n_classes} классов',
                   ha='center', va='center', transform=ax.transAxes, fontsize=13)
            ax.set_title(f'ROC кривая ({n_classes} классов)', fontsize=16, fontweight='bold', pad=15)
        
        self.save_plot(fig, filename)
    
    def create_feature_importance(self, feature_importance, feature_names, 
                                 top_n=15, filename='vanilla_feature_importance.png'):
        """Важность признаков"""
        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(top_n, len(feature_importance))
        sorted_indices = np.argsort(feature_importance)[::-1][:top_n]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        bars = ax.bar(range(len(sorted_importance)), sorted_importance, color=colors,
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Топ-{top_n} важных признаков (ANFIS)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Признаки', fontsize=13, fontweight='bold')
        ax.set_ylabel('Важность', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels([name[:15] for name in sorted_names], rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        self.save_plot(fig, filename)
    
    def create_precision_recall_curve(self, y_test, y_prob, filename='vanilla_pr_curve.png'):
        """Precision-Recall кривая"""
        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            ax.plot(recall_curve, precision_curve, color='purple', lw=2.5, alpha=0.8,
                   label=f'PR (AP = {avg_precision:.4f})')
            ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
            ax.set_title('Precision-Recall кривая', fontsize=16, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'PR кривые не поддерживаются\nдля {n_classes} классов',
                   ha='center', va='center', transform=ax.transAxes, fontsize=13)
            ax.set_title(f'Precision-Recall ({n_classes} классов)', fontsize=16, fontweight='bold', pad=15)
        
        self.save_plot(fig, filename)
    
    def create_prediction_distribution(self, y_test, y_prob, filename='vanilla_prediction_distribution.png'):
        """Распределение предсказаний"""
        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()
        
        if n_classes == 2:
            ax.hist(y_prob[y_test_values == 0], bins=20, alpha=0.7, label='Класс 0',
                   color='#3498db', density=True, edgecolor='navy', linewidth=1.5)
            ax.hist(y_prob[y_test_values == 1], bins=20, alpha=0.7, label='Класс 1',
                   color='#e74c3c', density=True, edgecolor='darkred', linewidth=1.5)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
        else:
            ax.hist(y_prob, bins=20, alpha=0.7, color='lightgreen', density=True,
                   edgecolor='darkgreen', linewidth=1.5)
        
        ax.set_xlabel('Предсказанная вероятность', fontsize=13, fontweight='bold')
        ax.set_ylabel('Плотность', fontsize=13, fontweight='bold')
        ax.set_title('Распределение вероятностей по классам', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        self.save_plot(fig, filename)
    
    def create_calibration_curve(self, y_test, y_prob, filename='vanilla_calibration_curve.png'):
        """Калибровочная кривая"""
        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
            ax.plot([0, 1], [0, 1], 'k--', label='Идеально калиброванная', linewidth=2)
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', color='red',
                   label='ANFIS', markersize=8, linewidth=2)
            ax.set_xlabel('Средняя предсказанная вероятность', fontsize=13, fontweight='bold')
            ax.set_ylabel('Доля положительных', fontsize=13, fontweight='bold')
            ax.set_title('Калибровочная кривая', fontsize=16, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Калибровка не поддерживается\nдля {n_classes} классов',
                   ha='center', va='center', transform=ax.transAxes, fontsize=13)
            ax.set_title(f'Калибровка ({n_classes} классов)', fontsize=16, fontweight='bold', pad=15)
        
        self.save_plot(fig, filename)
    
    def create_error_distribution(self, y_test, y_pred, y_prob, filename='vanilla_error_distribution.png'):
        """Распределение ошибок"""
        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()
        
        if n_classes == 2:
            errors = y_prob - y_test_values
        else:
            errors = y_pred - y_test_values
        
        ax.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black', linewidth=1.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Ошибка предсказания', fontsize=13, fontweight='bold')
        ax.set_ylabel('Частота', fontsize=13, fontweight='bold')
        ax.set_title('Распределение ошибок предсказания', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, filename)
    
    def create_all_vanilla_plots(self, vanilla_results, feature_names, X_test, y_test):
        """Создает все отдельные графики для Vanilla модели"""
        y_pred = np.array(vanilla_results['predictions']).flatten()
        y_prob = np.array(vanilla_results['probabilities']).flatten()
        metrics = vanilla_results['metrics']
        feature_importance = vanilla_results['feature_importance']
        
        print("Создание отдельных графиков для Vanilla ANFIS...")
        
        self.create_confusion_matrix(y_test, y_pred, 'vanilla_confusion_matrix.png')
        self.create_roc_curve(y_test, y_prob, metrics, 'vanilla_roc_curve.png')
        self.create_feature_importance(feature_importance, feature_names, 
                                      filename='vanilla_feature_importance.png')
        self.create_prediction_distribution(y_test, y_prob, 'vanilla_prediction_distribution.png')
        self.create_precision_recall_curve(y_test, y_prob, 'vanilla_pr_curve.png')
        self.create_calibration_curve(y_test, y_prob, 'vanilla_calibration_curve.png')
        self.create_error_distribution(y_test, y_pred, y_prob, 'vanilla_error_distribution.png')
        
        print("[OK] Все отдельные графики Vanilla созданы")
    
    def create_training_loss_plot(self, history, filename='training_loss.png'):
        """График динамики потерь"""
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['total_loss']) + 1)
        
        ax.plot(epochs, history['total_loss'], label='Total Loss', linewidth=2.5, color='red', alpha=0.8)
        if 'main_loss' in history:
            ax.plot(epochs, history['main_loss'], label='Main Loss', linewidth=2, color='blue', alpha=0.7, linestyle='--')
        if 'shap_loss' in history:
            ax.plot(epochs, history['shap_loss'], label='SHAP Loss', linewidth=2, color='green', alpha=0.7, linestyle=':')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title(
            f'Динамика потерь ({self.method_label_inline})',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        self.save_plot(fig, filename)
    
    def create_metrics_comparison_bar(self, vanilla_metrics, regularized_metrics, 
                                      task_type='classification', filename='metrics_comparison.png'):
        """Сравнение метрик столбчатой диаграммой"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if task_type == 'classification':
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            vanilla_scores = [vanilla_metrics.get(k, 0) for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
            regularized_scores = [regularized_metrics.get(k, 0) for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        else:
            metrics_names = ['RMSE', 'MAE', 'R²']
            vanilla_scores = [vanilla_metrics.get(k, 0) for k in ['rmse', 'mae', 'r2']]
            regularized_scores = [regularized_metrics.get(k, 0) for k in ['rmse', 'mae', 'r2']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla ANFIS',
                      color='#3498db', alpha=0.8, edgecolor='navy', linewidth=1.5)
        bars2 = ax.bar(
            x + width/2,
            regularized_scores,
            width,
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
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Метрики', fontsize=13, fontweight='bold')
        ax.set_ylabel('Значения', fontsize=13, fontweight='bold')
        ax.set_title('Сравнение метрик моделей', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        self.save_plot(fig, filename)
    
    def create_shap_comparison_bar(self, posthoc_shap, embedded_shap, feature_names,
                                   top_n=15, filename='shap_comparison_bar.png'):
        """Сравнение SHAP важности столбчатой диаграммой"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Нормализация
        posthoc_norm = posthoc_shap / (np.sum(np.abs(posthoc_shap)) + 1e-8)
        embedded_norm = embedded_shap / (np.sum(np.abs(embedded_shap)) + 1e-8)
        
        top_n = min(top_n, len(feature_names))
        sorted_idx = np.argsort(np.abs(posthoc_shap))[::-1][:top_n]
        
        x_pos = np.arange(top_n)
        width = 0.35
        
        ax.bar(x_pos - width/2, posthoc_norm[sorted_idx], width,
              label='Post-hoc SHAP', color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
        ax.bar(x_pos + width/2, embedded_norm[sorted_idx], width,
              label='Embedded SHAP', color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Признаки', fontsize=13, fontweight='bold')
        ax.set_ylabel('Нормализованная важность', fontsize=13, fontweight='bold')
        ax.set_title(f'Сравнение важности SHAP-признаков (топ-{top_n})', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([feature_names[i][:12] for i in sorted_idx], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        self.save_plot(fig, filename)
    
    def create_shap_scatter(self, posthoc_shap, embedded_shap, filename='shap_scatter.png'):
        """Scatter plot сравнения SHAP значений"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Нормализация
        posthoc_norm = posthoc_shap / (np.sum(np.abs(posthoc_shap)) + 1e-8)
        embedded_norm = embedded_shap / (np.sum(np.abs(embedded_shap)) + 1e-8)
        
        ax.scatter(posthoc_norm, embedded_norm, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        min_val = min(posthoc_norm.min(), embedded_norm.min())
        max_val = max(posthoc_norm.max(), embedded_norm.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Идеальная корреляция')
        
        correlation = np.corrcoef(posthoc_norm, embedded_norm)[0, 1]
        ax.set_xlabel('Post-hoc SHAP (нормализовано)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Embedded SHAP (нормализовано)', fontsize=13, fontweight='bold')
        ax.set_title(f'Корреляция важностей (r={correlation:.3f})', fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, filename)

