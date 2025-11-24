"""
Модуль для создания отдельных качественных графиков для презентации
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.patches as mpatches

from utils.method_labels import (
    METHOD_ALIAS_RU,
    METHOD_LABEL_COMPACT,
    METHOD_LABEL_INLINE,
    METHOD_LABEL_SHORT,
    METHOD_LABEL_SHORT_INLINE,
    METHOD_MODEL_LABEL,
    METHOD_MODEL_LABEL_RU,
    METHOD_MODEL_LABEL_SINGLE,
    METHOD_MODEL_LABEL_SHORT,
    METHOD_NAME_EN,
)

class PresentationPlotter:
    """Класс для создания отдельных качественных графиков для презентации"""
    
    def __init__(self, config, save_dir=None):
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else None
        self.task_type = config['dataset']['task_type']
        self.method_name_en = METHOD_NAME_EN
        self.method_alias = METHOD_ALIAS_RU
        self.method_label_compact = METHOD_LABEL_COMPACT
        self.method_label_inline = METHOD_LABEL_INLINE
        self.method_label_short = METHOD_LABEL_SHORT
        self.method_label_short_inline = METHOD_LABEL_SHORT_INLINE
        self.method_model_label = METHOD_MODEL_LABEL
        self.method_model_label_single = METHOD_MODEL_LABEL_SINGLE
        self.method_model_label_ru = METHOD_MODEL_LABEL_RU
        self.method_model_label_short = METHOD_MODEL_LABEL_SHORT
        self.short_vs_label = f"Vanilla vs {self.method_model_label_short}"
        
        # Настройка строгого стиля для военной презентации
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("muted")
        
        # Разнообразная научная цветовая палитра
        self.colors = {
            'vanilla': '#1a237e',      # Темно-синий
            'regularized': '#2e7d32',  # Темно-зеленый
            'posthoc': '#424242',      # Темно-серый
            'train': '#1565c0',        # Синий
            'val': '#616161',         # Серый
            'background': '#FFFFFF',   # Белый фон
            'grid': '#B0BEC5',        # Светло-серый для сетки
            'text': '#212121',        # Темно-серый для текста
            # Синие оттенки для Vanilla
            'blue1': '#0d47a1',       # Темно-синий
            'blue2': '#1565c0',       # Синий
            'blue3': '#1976d2',       # Светло-синий
            'blue4': '#42a5f5',       # Голубой
            # Зеленые оттенки для Regularized
            'green1': '#1b5e20',      # Темно-зеленый
            'green2': '#2e7d32',      # Зеленый
            'green3': '#388e3c',      # Светло-зеленый
            'green4': '#66bb6a',      # Светло-зеленый
            # Дополнительные цвета для разнообразия
            'orange1': '#e65100',     # Темно-оранжевый
            'orange2': '#ff6f00',     # Оранжевый
            'purple1': '#4a148c',     # Темно-фиолетовый
            'purple2': '#6a1b9a',     # Фиолетовый
            'teal1': '#004d40',       # Темно-бирюзовый
            'teal2': '#00695c',       # Бирюзовый
            'red1': '#b71c1c',        # Темно-красный
            'red2': '#c62828',        # Красный
            'gray1': '#757575',       # Серый
            'gray2': '#9e9e9e',       # Светло-серый
            'dark_blue': '#0d47a1',
            'dark_green': '#1b5e20'
        }
        
        # Настройка параметров matplotlib для профессионального вида
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'axes.labelsize': 13,
            'axes.titlesize': 15,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'legend.framealpha': 0.95,
            'legend.shadow': True,
            'figure.titlesize': 17,
            'figure.titleweight': 'bold',
            'axes.linewidth': 1.5,
            'grid.linewidth': 1.0,
            'grid.alpha': 0.4,
            'lines.linewidth': 2.5,
            'patch.linewidth': 1.5,
            'axes.edgecolor': '#424242',
            'axes.labelcolor': '#212121',
            'text.color': '#212121'
        })
        
    def create_training_metrics_comparison(self, vanilla_history, regularized_history, save_name='training_metrics_comparison.png'):
        """Создание графика сравнения метрик обучения Vanilla и модели с Baseline-Masked Feature Sensitivity Regularization"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f'Обучение: {self.short_vs_label}',
            fontsize=20,
            fontweight='bold',
            y=0.995,
            color='#212121'
        )
        
        epochs_vanilla = range(1, len(vanilla_history.get('train_metrics', [])) + 1) if vanilla_history.get('train_metrics') else []
        epochs_reg = range(1, len(regularized_history.get('train_metrics', [])) + 1) if regularized_history.get('train_metrics') else []
        
        # Если у vanilla нет истории, показываем только regularized
        has_vanilla_history = bool(vanilla_history.get('train_metrics'))
        
        if self.task_type == 'classification':
            # Accuracy
            ax = axes[0, 0]
            if has_vanilla_history and vanilla_history.get('train_metrics'):
                # Для классификации используем accuracy, для регрессии - r2
                task_type = self.config.get('dataset', {}).get('task_type', 'classification')
                if task_type == 'classification':
                    train_acc_v = [m.get('accuracy', 0) for m in vanilla_history['train_metrics'] if m]
                    val_acc_v = [m.get('accuracy', 0) for m in vanilla_history.get('val_metrics', []) if m is not None]
                else:
                    train_acc_v = [m.get('r2', -m.get('rmse', 0)) for m in vanilla_history['train_metrics'] if m]
                    val_acc_v = [m.get('r2', -m.get('rmse', 0)) for m in vanilla_history.get('val_metrics', []) if m is not None]
                if train_acc_v:
                    ax.plot(epochs_vanilla[:len(train_acc_v)], train_acc_v, 'o-', color=self.colors['vanilla'], 
                           linewidth=2.5, markersize=6, label='Vanilla Train', alpha=0.9)
                if val_acc_v:
                    ax.plot(epochs_vanilla[:len(val_acc_v)], val_acc_v, 's--', color=self.colors['vanilla'], 
                           linewidth=2, markersize=5, label='Vanilla Val', alpha=0.7)
            
            if regularized_history.get('train_metrics'):
                # Для классификации используем accuracy, для регрессии - r2
                task_type = self.config.get('dataset', {}).get('task_type', 'classification')
                if task_type == 'classification':
                    train_acc_r = [m.get('accuracy', 0) for m in regularized_history['train_metrics'] if m]
                    val_acc_r = [m.get('accuracy', 0) for m in regularized_history.get('val_metrics', []) if m is not None]
                else:
                    train_acc_r = [m.get('r2', -m.get('rmse', 0)) for m in regularized_history['train_metrics'] if m]
                    val_acc_r = [m.get('r2', -m.get('rmse', 0)) for m in regularized_history.get('val_metrics', []) if m is not None]
                if train_acc_r:
                    ax.plot(epochs_reg[:len(train_acc_r)], train_acc_r, 'o-', color=self.colors['regularized'], 
                           linewidth=2.5, markersize=6, label=f'{self.method_label_short} Train', alpha=0.9)
                if val_acc_r:
                    ax.plot(epochs_reg[:len(val_acc_r)], val_acc_r, 's--', color=self.colors['regularized'], 
                           linewidth=2, markersize=5, label=f'{self.method_label_short} Val', alpha=0.7)
            
            ax.set_xlabel('Эпоха', fontsize=14, fontweight='bold', color='#212121')
            ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='#212121')
            ax.set_title('Сравнение Accuracy во время обучения', fontsize=16, fontweight='bold', pad=18, color='#212121')
            ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#B0BEC5')
            ax.set_ylim([0, 1.05])
            ax.tick_params(labelsize=12, colors='#212121')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # F1-Score
            ax = axes[0, 1]
            if has_vanilla_history and vanilla_history.get('train_metrics'):
                train_f1_v = [m['f1_score'] for m in vanilla_history['train_metrics'] if m]
                val_f1_v = [m['f1_score'] for m in vanilla_history.get('val_metrics', []) if m is not None]
                if train_f1_v:
                    ax.plot(epochs_vanilla[:len(train_f1_v)], train_f1_v, 'o-', color=self.colors['vanilla'], 
                           linewidth=2.5, markersize=6, label='Vanilla Train', alpha=0.9)
                if val_f1_v:
                    ax.plot(epochs_vanilla[:len(val_f1_v)], val_f1_v, 's--', color=self.colors['vanilla'], 
                           linewidth=2, markersize=5, label='Vanilla Val', alpha=0.7)
            
            if regularized_history.get('train_metrics'):
                train_f1_r = [m['f1_score'] for m in regularized_history['train_metrics'] if m]
                val_f1_r = [m['f1_score'] for m in regularized_history.get('val_metrics', []) if m is not None]
                if train_f1_r:
                    ax.plot(epochs_reg[:len(train_f1_r)], train_f1_r, 'o-', color=self.colors['regularized'], 
                           linewidth=2.5, markersize=6, label=f'{self.method_label_short} Train', alpha=0.9)
                if val_f1_r:
                    ax.plot(epochs_reg[:len(val_f1_r)], val_f1_r, 's--', color=self.colors['regularized'], 
                           linewidth=2, markersize=5, label=f'{self.method_label_short} Val', alpha=0.7)
            
            ax.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
            ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
            ax.set_title('Сравнение F1-Score во время обучения', fontsize=15, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            ax.set_ylim([0, 1.05])
            ax.tick_params(labelsize=11)
            
            # ROC-AUC
            ax = axes[1, 0]
            if has_vanilla_history and vanilla_history.get('train_metrics'):
                train_roc_v = [m['roc_auc'] for m in vanilla_history['train_metrics'] if m]
                val_roc_v = [m['roc_auc'] for m in vanilla_history.get('val_metrics', []) if m is not None]
                if train_roc_v:
                    ax.plot(epochs_vanilla[:len(train_roc_v)], train_roc_v, 'o-', color=self.colors['vanilla'], 
                           linewidth=2.8, markersize=7, label='Vanilla Train', alpha=0.95, zorder=3)
                if val_roc_v:
                    ax.plot(epochs_vanilla[:len(val_roc_v)], val_roc_v, 's--', color=self.colors['vanilla'], 
                           linewidth=2.5, markersize=6, label='Vanilla Val', alpha=0.75, zorder=2)
            
            if regularized_history.get('train_metrics'):
                train_roc_r = [m['roc_auc'] for m in regularized_history['train_metrics'] if m]
                val_roc_r = [m['roc_auc'] for m in regularized_history.get('val_metrics', []) if m is not None]
                if train_roc_r:
                    ax.plot(epochs_reg[:len(train_roc_r)], train_roc_r, 'o-', color=self.colors['regularized'], 
                           linewidth=2.8, markersize=7, label=f'{self.method_name_en} Train', alpha=0.95, zorder=3)
                if val_roc_r:
                    ax.plot(epochs_reg[:len(val_roc_r)], val_roc_r, 's--', color=self.colors['regularized'], 
                           linewidth=2.5, markersize=6, label=f'{self.method_name_en} Val', alpha=0.75, zorder=2)
            
            ax.set_xlabel('Эпоха', fontsize=14, fontweight='bold', color='#212121')
            ax.set_ylabel('ROC-AUC', fontsize=14, fontweight='bold', color='#212121')
            ax.set_title('Сравнение ROC-AUC во время обучения', fontsize=16, fontweight='bold', pad=18, color='#212121')
            ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#B0BEC5', zorder=0)
            ax.set_ylim([0, 1.05])
            ax.tick_params(labelsize=12, colors='#212121')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Время обучения
            ax = axes[1, 1]
            methods = ['Vanilla ANFIS', self.method_model_label_short]
            times = []
            if has_vanilla_history and vanilla_history.get('epoch_times'):
                times.append(sum(vanilla_history['epoch_times']))
            elif vanilla_history.get('train_metrics'):
                times.append(vanilla_history.get('training_time', 0))
            else:
                times.append(0)
            
            if regularized_history.get('epoch_times'):
                times.append(sum(regularized_history['epoch_times']))
            elif regularized_history.get('train_metrics'):
                times.append(regularized_history.get('training_time', 0))
            else:
                times.append(0)
            
            bars = ax.bar(methods, times, color=[self.colors['vanilla'], self.colors['regularized']], 
                         alpha=0.95, edgecolor='#212121', linewidth=2.5, zorder=3)
            ax.set_ylabel('Время обучения (сек)', fontsize=14, fontweight='bold', color='#212121')
            ax.set_title('Время обучения', fontsize=16, fontweight='bold', pad=18, color='#212121')
            ax.set_xticklabels(methods, fontsize=13, color='#212121')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
            ax.tick_params(labelsize=12, colors='#212121')
            
            # Добавляем значения на столбцы с улучшенным форматированием
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                           f'{height:.2f} сек', ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='#212121',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', 
                                    edgecolor='#424242', linewidth=1.5, alpha=0.9))
            
            # Добавляем отношение времени
            if times[0] > 0 and times[1] > 0:
                ratio = times[1] / times[0]
                ax.text(0.5, 0.95, f'Отношение: {ratio:.2f}x', 
                       transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', 
                                edgecolor=self.colors['regularized'], linewidth=2, alpha=0.9))
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout(pad=3.5, rect=[0, 0, 1, 0.98])
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_predictions_comparison(self, vanilla_pred, regularized_pred, posthoc_vanilla_shap, 
                                     posthoc_reg_shap, feature_names, save_name='predictions_comparison.png'):
        """Сравнение предсказаний и важности признаков post-hoc"""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('white')
        
        # 1. Scatter plot предсказаний
        ax1 = plt.subplot(2, 3, 1)
        correlation = np.corrcoef(vanilla_pred, regularized_pred)[0, 1]
        # Используем разнообразные цвета для точек scatter plot
        scatter_colors = [self.colors['blue2'], self.colors['purple2'], self.colors['teal2'], 
                         self.colors['green2'], self.colors['orange2']]
        color_array = [scatter_colors[i % len(scatter_colors)] for i in range(len(vanilla_pred))]
        plt.scatter(vanilla_pred, regularized_pred, alpha=0.7, s=60, c=color_array, 
                   edgecolors='#212121', linewidth=0.8)
        min_val = min(vanilla_pred.min(), regularized_pred.min())
        max_val = max(vanilla_pred.max(), regularized_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--', color=self.colors['gray1'], alpha=0.8, linewidth=2.5, 
                label='Идеальная корреляция')
        plt.xlabel('Vanilla ANFIS предсказания', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel(f'{self.method_label_short} предсказания', fontsize=14, fontweight='bold', color='#212121')
        plt.title(f'Корреляция предсказаний (r = {correlation:.4f})', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Сравнение важности признаков Post-hoc (Vanilla)
        ax2 = plt.subplot(2, 3, 2)
        top_n = min(15, len(feature_names))
        sorted_idx_v = np.argsort(np.abs(posthoc_vanilla_shap))[::-1][:top_n]
        posthoc_v_norm = posthoc_vanilla_shap / (np.sum(np.abs(posthoc_vanilla_shap)) + 1e-8)
        
        # Используем разнообразные цвета для Vanilla (синие, фиолетовые, бирюзовые)
        color_palette_v = [self.colors['blue1'], self.colors['blue2'], self.colors['purple1'], 
                          self.colors['purple2'], self.colors['teal1'], self.colors['teal2']]
        colors_bars = [color_palette_v[i % len(color_palette_v)] for i in range(top_n)]
        bars = plt.barh(range(top_n), posthoc_v_norm[sorted_idx_v], color=colors_bars, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:20] for i in sorted_idx_v], fontsize=9)
        plt.xlabel('Важность признака', fontsize=13, fontweight='bold')
        plt.title(f'Топ-{top_n} признаков (Post-hoc, Vanilla)', fontsize=15, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')
        plt.tick_params(labelsize=10)
        
        # 3. Сравнение важности признаков Post-hoc (Regularized)
        ax3 = plt.subplot(2, 3, 3)
        sorted_idx_r = np.argsort(np.abs(posthoc_reg_shap))[::-1][:top_n]
        posthoc_r_norm = posthoc_reg_shap / (np.sum(np.abs(posthoc_reg_shap)) + 1e-8)
        
        # Используем разнообразные цвета для Regularized (зеленые, оранжевые, красные)
        color_palette_r = [self.colors['green1'], self.colors['green2'], self.colors['green3'],
                          self.colors['orange1'], self.colors['orange2'], self.colors['red1']]
        colors_bars = [color_palette_r[i % len(color_palette_r)] for i in range(top_n)]
        bars = plt.barh(range(top_n), posthoc_r_norm[sorted_idx_r], color=colors_bars, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:20] for i in sorted_idx_r], fontsize=10, color='#212121')
        plt.xlabel('Важность признака', fontsize=14, fontweight='bold', color='#212121')
        plt.title(
            f'Топ-{top_n} признаков (Post-hoc, {self.method_label_short})',
            fontsize=16,
            fontweight='bold',
            pad=18,
            color='#212121'
        )
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='x', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=11, colors='#212121')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Сравнение важности признаков (бок о бок)
        ax4 = plt.subplot(2, 3, 4)
        # Берем топ-10 признаков из обеих моделей
        top_10 = min(10, len(feature_names))
        common_features = set(sorted_idx_v[:top_10]) | set(sorted_idx_r[:top_10])
        common_features = sorted(list(common_features), key=lambda x: posthoc_v_norm[x] + posthoc_r_norm[x], reverse=True)[:top_10]
        
        x_pos = np.arange(len(common_features))
        width = 0.35
        
        # Используем более контрастные цвета для сравнения
        bars1 = plt.bar(x_pos - width/2, [posthoc_v_norm[i] for i in common_features], width,
                       label='Post-hoc Vanilla', color=self.colors['blue1'], alpha=0.9, 
                       edgecolor='#212121', linewidth=2)
        bars2 = plt.bar(x_pos + width/2, [posthoc_r_norm[i] for i in common_features], width,
                       label=f'Post-hoc {self.method_label_short}', color=self.colors['green2'], alpha=0.9,
                       edgecolor='#212121', linewidth=2)
        
        plt.xlabel('Признаки', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel('Нормализованная важность', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Post-hoc SHAP: сравнение важности', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.xticks(x_pos, [feature_names[i][:12] for i in common_features], rotation=45, ha='right', fontsize=10, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=11, colors='#212121')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Корреляция важностей признаков
        ax5 = plt.subplot(2, 3, 5)
        corr_importance = np.corrcoef(posthoc_v_norm, posthoc_r_norm)[0, 1]
        # Используем разнообразные цвета для scatter plot
        scatter_colors = [self.colors['blue2'], self.colors['purple2'], self.colors['teal2'], 
                         self.colors['green2'], self.colors['orange2']]
        color_array = [scatter_colors[i % len(scatter_colors)] for i in range(len(posthoc_v_norm))]
        plt.scatter(posthoc_v_norm, posthoc_r_norm, alpha=0.6, s=50, c=color_array, 
                   edgecolors='black', linewidth=0.5)
        min_imp = min(posthoc_v_norm.min(), posthoc_r_norm.min())
        max_imp = max(posthoc_v_norm.max(), posthoc_r_norm.max())
        plt.plot([min_imp, max_imp], [min_imp, max_imp], '--', color=self.colors['gray1'], alpha=0.8, linewidth=2.5)
        plt.xlabel('Post-hoc важность (Vanilla)', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel(f'Post-hoc важность ({self.method_label_short})', fontsize=14, fontweight='bold', color='#212121')
        plt.title(f'Корреляция важностей признаков\n(r = {corr_importance:.4f})', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        plt.tight_layout(pad=3.5, rect=[0, 0, 1, 0.98])
        plt.suptitle('Предсказания и SHAP (Post-hoc)', 
                    fontsize=19, fontweight='bold', y=0.995, color='#212121')
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_rules_comparison_detailed(self, rules_comparison, feature_names, save_name='rules_comparison_detailed.png'):
        """Детальное сравнение правил Vanilla ANFIS и ANFIS с Baseline-Masked Feature Sensitivity Regularization"""
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('white')
        
        similarities = rules_comparison['rule_similarities']
        coeff_diffs = rules_comparison['coefficient_differences']
        mf_diffs = rules_comparison['mf_differences']
        
        # 1. Сходство правил
        ax1 = plt.subplot(3, 3, 1)
        # Используем разнообразные цвета для сходства (зеленые, оранжевые)
        color_palette_sim = [self.colors['green3'], self.colors['green2'], self.colors['green1'],
                            self.colors['orange2'], self.colors['orange1'], self.colors['teal2']]
        colors_sim = [color_palette_sim[i % len(color_palette_sim)] for i, s in enumerate(similarities)]
        bars = plt.bar(range(len(similarities)), similarities, color=colors_sim, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        plt.axhline(y=rules_comparison['average_similarity'], color=self.colors['dark_blue'], linestyle='--', 
                   linewidth=2, label=f'Среднее: {rules_comparison["average_similarity"]:.3f}')
        plt.xlabel('Номер правила', fontsize=13, fontweight='bold')
        plt.ylabel('Сходство', fontsize=13, fontweight='bold')
        plt.title('Сходство правил между моделями', fontsize=15, fontweight='bold', pad=15)
        plt.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
        plt.tick_params(labelsize=11)
        plt.ylim([0, 1.1])
        
        # 2. Различия коэффициентов
        ax2 = plt.subplot(3, 3, 2)
        # Используем яркий цвет для различий коэффициентов
        plt.bar(range(len(coeff_diffs)), coeff_diffs, color=self.colors['red1'], alpha=0.9, 
               edgecolor='#212121', linewidth=2)
        plt.axhline(y=rules_comparison['average_coeff_diff'], color=self.colors['dark_blue'], linestyle='--', 
                   linewidth=2, label=f'Среднее: {rules_comparison["average_coeff_diff"]:.4f}')
        plt.xlabel('Номер правила', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel('Различие коэффициентов', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Различия коэффициентов правил', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Различия функций принадлежности
        ax3 = plt.subplot(3, 3, 3)
        # Используем яркий цвет для различий функций принадлежности
        plt.bar(range(len(mf_diffs)), mf_diffs, color=self.colors['orange1'], alpha=0.9, 
               edgecolor='#212121', linewidth=2)
        plt.axhline(y=rules_comparison['average_mf_diff'], color=self.colors['dark_blue'], linestyle='--', 
                   linewidth=2, label=f'Среднее: {rules_comparison["average_mf_diff"]:.4f}')
        plt.xlabel('Номер правила', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel('Различие функций принадлежности', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Различия функций принадлежности', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Распределение сходства
        ax4 = plt.subplot(3, 3, 4)
        # Используем разнообразные цвета для гистограммы
        plt.hist(similarities, bins=15, color=self.colors['teal2'], alpha=0.7, edgecolor='black', linewidth=1.5)
        plt.axvline(x=rules_comparison['average_similarity'], color=self.colors['dark_blue'], linestyle='--', 
                   linewidth=2, label=f'Среднее: {rules_comparison["average_similarity"]:.3f}')
        plt.xlabel('Сходство', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel('Частота', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Распределение сходства правил', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Комбинированный график различий
        ax5 = plt.subplot(3, 3, 5)
        x = range(len(coeff_diffs))
        width = 0.35
        # Используем яркие контрастные цвета для различий
        plt.bar([i - width/2 for i in x], coeff_diffs, width, label='Коэффициенты', 
               color=self.colors['red1'], alpha=0.9, edgecolor='#212121', linewidth=2)
        plt.bar([i + width/2 for i in x], mf_diffs, width, label='Функции принадлежности', 
               color=self.colors['orange1'], alpha=0.9, edgecolor='#212121', linewidth=2)
        plt.xlabel('Номер правила', fontsize=14, fontweight='bold', color='#212121')
        plt.ylabel('Различие', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Сравнение различий', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True, fancybox=True, edgecolor='#424242')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 6. Box plot различий
        ax6 = plt.subplot(3, 3, 6)
        data_to_plot = [coeff_diffs, mf_diffs]
        bp = plt.boxplot(data_to_plot, labels=['Коэффициенты', 'Функции\nпринадлежности'], 
                        patch_artist=True, widths=0.6)
        # Используем яркие контрастные цвета для box plot
        bp['boxes'][0].set_facecolor(self.colors['red1'])
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(self.colors['orange1'])
        bp['boxes'][1].set_alpha(0.8)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        plt.ylabel('Различие', fontsize=14, fontweight='bold', color='#212121')
        plt.title('Распределение различий\n(Box Plot)', fontsize=16, fontweight='bold', pad=18, color='#212121')
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='y', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        regularized_rules_count = rules_comparison['num_rules'].get(
            self.method_model_label_ru,
            'N/A'
        )

        # 7. Сводная статистика
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        
        stats_text = f"""СТАТИСТИКА СРАВНЕНИЯ ПРАВИЛ
========================================

Количество правил:
  Vanilla ANFIS:        {rules_comparison['num_rules'].get('Vanilla ANFIS', 'N/A')}
  {self.method_model_label_short}:  {regularized_rules_count}

Средние показатели:
  Сходство:            {rules_comparison['average_similarity']:.4f}
  Различие коэфф.:     {rules_comparison['average_coeff_diff']:.4f}
  Различие ФП:         {rules_comparison['average_mf_diff']:.4f}

Интерпретация:
  Сходство > 0.7:      Высокое
  Сходство 0.5-0.7:    Среднее
  Сходство < 0.5:      Низкое
"""
        
        plt.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9, 
                         edgecolor=self.colors['gray1'], linewidth=2))
        
        # 8. Heatmap сходства правил
        ax8 = plt.subplot(3, 3, 8)
        similarity_matrix = np.array(similarities).reshape(-1, 1)
        # Используем градации серого вместо RdYlGn
        from matplotlib.colors import LinearSegmentedColormap
        gray_cmap = LinearSegmentedColormap.from_list('gray', ['#ffffff', '#757575', '#212121'])
        im = plt.imshow(similarity_matrix, cmap=gray_cmap, aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax8, label='Сходство', shrink=0.8)
        plt.xlabel('Правило', fontsize=13, fontweight='bold')
        plt.ylabel('Индекс правила', fontsize=13, fontweight='bold')
        plt.title('Heatmap сходства правил', fontsize=15, fontweight='bold', pad=15)
        plt.yticks(range(len(similarities)), [f'Правило {i+1}' for i in range(len(similarities))], fontsize=9)
        plt.tick_params(labelsize=10)
        
        # 9. Информация о правилах
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        info_text = f"""ИНФОРМАЦИЯ О ПРАВИЛАХ
========================================

Правила ANFIS представляют собой
нечеткие логические правила вида:

ЕСЛИ x1 is A1 AND x2 is A2 ...
ТО y = f(x1, x2, ...)

Где:
  - Ai - функции принадлежности
  - f - линейная функция от признаков

{self.method_label_short_inline} влияет на:
  - Параметры функций принадлежности
  - Коэффициенты выходных функций
  - Распределение важности признаков

Высокое сходство правил означает,
что регуляризация сохраняет основную
логику модели.
"""
        
        plt.text(0.05, 0.95, info_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9, 
                         edgecolor=self.colors['gray1'], linewidth=2))
        
        plt.tight_layout(pad=3.5, rect=[0, 0, 1, 0.98])
        plt.suptitle(
            f'Правила: {self.short_vs_label}',
            fontsize=19,
            fontweight='bold',
            y=0.995,
            color='#212121'
        )
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_loss_comparison(self, vanilla_history, regularized_history, save_name='loss_comparison.png'):
        """Сравнение потерь во время обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        # Total Loss
        ax1 = axes[0]
        has_vanilla_loss = bool(vanilla_history.get('total_loss'))
        if has_vanilla_loss and vanilla_history.get('total_loss'):
            epochs_v = range(1, len(vanilla_history['total_loss']) + 1)
            ax1.plot(epochs_v, vanilla_history['total_loss'], 'o-', color=self.colors['vanilla'], 
                    linewidth=2.5, markersize=6, label='Vanilla ANFIS', alpha=0.9)
        if regularized_history.get('total_loss'):
            epochs_r = range(1, len(regularized_history['total_loss']) + 1)
            ax1.plot(
                epochs_r,
                regularized_history['total_loss'],
                's-',
                color=self.colors['regularized'],
                linewidth=2.5,
                markersize=6,
                label=self.method_label_inline,
                alpha=0.9
            )
        ax1.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Total Loss', fontsize=13, fontweight='bold')
        ax1.set_title('Сравнение Total Loss', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax1.tick_params(labelsize=11)
        
        # Main Loss и SHAP Loss для регуляризованной модели
        ax2 = axes[1]
        if regularized_history.get('main_loss') and regularized_history.get('shap_loss'):
            epochs_r = range(1, len(regularized_history['main_loss']) + 1)
            ax2.plot(epochs_r, regularized_history['main_loss'], 'o-', color=self.colors['dark_blue'], 
                    linewidth=2.5, markersize=6, label='Main Loss', alpha=0.9)
            shap_weighted = [h * self.config.get('shap', {}).get('gamma', 0.5) 
                            for h in regularized_history['shap_loss']]
            ax2.plot(epochs_r, shap_weighted, 's-', color=self.colors['dark_green'], 
                    linewidth=2.5, markersize=6, label=f'SHAP Loss × γ', alpha=0.9)
        ax2.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax2.set_title(
            f'Детализация потерь\n({self.method_label_inline})',
            fontsize=15,
            fontweight='bold',
            pad=15
        )
        ax2.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax2.tick_params(labelsize=11)
        
        plt.tight_layout(pad=3.0)
        plt.suptitle('Сравнение потерь во время обучения', fontsize=17, fontweight='bold', y=0.98)
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_roc_comparison(self, vanilla_results, regularized_results, y_test, save_name='roc_comparison.png'):
        """Сравнение ROC кривых"""
        if self.task_type != 'classification':
            return
        
        fig = plt.figure(figsize=(14, 11))
        fig.patch.set_facecolor('white')
        ax = plt.gca()
        
        y_prob_vanilla = np.array(vanilla_results['probabilities']).flatten()
        y_prob_reg = np.array(regularized_results['probabilities']).flatten()
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test).flatten()
        
        fpr_v, tpr_v, _ = roc_curve(y_test_values, y_prob_vanilla)
        fpr_r, tpr_r, _ = roc_curve(y_test_values, y_prob_reg)
        
        auc_v = vanilla_results['metrics']['roc_auc']
        auc_r = regularized_results['metrics']['roc_auc']
        
        # Основные кривые
        plt.plot(fpr_v, tpr_v, color=self.colors['vanilla'], linewidth=3.5, alpha=0.95,
                label=f'Vanilla ANFIS (AUC = {auc_v:.4f})', zorder=3)
        plt.plot(
            fpr_r,
            tpr_r,
            color=self.colors['regularized'],
            linewidth=3.5,
            alpha=0.95,
            label=f'{self.method_label_inline} (AUC = {auc_r:.4f})',
            zorder=3
        )
        plt.plot([0, 1], [0, 1], color='#757575', linewidth=2.5, linestyle='--', alpha=0.6, 
                label='Случайная классификация', zorder=1)
        
        # Заливка под кривыми для лучшей визуализации
        ax.fill_between(fpr_v, tpr_v, alpha=0.2, color=self.colors['vanilla'], zorder=2)
        ax.fill_between(fpr_r, tpr_r, alpha=0.2, color=self.colors['regularized'], zorder=2)
        
        # Аннотация улучшения
        improvement = auc_r - auc_v
        if improvement > 0:
            # Находим точку на кривой для аннотации
            mid_idx = len(fpr_r) // 2
            ax.annotate(f'Улучшение: +{improvement:.4f}', 
                       xy=(fpr_r[mid_idx], tpr_r[mid_idx]), 
                       xytext=(0.6, 0.3),
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', 
                                edgecolor=self.colors['regularized'], linewidth=2),
                       arrowprops=dict(arrowstyle='->', color=self.colors['regularized'], lw=2))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold', color='#212121')
        plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold', color='#212121')
        plt.title(f'ROC: {self.short_vs_label}', fontsize=19, fontweight='bold', pad=25, color='#212121')
        plt.legend(loc="lower right", fontsize=13, framealpha=0.95, shadow=True, 
                  fancybox=True, edgecolor='#424242', frameon=True)
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=13, colors='#212121')
        
        # Убираем верхнюю и правую границы
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#424242')
        ax.spines['bottom'].set_color('#424242')
        
        plt.tight_layout(pad=3.0)
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_metrics_bar_comparison(self, vanilla_results, regularized_results, save_name='metrics_bar_comparison.png'):
        """Сравнение метрик в виде столбчатой диаграммы"""
        if self.task_type != 'classification':
            return
        
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('white')
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        vanilla_scores = [vanilla_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        regularized_scores = [regularized_results['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        # Используем более контрастные цвета для метрик
        bars1 = plt.bar(x - width/2, vanilla_scores, width, label='Vanilla ANFIS', 
                       color=self.colors['blue1'], alpha=0.95, edgecolor='#212121', linewidth=2.5,
                       zorder=3)
        bars2 = plt.bar(x + width/2, regularized_scores, width, label=self.method_label_short, 
                       color=self.colors['green2'], alpha=0.95, edgecolor='#212121', linewidth=2.5,
                       zorder=3)
        
        # Добавляем значения на столбцы
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            h1 = bar1.get_height()
            h2 = bar2.get_height()
            
            plt.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.01,
                    f'{h1:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=self.colors['blue1'])
            plt.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.01,
                    f'{h2:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=self.colors['green2'])
        
        plt.xlabel('Метрики', fontsize=15, fontweight='bold')
        plt.ylabel('Значения', fontsize=15, fontweight='bold')
        plt.title(f'Метрики: {self.short_vs_label}', fontsize=18, fontweight='bold', pad=20)
        plt.xticks(x, metrics_names, fontsize=13)
        plt.legend(loc='upper left', fontsize=13, framealpha=0.95, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
        plt.ylim([0, 1.15])
        plt.tick_params(labelsize=12)
        
        plt.tight_layout(pad=3.0)
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_feature_importance_comparison(self, vanilla_importance, regularized_importance, 
                                           posthoc_vanilla, posthoc_reg, feature_names, 
                                           save_name='feature_importance_comparison.png'):
        """Сравнение важности SHAP-признаков: встроенная vs post-hoc"""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('white')
        
        top_n = min(15, len(feature_names))
        
        # Нормализация
        vanilla_norm = vanilla_importance / (np.sum(vanilla_importance) + 1e-8)
        regularized_norm = regularized_importance / (np.sum(regularized_importance) + 1e-8)
        posthoc_v_norm = posthoc_vanilla / (np.sum(np.abs(posthoc_vanilla)) + 1e-8)
        posthoc_r_norm = posthoc_reg / (np.sum(np.abs(posthoc_reg)) + 1e-8)
        
        # 1. Встроенная важность Vanilla
        ax1 = plt.subplot(2, 2, 1)
        sorted_idx = np.argsort(vanilla_norm)[::-1][:top_n]
        # Используем разнообразные цвета для Vanilla
        color_palette_v = [self.colors['blue1'], self.colors['blue2'], self.colors['purple1'], 
                          self.colors['purple2'], self.colors['teal1'], self.colors['teal2']]
        colors = [color_palette_v[i % len(color_palette_v)] for i in range(top_n)]
        plt.barh(range(top_n), vanilla_norm[sorted_idx], color=colors, alpha=0.8, 
                edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:25] for i in sorted_idx], fontsize=10)
        plt.xlabel('Важность', fontsize=13, fontweight='bold')
        plt.title('Встроенная важность (Vanilla ANFIS)', fontsize=15, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')
        plt.tick_params(labelsize=11)
        
        # 2. Встроенная важность Regularized
        ax2 = plt.subplot(2, 2, 2)
        sorted_idx_r = np.argsort(regularized_norm)[::-1][:top_n]
        # Используем разнообразные цвета для Regularized
        color_palette_r = [self.colors['green1'], self.colors['green2'], self.colors['green3'],
                          self.colors['orange1'], self.colors['orange2'], self.colors['red1']]
        colors = [color_palette_r[i % len(color_palette_r)] for i in range(top_n)]
        plt.barh(range(top_n), regularized_norm[sorted_idx_r], color=colors, alpha=0.8, 
                edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:25] for i in sorted_idx_r], fontsize=10)
        plt.xlabel('Важность', fontsize=13, fontweight='bold')
        plt.title(
            f'Встроенная важность ({self.method_label_short})',
            fontsize=15,
            fontweight='bold',
            pad=15
        )
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')
        plt.tick_params(labelsize=11)
        
        # 3. Post-hoc важность Vanilla
        ax3 = plt.subplot(2, 2, 3)
        sorted_idx_ph_v = np.argsort(np.abs(posthoc_v_norm))[::-1][:top_n]
        # Используем градиент синих оттенков для Post-hoc Vanilla
        colors = [self.colors['blue1'] if i < top_n//3 else self.colors['blue2'] if i < 2*top_n//3 else self.colors['blue3'] for i in range(top_n)]
        plt.barh(range(top_n), posthoc_v_norm[sorted_idx_ph_v], color=colors, alpha=0.8, 
                edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:25] for i in sorted_idx_ph_v], fontsize=10)
        plt.xlabel('Важность', fontsize=13, fontweight='bold')
        plt.title('Post-hoc важность (Vanilla ANFIS)', fontsize=15, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')
        plt.tick_params(labelsize=11)
        
        # 4. Post-hoc важность Regularized
        ax4 = plt.subplot(2, 2, 4)
        sorted_idx_ph_r = np.argsort(np.abs(posthoc_r_norm))[::-1][:top_n]
        # Используем градиент зеленых оттенков для Post-hoc BMFSR
        colors = [self.colors['green1'] if i < top_n//3 else self.colors['green2'] if i < 2*top_n//3 else self.colors['green3'] for i in range(top_n)]
        plt.barh(range(top_n), posthoc_r_norm[sorted_idx_ph_r], color=colors, alpha=0.8, 
                edgecolor='black', linewidth=1)
        plt.yticks(range(top_n), [feature_names[i][:25] for i in sorted_idx_ph_r], fontsize=11, color='#212121')
        plt.xlabel('Важность', fontsize=14, fontweight='bold', color='#212121')
        plt.title(
            f'Post-hoc важность ({self.method_label_short})',
            fontsize=16,
            fontweight='bold',
            pad=18,
            color='#212121'
        )
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='x', color='#B0BEC5', zorder=0)
        plt.tick_params(labelsize=12, colors='#212121')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout(pad=3.5, rect=[0, 0, 1, 0.98])
        plt.suptitle('SHAP важность: встроенная vs Post-hoc', 
                    fontsize=19, fontweight='bold', y=0.995, color='#212121')
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_training_time_comparison(self, save_name='08_training_time_comparison.png'):
        """Сравнение времени обучения и post-hoc анализа в одном столбце для каждой модели"""
        import json
        
        if not self.save_dir:
            print("[WARNING] save_dir не указан, график не будет сохранен")
            return
        
        # Загружаем данные о времени
        time_analysis_path = self.save_dir / 'time_analysis.json'
        
        if not time_analysis_path.exists():
            print(f"[WARNING] Файл {time_analysis_path} не найден")
            return
        posthoc_params_path = self.save_dir / 'posthoc_parameters.json'
        if not posthoc_params_path.exists():
            print(f"[WARNING] Файл {posthoc_params_path} не найден")
            return
        
        # Читаем данные
        with open(time_analysis_path, 'r') as f:
            time_data = json.load(f)
        with open(posthoc_params_path, 'r') as f:
            posthoc_data = json.load(f)
        
        # Извлекаем времена обучения (только обучение, без post-hoc)
        vanilla_train_time = time_data.get('vanilla_time', 0)
        regularized_train_time = time_data.get('regularized_time', 0)
        speedup = time_data.get('speedup', 0)
        # Времена post-hoc
        vanilla_posthoc_time = posthoc_data.get('vanilla', {}).get('analysis_time', 0)
        regularized_posthoc_time = posthoc_data.get('regularized', {}).get('analysis_time', 0)
        vanilla_total = vanilla_train_time + vanilla_posthoc_time
        regularized_total = regularized_train_time + regularized_posthoc_time
        
        # Создаем единый график
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        # Данные для столбчатой диаграммы
        methods = ['Vanilla ANFIS', self.method_model_label_short]
        train_times = [vanilla_train_time, regularized_train_time]
        posthoc_times = [vanilla_posthoc_time, regularized_posthoc_time]
        
        # Цвета для столбцов
        colors = [self.colors['blue2'], self.colors['green2']]
        posthoc_colors = [self.colors['orange1'], self.colors['orange2']]
        
        x = np.arange(len(methods))
        width = 0.55
        # Нижний сегмент — само обучение
        bars_train = ax.bar(x, train_times, color=colors, alpha=0.95,
                            edgecolor='black', linewidth=1.8, width=width,
                            label='Обучение модели')
        # Верхний сегмент — post-hoc SHAP
        bars_posthoc = ax.bar(x, posthoc_times, bottom=train_times,
                              color=[posthoc_colors[i % len(posthoc_colors)] for i in range(len(methods))],
                              alpha=0.85, width=width,
                              edgecolor='#424242', linewidth=1.5,
                              label='Post-hoc SHAP анализ')
        
        # Подписи времени для каждого сегмента
        for bar, time_val in zip(bars_train, train_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{time_val:.2f} c', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
        for bar, base, extra in zip(bars_posthoc, train_times, posthoc_times):
            if extra <= 0:
                continue
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., base + height/2,
                    f'+{extra:.2f} c', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#212121')
        # Общее значение на вершине
        for xpos, total in zip(x, [vanilla_total, regularized_total]):
            ax.text(xpos, total + max(vanilla_total, regularized_total)*0.015,
                    f'Σ {total:.2f} c', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#212121')
        
        # Надпись о соотношении скоростей без дополнительной панели
        if speedup > 0:
            total_speed_ratio = (vanilla_total / regularized_total) if regularized_total > 0 else speedup
            display_ratio = max(total_speed_ratio, 1/total_speed_ratio) if total_speed_ratio > 0 else speedup
            ratio_text = f'Отношение скоростей (по сумме): {display_ratio:.2f}x'
            ax.text(0.98, 0.95, ratio_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8,
                              edgecolor='#424242', linewidth=1))
        
        ax.set_ylabel('Время обучения (секунды)', fontsize=14, fontweight='bold', color='#212121')
        ax.set_title(f'Время: {self.short_vs_label}', fontsize=17, fontweight='bold', 
                     pad=15, color='#212121')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', axis='y', linewidth=0.9)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#212121', labelsize=12)
        ax.legend(loc='upper left', framealpha=0.9)
        
        plt.tight_layout(pad=2.0)
        
        # Сохранение
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"[INFO] График времени обучения сохранен: {save_path}")
        plt.close()

