"""
Графики для сравнения gamma для конкретного датасета
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

class DatasetGammaPlotter:
    """Класс для создания графиков сравнения gamma для конкретного датасета"""
    
    def __init__(self, dataset_name, results_dir=None):
        self.dataset_name = dataset_name
        if results_dir is None:
            self.results_dir = Path(f'results/{dataset_name}/gamma_experiments')
        else:
            self.results_dir = Path(results_dir)
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("muted")
        
        # Цветовая палитра
        self.colors = {
            'gamma_0.1': '#42a5f5',  # Голубой
            'gamma_0.3': '#1565c0',  # Синий
            'gamma_0.5': '#2e7d32',  # Зеленый
            'gamma_0.7': '#e65100',  # Оранжевый
            'vanilla': '#757575',    # Серый
        }
        
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
        })
    
    def load_summary(self):
        """Загрузка сводных результатов"""
        summary_path = self.results_dir / 'summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(f"Файл {summary_path} не найден")
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        return summary
    
    def create_metrics_comparison(self, save_name='gamma_metrics_comparison.png'):
        """Сравнение метрик для разных значений gamma"""
        summary = self.load_summary()
        
        # Подготовка данных
        gammas = []
        vanilla_acc = []
        regularized_acc = []
        vanilla_roc = []
        regularized_roc = []
        
        for exp in summary['experiments']:
            gammas.append(exp['gamma'])
            vanilla_acc.append(exp['vanilla']['metrics'].get('accuracy', 0))
            regularized_acc.append(exp['regularized']['metrics'].get('accuracy', 0))
            vanilla_roc.append(exp['vanilla']['metrics'].get('roc_auc', 0))
            regularized_roc.append(exp['regularized']['metrics'].get('roc_auc', 0))
        
        # Создание графика
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Сравнение метрик для разных gamma: {self.dataset_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        x = np.arange(len(gammas))
        width = 0.35
        
        # Accuracy
        ax1 = axes[0]
        ax1.bar(x - width/2, vanilla_acc, width, label='Vanilla', 
               color=self.colors['vanilla'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax1.bar(x + width/2, regularized_acc, width, label='Regularized', 
               color=self.colors['gamma_0.5'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Gamma', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'γ={g}' for g in gammas])
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.set_ylim([0, 1.05])
        
        # ROC-AUC
        ax2 = axes[1]
        ax2.bar(x - width/2, vanilla_roc, width, label='Vanilla', 
               color=self.colors['vanilla'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax2.bar(x + width/2, regularized_roc, width, label='Regularized', 
               color=self.colors['gamma_0.5'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Gamma', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
        ax2.set_title('ROC-AUC', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'γ={g}' for g in gammas])
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_improvement_plot(self, save_name='gamma_improvement_heatmap.png'):
        """График улучшения метрик"""
        summary = self.load_summary()
        
        gammas = []
        improvements_roc = []
        improvements_acc = []
        
        for exp in summary['experiments']:
            gammas.append(exp['gamma'])
            vanilla_roc = exp['vanilla']['metrics'].get('roc_auc', 0)
            regularized_roc = exp['regularized']['metrics'].get('roc_auc', 0)
            vanilla_acc = exp['vanilla']['metrics'].get('accuracy', 0)
            regularized_acc = exp['regularized']['metrics'].get('accuracy', 0)
            
            improvements_roc.append(regularized_roc - vanilla_roc)
            improvements_acc.append(regularized_acc - vanilla_acc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        x = np.arange(len(gammas))
        width = 0.35
        
        ax.bar(x - width/2, improvements_roc, width, label='ROC-AUC', 
              color=self.colors['gamma_0.5'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, improvements_acc, width, label='Accuracy', 
              color=self.colors['gamma_0.7'], alpha=0.9, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Gamma', fontsize=12, fontweight='bold')
        ax.set_ylabel('Улучшение', fontsize=12, fontweight='bold')
        ax.set_title(f'Улучшение метрик для разных gamma: {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'γ={g}' for g in gammas])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_time_comparison(self, save_name='gamma_time_comparison.png'):
        """Сравнение времени обучения"""
        summary = self.load_summary()
        
        gammas = []
        vanilla_times = []
        regularized_times = []
        
        for exp in summary['experiments']:
            gammas.append(exp['gamma'])
            vanilla_times.append(exp['vanilla']['training_time'])
            regularized_times.append(exp['regularized']['training_time'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        x = np.arange(len(gammas))
        width = 0.35
        
        ax.bar(x - width/2, vanilla_times, width, label='Vanilla', 
              color=self.colors['vanilla'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, regularized_times, width, label='Regularized', 
              color=self.colors['gamma_0.5'], alpha=0.9, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Gamma', fontsize=12, fontweight='bold')
        ax.set_ylabel('Время обучения (сек)', fontsize=12, fontweight='bold')
        ax.set_title(f'Время обучения для разных gamma: {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'γ={g}' for g in gammas])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_optimal_gamma_analysis(self, save_name='optimal_gamma_analysis.png'):
        """Анализ оптимального gamma"""
        summary = self.load_summary()
        
        gammas = []
        improvements_roc = []
        
        for exp in summary['experiments']:
            gammas.append(exp['gamma'])
            vanilla_roc = exp['vanilla']['metrics'].get('roc_auc', 0)
            regularized_roc = exp['regularized']['metrics'].get('roc_auc', 0)
            improvements_roc.append(regularized_roc - vanilla_roc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        ax.plot(gammas, improvements_roc, 'o-', linewidth=2.5, markersize=10, 
               color=self.colors['gamma_0.5'], alpha=0.8, label='Улучшение ROC-AUC')
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Gamma', fontsize=12, fontweight='bold')
        ax.set_ylabel('Улучшение ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_title(f'Зависимость улучшения ROC-AUC от gamma: {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Находим оптимальное gamma
        best_idx = np.argmax(improvements_roc)
        best_gamma = gammas[best_idx]
        best_improvement = improvements_roc[best_idx]
        
        ax.annotate(f'Оптимальное: γ={best_gamma}\nУлучшение: {best_improvement:+.4f}',
                   xy=(best_gamma, best_improvement),
                   xytext=(best_gamma + 0.1, best_improvement + 0.01),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_summary_report(self, save_name='gamma_summary.txt'):
        """Создание текстового отчета"""
        summary = self.load_summary()
        
        report_lines = [
            f"=== ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ С GAMMA: {self.dataset_name.upper()} ===\n",
            f"Всего экспериментов: {summary['total_experiments']}\n",
            "РЕЗУЛЬТАТЫ:\n",
            f"{'Gamma':<10} {'Vanilla ROC':<15} {'Regularized ROC':<18} {'Улучшение':<12} {'Vanilla Acc':<15} {'Regularized Acc':<18} {'Время (с)':<12}",
            "-" * 100
        ]
        
        for exp in sorted(summary['experiments'], key=lambda x: x['gamma']):
            gamma = exp['gamma']
            v_roc = exp['vanilla']['metrics'].get('roc_auc', 0)
            r_roc = exp['regularized']['metrics'].get('roc_auc', 0)
            v_acc = exp['vanilla']['metrics'].get('accuracy', 0)
            r_acc = exp['regularized']['metrics'].get('accuracy', 0)
            time = exp['regularized']['training_time']
            improvement = r_roc - v_roc
            
            report_lines.append(
                f"{gamma:<10} {v_roc:<15.4f} {r_roc:<18.4f} {improvement:+.4f}        {v_acc:<15.4f} {r_acc:<18.4f} {time:<12.2f}"
            )
        
        # Находим лучший результат
        best_exp = max(summary['experiments'], 
                      key=lambda x: x['regularized']['metrics'].get('roc_auc', 0))
        best_gamma = best_exp['gamma']
        best_roc = best_exp['regularized']['metrics'].get('roc_auc', 0)
        best_improvement = best_roc - best_exp['vanilla']['metrics'].get('roc_auc', 0)
        
        report_lines.extend([
            "\n",
            "ВЫВОДЫ:\n",
            f"Оптимальное значение gamma: {best_gamma}",
            f"Лучший ROC-AUC: {best_roc:.4f}",
            f"Улучшение относительно Vanilla: {best_improvement:+.4f}"
        ])
        
        report_text = "\n".join(report_lines)
        
        save_path = self.results_dir / save_name
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"[INFO] Отчет сохранен: {save_path}")
        print("\n" + report_text)
    
    def create_all_plots(self):
        """Создание всех графиков и отчета"""
        print(f"\n[INFO] Создание графиков для {self.dataset_name}...")
        
        try:
            self.create_metrics_comparison()
            self.create_improvement_plot()
            self.create_time_comparison()
            self.create_optimal_gamma_analysis()
            self.create_summary_report()
            print(f"\n[OK] Все графики и отчет созданы для {self.dataset_name}!")
        except Exception as e:
            print(f"\n[ERROR] Ошибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'breast_cancer'
    plotter = DatasetGammaPlotter(dataset_name)
    plotter.create_all_plots()

