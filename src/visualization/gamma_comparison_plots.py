"""
Графики для сравнения результатов экспериментов с разными значениями gamma
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from matplotlib.patches import Rectangle

class GammaComparisonPlotter:
    """Класс для создания графиков сравнения результатов с разными gamma"""
    
    def __init__(self, results_dir='results/gamma_experiments'):
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
    
    def create_gamma_comparison_metrics(self, save_name='gamma_metrics_comparison.png'):
        """Сравнение метрик для разных значений gamma"""
        summary = self.load_summary()
        
        # Подготовка данных
        data = []
        for exp in summary['experiments']:
            dataset = exp['dataset']
            gamma = exp['gamma']
            vanilla_metrics = exp['vanilla']['metrics']
            regularized_metrics = exp['regularized']['metrics']
            
            if 'accuracy' in vanilla_metrics:  # Классификация
                data.append({
                    'dataset': dataset,
                    'gamma': gamma,
                    'metric': 'Accuracy',
                    'vanilla': vanilla_metrics['accuracy'],
                    'regularized': regularized_metrics['accuracy']
                })
                data.append({
                    'dataset': dataset,
                    'gamma': gamma,
                    'metric': 'ROC-AUC',
                    'vanilla': vanilla_metrics['roc_auc'],
                    'regularized': regularized_metrics['roc_auc']
                })
                data.append({
                    'dataset': dataset,
                    'gamma': gamma,
                    'metric': 'F1-Score',
                    'vanilla': vanilla_metrics['f1_score'],
                    'regularized': regularized_metrics['f1_score']
                })
        
        df = pd.DataFrame(data)
        
        # Создание графика
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('white')
        
        metrics = ['Accuracy', 'ROC-AUC', 'F1-Score']
        datasets = df['dataset'].unique()
        gammas = sorted(df['gamma'].unique())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_data = df[df['metric'] == metric]
            
            x = np.arange(len(datasets))
            width = 0.15
            
            # Vanilla
            vanilla_values = [metric_data[(metric_data['dataset'] == d)]['vanilla'].iloc[0] 
                            for d in datasets]
            ax.bar(x - width*2, vanilla_values, width, label='Vanilla', 
                  color=self.colors['vanilla'], alpha=0.9, edgecolor='black', linewidth=1.5)
            
            # Разные gamma
            for i, gamma in enumerate(gammas):
                gamma_values = [metric_data[(metric_data['dataset'] == d) & 
                                          (metric_data['gamma'] == gamma)]['regularized'].iloc[0] 
                              if len(metric_data[(metric_data['dataset'] == d) & 
                                                (metric_data['gamma'] == gamma)]) > 0 else 0
                              for d in datasets]
                ax.bar(x - width + i*width, gamma_values, width, 
                      label=f'γ={gamma}', 
                      color=self.colors[f'gamma_{gamma}'], 
                      alpha=0.9, edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Датасет', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'Сравнение {metric}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=15, ha='right')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0.85, 1.0])
        
        plt.tight_layout(pad=3.0)
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_gamma_improvement_heatmap(self, save_name='gamma_improvement_heatmap.png'):
        """Heatmap улучшения ROC-AUC для разных gamma и датасетов"""
        summary = self.load_summary()
        
        # Подготовка данных
        datasets = []
        gammas = []
        improvements = []
        
        for exp in summary['experiments']:
            datasets.append(exp['dataset'])
            gammas.append(exp['gamma'])
            improvement = (exp['regularized']['metrics'].get('roc_auc', 0) - 
                          exp['vanilla']['metrics'].get('roc_auc', 0))
            improvements.append(improvement)
        
        df = pd.DataFrame({
            'dataset': datasets,
            'gamma': gammas,
            'improvement': improvements
        })
        
        # Создание матрицы
        datasets_unique = sorted(df['dataset'].unique())
        gammas_unique = sorted(df['gamma'].unique())
        
        matrix = np.zeros((len(datasets_unique), len(gammas_unique)))
        
        for i, dataset in enumerate(datasets_unique):
            for j, gamma in enumerate(gammas_unique):
                value = df[(df['dataset'] == dataset) & (df['gamma'] == gamma)]['improvement']
                if len(value) > 0:
                    matrix[i, j] = value.iloc[0]
        
        # Создание графика
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.01, vmax=0.01)
        
        # Подписи
        ax.set_xticks(np.arange(len(gammas_unique)))
        ax.set_yticks(np.arange(len(datasets_unique)))
        ax.set_xticklabels([f'γ={g}' for g in gammas_unique])
        ax.set_yticklabels(datasets_unique)
        
        # Добавление значений в ячейки
        for i in range(len(datasets_unique)):
            for j in range(len(gammas_unique)):
                text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xlabel('Значение gamma', fontsize=12, fontweight='bold')
        ax.set_ylabel('Датасет', fontsize=12, fontweight='bold')
        ax.set_title('Улучшение ROC-AUC для разных gamma и датасетов', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Цветовая шкала
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Улучшение ROC-AUC', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_gamma_time_comparison(self, save_name='gamma_time_comparison.png'):
        """Сравнение времени обучения для разных gamma"""
        summary = self.load_summary()
        
        # Подготовка данных
        data = []
        for exp in summary['experiments']:
            data.append({
                'dataset': exp['dataset'],
                'gamma': exp['gamma'],
                'vanilla_time': exp['vanilla']['training_time'],
                'regularized_time': exp['regularized']['training_time']
            })
        
        df = pd.DataFrame(data)
        
        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        datasets = sorted(df['dataset'].unique())
        gammas = sorted(df['gamma'].unique())
        
        x = np.arange(len(datasets))
        width = 0.15
        
        # Vanilla
        vanilla_times = [df[(df['dataset'] == d)]['vanilla_time'].iloc[0] for d in datasets]
        ax.bar(x - width*2, vanilla_times, width, label='Vanilla', 
              color=self.colors['vanilla'], alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Разные gamma
        for i, gamma in enumerate(gammas):
            gamma_times = [df[(df['dataset'] == d) & (df['gamma'] == gamma)]['regularized_time'].iloc[0] 
                          if len(df[(df['dataset'] == d) & (df['gamma'] == gamma)]) > 0 else 0
                          for d in datasets]
            ax.bar(x - width + i*width, gamma_times, width, 
                  label=f'γ={gamma}', 
                  color=self.colors[f'gamma_{gamma}'], 
                  alpha=0.9, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Датасет', fontsize=12, fontweight='bold')
        ax.set_ylabel('Время обучения (сек)', fontsize=12, fontweight='bold')
        ax.set_title('Сравнение времени обучения для разных gamma', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_optimal_gamma_analysis(self, save_name='optimal_gamma_analysis.png'):
        """Анализ оптимального значения gamma для каждого датасета"""
        summary = self.load_summary()
        
        # Подготовка данных
        datasets = {}
        for exp in summary['experiments']:
            dataset = exp['dataset']
            gamma = exp['gamma']
            improvement = (exp['regularized']['metrics'].get('roc_auc', 0) - 
                          exp['vanilla']['metrics'].get('roc_auc', 0))
            
            if dataset not in datasets:
                datasets[dataset] = {}
            datasets[dataset][gamma] = improvement
        
        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        for dataset, gamma_data in datasets.items():
            gammas = sorted(gamma_data.keys())
            improvements = [gamma_data[g] for g in gammas]
            ax.plot(gammas, improvements, 'o-', label=dataset, linewidth=2.5, 
                   markersize=8, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Значение gamma', fontsize=12, fontweight='bold')
        ax.set_ylabel('Улучшение ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_title('Зависимость улучшения ROC-AUC от значения gamma', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] График сохранен: {save_path}")
        plt.close()
    
    def create_all_plots(self):
        """Создание всех графиков"""
        print("\n[INFO] Создание графиков сравнения gamma...")
        
        try:
            self.create_gamma_comparison_metrics()
            self.create_gamma_improvement_heatmap()
            self.create_gamma_time_comparison()
            self.create_optimal_gamma_analysis()
            print("\n[OK] Все графики созданы успешно!")
        except Exception as e:
            print(f"\n[ERROR] Ошибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/gamma_experiments'
    plotter = GammaComparisonPlotter(results_dir)
    plotter.create_all_plots()

