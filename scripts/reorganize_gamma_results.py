"""
Скрипт для реорганизации результатов gamma экспериментов
Перемещает результаты в папки соответствующих датасетов
"""
import json
import shutil
from pathlib import Path
import pandas as pd

def reorganize_results():
    """Реорганизация результатов gamma экспериментов"""
    base_dir = Path('results')
    gamma_experiments_dir = base_dir / 'gamma_experiments'
    
    if not gamma_experiments_dir.exists():
        print("[ERROR] Папка gamma_experiments не найдена")
        return
    
    # Загружаем сводку
    summary_path = gamma_experiments_dir / 'summary.json'
    if not summary_path.exists():
        print("[ERROR] Файл summary.json не найден")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("=== РЕОРГАНИЗАЦИЯ РЕЗУЛЬТАТОВ ===\n")
    
    # Группируем по датасетам
    datasets_results = {}
    for exp in summary['experiments']:
        dataset = exp['dataset']
        if dataset not in datasets_results:
            datasets_results[dataset] = []
        datasets_results[dataset].append(exp)
    
    # Перемещаем результаты для каждого датасета
    for dataset, experiments in datasets_results.items():
        dataset_dir = base_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Обработка {dataset}...")
        
        # Создаем папку для gamma экспериментов внутри датасета
        gamma_dir = dataset_dir / 'gamma_experiments'
        gamma_dir.mkdir(exist_ok=True)
        
        # Сохраняем результаты для этого датасета
        dataset_summary = {
            'dataset': dataset,
            'experiments': experiments,
            'total_experiments': len(experiments),
            'timestamp': summary.get('timestamp', '')
        }
        
        with open(gamma_dir / 'summary.json', 'w') as f:
            json.dump(dataset_summary, f, indent=2, default=str)
        
        # Перемещаем отдельные результаты экспериментов
        for exp in experiments:
            gamma_value = exp['gamma']
            source_dir = base_dir / f"{dataset}_gamma_{gamma_value}"
            
            if source_dir.exists():
                # Перемещаем в подпапку
                target_dir = gamma_dir / f"gamma_{gamma_value}"
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(source_dir), str(target_dir))
                print(f"  ✓ Перемещен gamma={gamma_value}")
        
        # Создаем CSV сводку для датасета
        dataset_data = []
        for exp in experiments:
            row = {
                'gamma': exp['gamma'],
                'vanilla_accuracy': exp['vanilla']['metrics'].get('accuracy', 0),
                'regularized_accuracy': exp['regularized']['metrics'].get('accuracy', 0),
                'vanilla_roc_auc': exp['vanilla']['metrics'].get('roc_auc', 0),
                'regularized_roc_auc': exp['regularized']['metrics'].get('roc_auc', 0),
                'vanilla_time': exp['vanilla']['training_time'],
                'regularized_time': exp['regularized']['training_time'],
                'improvement_roc_auc': exp['regularized']['metrics'].get('roc_auc', 0) - exp['vanilla']['metrics'].get('roc_auc', 0),
                'improvement_accuracy': exp['regularized']['metrics'].get('accuracy', 0) - exp['vanilla']['metrics'].get('accuracy', 0)
            }
            dataset_data.append(row)
        
        df = pd.DataFrame(dataset_data)
        df.to_csv(gamma_dir / 'comparison.csv', index=False)
        print(f"  ✓ Создан comparison.csv")
    
    print("\n✅ Реорганизация завершена!")
    print("\nСтруктура результатов:")
    for dataset in datasets_results.keys():
        print(f"  results/{dataset}/gamma_experiments/")

if __name__ == '__main__':
    reorganize_results()

