#!/usr/bin/env python3
"""
Скрипт для проверки прогресса обучения
"""
import sys
from pathlib import Path
import json
import glob

def check_progress():
    """Проверяет прогресс обучения"""
    datasets = ['breast_cancer', 'heart_disease', 'pima_diabetes', 'wine_quality']
    gammas = [0.1, 0.3, 0.5, 0.7]
    
    total_experiments = len(datasets) * len(gammas)
    completed = 0
    
    print("="*70)
    print("ПРОГРЕСС ОБУЧЕНИЯ")
    print("="*70)
    print()
    
    for dataset in datasets:
        gamma_dir = Path(f"results/{dataset}/gamma_experiments")
        summary_path = gamma_dir / "summary.json"
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                num_exp = len(summary.get('experiments', []))
                completed += num_exp
                
                status = "✅" if num_exp == 4 else "⏳"
                print(f"{status} {dataset}: {num_exp}/4 экспериментов")
                
                if num_exp > 0:
                    completed_gammas = [exp['gamma'] for exp in summary.get('experiments', [])]
                    print(f"   Завершено gamma: {sorted(completed_gammas)}")
            except Exception as e:
                print(f"⚠️  {dataset}: ошибка чтения - {e}")
        else:
            print(f"⏳ {dataset}: ожидание...")
    
    print()
    print(f"Прогресс: {completed}/{total_experiments} экспериментов ({completed*100//total_experiments}%)")
    print("="*70)
    
    return completed == total_experiments

if __name__ == "__main__":
    is_complete = check_progress()
    sys.exit(0 if is_complete else 1)

