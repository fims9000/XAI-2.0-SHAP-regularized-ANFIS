#!/usr/bin/env python3
"""
Скрипт для анализа финальных результатов обучения
"""
import sys
from pathlib import Path
import json
import pandas as pd

def analyze_results():
    """Анализирует финальные результаты"""
    datasets = ['breast_cancer', 'heart_disease', 'pima_diabetes', 'wine_quality']
    
    print("="*70)
    print("АНАЛИЗ ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ")
    print("="*70)
    print()
    
    all_results = []
    
    for dataset in datasets:
        gamma_dir = Path(f"results/{dataset}/gamma_experiments")
        summary_path = gamma_dir / "summary.json"
        
        if not summary_path.exists():
            print(f"❌ {dataset}: результаты не найдены")
            continue
        
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            experiments = summary.get('experiments', [])
            
            if len(experiments) != 4:
                print(f"⚠️  {dataset}: неполные результаты ({len(experiments)}/4)")
                continue
            
            print(f"{dataset.upper()}:")
            print("-" * 70)
            
            # Определяем тип задачи
            is_classification = 'roc_auc' in experiments[0]['regularized']['metrics']
            
            if is_classification:
                # Классификация
                print("Метрики классификации:")
                print()
                
                roc_values = []
                acc_values = []
                f1_values = []
                
                for exp in experiments:
                    gamma = exp['gamma']
                    metrics = exp['regularized']['metrics']
                    
                    roc = metrics.get('roc_auc', 0)
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_score', 0)
                    
                    roc_values.append((gamma, roc))
                    acc_values.append((gamma, acc))
                    f1_values.append((gamma, f1))
                    
                    print(f"  Gamma {gamma}:")
                    print(f"    ROC-AUC: {roc:.4f}")
                    print(f"    Accuracy: {acc:.4f}")
                    print(f"    F1-Score: {f1:.4f}")
                    print()
                
                # Находим оптимальное gamma
                best_gamma_roc = max(roc_values, key=lambda x: x[1])
                best_gamma_acc = max(acc_values, key=lambda x: x[1])
                best_gamma_f1 = max(f1_values, key=lambda x: x[1])
                
                print(f"Оптимальное gamma:")
                print(f"  По ROC-AUC: {best_gamma_roc[0]} (ROC-AUC={best_gamma_roc[1]:.4f})")
                print(f"  По Accuracy: {best_gamma_acc[0]} (Accuracy={best_gamma_acc[1]:.4f})")
                print(f"  По F1-Score: {best_gamma_f1[0]} (F1={best_gamma_f1[1]:.4f})")
                
                roc_range = max([r[1] for r in roc_values]) - min([r[1] for r in roc_values])
                print(f"  Диапазон ROC-AUC: {roc_range:.4f}")
                
                all_results.append({
                    'dataset': dataset,
                    'task': 'classification',
                    'best_gamma': best_gamma_roc[0],
                    'best_roc_auc': best_gamma_roc[1],
                    'roc_range': roc_range,
                    'best_accuracy': best_gamma_acc[1],
                    'best_f1': best_gamma_f1[1]
                })
            else:
                # Регрессия
                print("Метрики регрессии:")
                print()
                
                rmse_values = []
                r2_values = []
                mae_values = []
                
                for exp in experiments:
                    gamma = exp['gamma']
                    metrics = exp['regularized']['metrics']
                    
                    rmse = metrics.get('rmse', 0)
                    r2 = metrics.get('r2', 0)
                    mae = metrics.get('mae', 0)
                    
                    rmse_values.append((gamma, rmse))
                    r2_values.append((gamma, r2))
                    mae_values.append((gamma, mae))
                    
                    print(f"  Gamma {gamma}:")
                    print(f"    RMSE: {rmse:.4f}")
                    print(f"    R²: {r2:.4f}")
                    print(f"    MAE: {mae:.4f}")
                    print()
                
                # Находим оптимальное gamma (минимальный RMSE, максимальный R²)
                best_gamma_rmse = min(rmse_values, key=lambda x: x[1])
                best_gamma_r2 = max(r2_values, key=lambda x: x[1])
                
                print(f"Оптимальное gamma:")
                print(f"  По RMSE: {best_gamma_rmse[0]} (RMSE={best_gamma_rmse[1]:.4f})")
                print(f"  По R²: {best_gamma_r2[0]} (R²={best_gamma_r2[1]:.4f})")
                
                r2_range = max([r[1] for r in r2_values]) - min([r[1] for r in r2_values])
                print(f"  Диапазон R²: {r2_range:.4f}")
                
                all_results.append({
                    'dataset': dataset,
                    'task': 'regression',
                    'best_gamma': best_gamma_r2[0],
                    'best_r2': best_gamma_r2[1],
                    'r2_range': r2_range,
                    'best_rmse': best_gamma_rmse[1]
                })
            
            print()
            
        except Exception as e:
            print(f"❌ {dataset}: ошибка анализа - {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговая сводка
    print("="*70)
    print("ИТОГОВАЯ СВОДКА")
    print("="*70)
    print()
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        print()
        
        print("Выводы:")
        print("-" * 70)
        
        classification_results = [r for r in all_results if r['task'] == 'classification']
        regression_results = [r for r in all_results if r['task'] == 'regression']
        
        if classification_results:
            print("Классификация:")
            avg_roc = sum([r['best_roc_auc'] for r in classification_results]) / len(classification_results)
            avg_acc = sum([r['best_accuracy'] for r in classification_results]) / len(classification_results)
            print(f"  Средний лучший ROC-AUC: {avg_roc:.4f}")
            print(f"  Средняя лучшая Accuracy: {avg_acc:.4f}")
            print()
        
        if regression_results:
            print("Регрессия:")
            avg_r2 = sum([r['best_r2'] for r in regression_results]) / len(regression_results)
            avg_rmse = sum([r['best_rmse'] for r in regression_results]) / len(regression_results)
            print(f"  Средний лучший R²: {avg_r2:.4f}")
            print(f"  Средний лучший RMSE: {avg_rmse:.4f}")
            print()
        
        # Анализ влияния gamma
        print("Влияние gamma:")
        high_range_datasets = [r for r in all_results if (r.get('roc_range', 0) > 0.05 or r.get('r2_range', 0) > 0.1)]
        if high_range_datasets:
            print(f"  Сильное влияние на {len(high_range_datasets)} датасетах:")
            for r in high_range_datasets:
                if r['task'] == 'classification':
                    print(f"    - {r['dataset']}: диапазон ROC-AUC = {r['roc_range']:.4f}")
                else:
                    print(f"    - {r['dataset']}: диапазон R² = {r['r2_range']:.4f}")
        else:
            print("  Умеренное влияние на всех датасетах")
    
    print("="*70)

if __name__ == "__main__":
    analyze_results()

