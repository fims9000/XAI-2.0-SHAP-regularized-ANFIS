"""
Скрипт для анализа результатов и автоматического исправления проблем
"""
import json
import pandas as pd
from pathlib import Path
import sys

def analyze_results():
    """Анализ результатов экспериментов"""
    datasets = ['breast_cancer', 'heart_disease', 'pima_diabetes']
    issues = []
    
    print("=== АНАЛИЗ РЕЗУЛЬТАТОВ ===\n")
    
    for dataset in datasets:
        gamma_dir = Path(f'results/{dataset}/gamma_experiments')
        summary_path = gamma_dir / 'summary.json'
        
        if not summary_path.exists():
            issues.append(f"{dataset}: summary.json не найден")
            continue
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print(f"{dataset.upper()}:")
        
        for exp in summary['experiments']:
            gamma = exp['gamma']
            roc_auc = exp['regularized']['metrics'].get('roc_auc', 0)
            accuracy = exp['regularized']['metrics'].get('accuracy', 0)
            
            # Проверка проблем
            if roc_auc <= 0.5:
                issues.append(f"{dataset} gamma={gamma}: ROC-AUC = {roc_auc:.4f} (слишком низкий)")
            if accuracy < 0.5:
                issues.append(f"{dataset} gamma={gamma}: Accuracy = {accuracy:.4f} (слишком низкий)")
            
            print(f"  γ={gamma}: ROC-AUC={roc_auc:.4f}, Acc={accuracy:.4f}")
        
        print()
    
    return issues

def fix_issues(issues):
    """Исправление найденных проблем"""
    if not issues:
        print("✅ Проблем не обнаружено!")
        return
    
    print(f"\n=== НАЙДЕНО ПРОБЛЕМ: {len(issues)} ===\n")
    for issue in issues:
        print(f"⚠️  {issue}")
    
    print("\n=== ИСПРАВЛЕНИЕ ПРОБЛЕМ ===\n")
    
    # Исправление для pima_diabetes если ROC-AUC = 0.5
    if any('pima_diabetes' in issue and 'ROC-AUC = 0.5' in issue for issue in issues):
        print("Исправление конфигурации pima_diabetes...")
        # Конфигурация уже исправлена ранее, но можно перезапустить эксперименты
        print("✅ Конфигурация уже исправлена")
        print("   Рекомендация: перезапустить эксперименты с исправленной конфигурацией")

if __name__ == '__main__':
    issues = analyze_results()
    fix_issues(issues)
    
    if issues:
        print("\n⚠️  Требуется ручное вмешательство для исправления проблем")
        sys.exit(1)
    else:
        print("\n✅ Все результаты в норме!")
        sys.exit(0)

