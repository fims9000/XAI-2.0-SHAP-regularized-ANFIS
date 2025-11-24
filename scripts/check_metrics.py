#!/usr/bin/env python3
"""
Скрипт для проверки всех метрик после обучения
"""
import json
import pandas as pd
from pathlib import Path
import sys

def check_metrics(results_dir):
    """Проверка всех метрик из результатов"""
    results_dir = Path(results_dir)
    
    print("=" * 60)
    print("ПРОВЕРКА МЕТРИК ПОСЛЕ ОБУЧЕНИЯ")
    print("=" * 60)
    
    # 1. Проверка CSV файла с метриками
    csv_file = results_dir / 'comparison_results.csv'
    if csv_file.exists():
        print("\n📊 МЕТРИКИ ИЗ CSV:")
        print("-" * 60)
        df = pd.read_csv(csv_file)
        print(df.to_string(index=False))
        print()
    else:
        print("⚠️ Файл comparison_results.csv не найден")
    
    # 2. Проверка JSON файлов
    json_files = {
        'time_analysis': results_dir / 'time_analysis.json',
        'system_audit': results_dir / 'system_audit.json',
        'posthoc_parameters': results_dir / 'posthoc_parameters.json'
    }
    
    for name, file_path in json_files.items():
        if file_path.exists():
            print(f"\n📄 {name.upper().replace('_', ' ')}:")
            print("-" * 60)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"⚠️ Файл {name}.json не найден")
    
    # 3. Проверка графиков
    print("\n📈 ГРАФИКИ:")
    print("-" * 60)
    png_files = list(results_dir.glob('*.png'))
    if png_files:
        print(f"Найдено графиков: {len(png_files)}")
        for png_file in sorted(png_files):
            size_kb = png_file.stat().st_size / 1024
            print(f"  ✅ {png_file.name} ({size_kb:.1f} KB)")
    else:
        print("⚠️ Графики не найдены")
    
    # 4. Детальный анализ метрик
    if csv_file.exists():
        print("\n" + "=" * 60)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ МЕТРИК")
        print("=" * 60)
        
        df = pd.read_csv(csv_file)
        
        if 'Vanilla' in df['Method'].values and 'Regularized' in df['Method'].values:
            vanilla = df[df['Method'] == 'Vanilla'].iloc[0]
            regularized = df[df['Method'] == 'Regularized'].iloc[0]
            
            print("\n🔵 VANILLA ANFIS:")
            for col in df.columns:
                if col != 'Method':
                    print(f"  {col}: {vanilla[col]:.6f}")
            
            print("\n🟢 SHAP-РЕГУЛЯРИЗОВАННАЯ ANFIS:")
            for col in df.columns:
                if col != 'Method':
                    print(f"  {col}: {regularized[col]:.6f}")
            
            print("\n📊 УЛУЧШЕНИЯ:")
            print("-" * 60)
            for col in df.columns:
                if col != 'Method' and col != 'Training_Time':
                    improvement = regularized[col] - vanilla[col]
                    percent = (improvement / vanilla[col] * 100) if vanilla[col] != 0 else 0
                    symbol = "⬆️" if improvement > 0 else "⬇️" if improvement < 0 else "➡️"
                    print(f"  {col}: {improvement:+.6f} ({percent:+.2f}%) {symbol}")
            
            print("\n⏱️ ВРЕМЯ ОБУЧЕНИЯ:")
            print("-" * 60)
            vanilla_time = vanilla['Training_Time']
            regularized_time = regularized['Training_Time']
            speedup = vanilla_time / regularized_time if regularized_time > 0 else 0
            print(f"  Vanilla: {vanilla_time:.2f} сек")
            print(f"  Regularized: {regularized_time:.2f} сек")
            print(f"  Отношение: {speedup:.2f}x")
            
            # Проверка качества модели
            print("\n✅ ПРОВЕРКА КАЧЕСТВА:")
            print("-" * 60)
            
            # Accuracy
            if regularized['accuracy'] > vanilla['accuracy']:
                print(f"  ✅ Accuracy улучшена: {vanilla['accuracy']:.4f} → {regularized['accuracy']:.4f}")
            else:
                print(f"  ⚠️ Accuracy снизилась: {vanilla['accuracy']:.4f} → {regularized['accuracy']:.4f}")
            
            # ROC-AUC
            if regularized['roc_auc'] > vanilla['roc_auc']:
                print(f"  ✅ ROC-AUC улучшен: {vanilla['roc_auc']:.4f} → {regularized['roc_auc']:.4f}")
            else:
                print(f"  ⚠️ ROC-AUC снизился: {vanilla['roc_auc']:.4f} → {regularized['roc_auc']:.4f}")
            
            # F1-Score
            if regularized['f1_score'] > vanilla['f1_score']:
                print(f"  ✅ F1-Score улучшен: {vanilla['f1_score']:.4f} → {regularized['f1_score']:.4f}")
            else:
                print(f"  ⚠️ F1-Score снизился: {vanilla['f1_score']:.4f} → {regularized['f1_score']:.4f}")
            
            # Precision
            if regularized['precision'] > vanilla['precision']:
                print(f"  ✅ Precision улучшен: {vanilla['precision']:.4f} → {regularized['precision']:.4f}")
            else:
                print(f"  ⚠️ Precision снизился: {vanilla['precision']:.4f} → {regularized['precision']:.4f}")
            
            # Recall
            if regularized['recall'] > vanilla['recall']:
                print(f"  ✅ Recall улучшен: {vanilla['recall']:.4f} → {regularized['recall']:.4f}")
            else:
                print(f"  ⚠️ Recall снизился: {vanilla['recall']:.4f} → {regularized['recall']:.4f}")
    
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # По умолчанию используем последнюю директорию результатов
        results_base = Path(__file__).parent.parent / 'results'
        if results_base.exists():
            # Находим последнюю директорию
            dirs = [d for d in results_base.iterdir() if d.is_dir()]
            if dirs:
                results_dir = max(dirs, key=lambda x: x.stat().st_mtime)
                print(f"Используется последняя директория результатов: {results_dir.name}")
            else:
                print("Директории результатов не найдены")
                sys.exit(1)
        else:
            print("Директория results не найдена")
            sys.exit(1)
    
    check_metrics(results_dir)

