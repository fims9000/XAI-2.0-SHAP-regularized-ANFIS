"""
Скрипт для экспериментов с разными значениями gamma
"""
import argparse
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer import ShapAwareANFISTrainer
from src.analysis.shap_analyzer import PostHocSHAPAnalyzer
from src.utils.method_labels import METHOD_LABEL_INLINE, METHOD_MODEL_LABEL_RU
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def run_experiment_with_gamma(dataset_name, gamma_value, config_path=None):
    """Запуск эксперимента с заданным значением gamma"""
    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {dataset_name} с gamma={gamma_value}")
    print(f"{'='*60}")
    
    # Загрузка конфигурации
    if config_path is None:
        config_path = f"configs/{dataset_name}.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    # Временно сохраняем обновленную конфигурацию
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Обновляем gamma в конфигурации
    config['shap']['gamma'] = gamma_value
    
    # Сохраняем временную конфигурацию
    temp_config_path = Path(f"configs/{dataset_name}_temp_gamma_{gamma_value}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Создаем директорию для результатов
    results_dir = Path(f"results/{dataset_name}_gamma_{gamma_value}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных (DataLoader ожидает путь к файлу, а не словарь)
    try:
        loader = DataLoader(str(temp_config_path))
        data_dict = loader.load_and_prepare_data()
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
    except Exception as e:
        # Удаляем временный файл при ошибке
        if temp_config_path.exists():
            temp_config_path.unlink()
        print(f"[ERROR] Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Удаляем временный файл конфигурации
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    task_type = config['dataset']['task_type']
    
    # 1. Vanilla ANFIS
    print(f"\n[1/3] Обучение Vanilla ANFIS...")
    manager = ANFISManager(config)
    vanilla_results = manager.train_vanilla_model(X_train, y_train, X_test, y_test)
    
    # 2. ANFIS с Baseline-Masked Feature Sensitivity Regularization
    print(f"\n[2/3] Обучение {METHOD_MODEL_LABEL_RU} (gamma={gamma_value})...")
    base_model = manager.create_model(verbose=False)
    
    # Предобучение
    init_size = min(100, len(X_train) // 2)
    X_init = X_train[:init_size]
    y_init = y_train[:init_size] if hasattr(y_train, 'iloc') else y_train[:init_size]
    y_init_values = y_init.values if hasattr(y_init, 'values') else y_init
    base_model.fit(X_init, y_init_values)
    
    # Baseline-Masked Feature Sensitivity Regularization
    trainer = ShapAwareANFISTrainer(base_model, config, gamma=gamma_value, verbose=True)
    X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    
    # Валидационная выборка (берем часть тестовой)
    val_size = min(len(X_test) // 4, 50)
    X_val_values = X_test.values if hasattr(X_test, 'values') else X_test
    y_val_values = y_test.values if hasattr(y_test, 'values') else y_test
    if len(X_val_values) > val_size:
        X_val_values = X_val_values[:val_size]
        y_val_values = y_val_values[:val_size]
    
    history = trainer.fit(
        X_train_values, y_train_values,
        X_val=X_val_values, y_val=y_val_values,
        epochs=config['shap']['training_epochs'],
        batch_size=config['shap']['batch_size'],
        lr=config['shap']['learning_rate']
    )
    
    # Предсказания
    X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    y_pred = trainer.predict(X_test_values)
    y_pred_bin = (y_pred > 0.5).astype(int) if task_type == 'classification' else y_pred
    
    # Метрики
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test_values, y_pred_bin),
            'precision': precision_score(y_test_values, y_pred_bin, zero_division=0),
            'recall': recall_score(y_test_values, y_pred_bin, zero_division=0),
            'f1_score': f1_score(y_test_values, y_pred_bin, zero_division=0),
            'roc_auc': roc_auc_score(y_test_values, y_pred) if len(np.unique(y_test_values)) > 1 else 0.0
        }
    else:
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_values, y_pred)),
            'mae': mean_absolute_error(y_test_values, y_pred),
            'r2': r2_score(y_test_values, y_pred)
        }
    
    feature_importance = trainer.get_global_shap_importance(X_test_values)
    
    regularized_results = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'training_time': trainer.training_time,
        'history': history
    }
    
    # 3. Post-hoc SHAP (опционально, для сравнения - пропускаем для скорости)
    print(f"\n[3/3] Post-hoc SHAP анализ пропущен (для ускорения экспериментов)...")
    posthoc_results = {'analysis_time': 0}
    
    # Сохранение результатов
    results = {
        'dataset': dataset_name,
        'gamma': gamma_value,
        'vanilla': {
            'metrics': vanilla_results['metrics'],
            'training_time': vanilla_results['training_time']
        },
        'regularized': {
            'metrics': regularized_results['metrics'],
            'training_time': regularized_results['training_time']
        },
        'posthoc': {
            'analysis_time': posthoc_results.get('analysis_time', 0)
        }
    }
    
    # Сохранение в JSON
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Сохранение в CSV
    comparison_data = {
        'Method': ['Vanilla', 'Regularized'],
        'gamma': [0, gamma_value],
        'training_time': [
            vanilla_results['training_time'],
            regularized_results['training_time']
        ]
    }
    
    if task_type == 'classification':
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            comparison_data[metric] = [
                vanilla_results['metrics'][metric],
                regularized_results['metrics'][metric]
            ]
    else:
        for metric in ['rmse', 'mae', 'r2']:
            comparison_data[metric] = [
                vanilla_results['metrics'][metric],
                regularized_results['metrics'][metric]
            ]
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(results_dir / 'comparison_results.csv', index=False)
    
    print(f"\n[OK] Результаты сохранены в {results_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Эксперименты с разными значениями gamma')
    parser.add_argument('--datasets', nargs='+', 
                       default=['breast_cancer', 'heart_disease', 'pima_diabetes'],
                       help='Список датасетов для экспериментов')
    parser.add_argument('--gammas', nargs='+', type=float,
                       default=[0.1, 0.3, 0.5, 0.7],
                       help='Список значений gamma для экспериментов')
    parser.add_argument('--output-dir', type=str, default='results/gamma_experiments',
                       help='Директория для сохранения сводных результатов')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"НАЧАЛО ЭКСПЕРИМЕНТОВ")
    print(f"Датасеты: {', '.join(args.datasets)}")
    print(f"Значения gamma: {', '.join(map(str, args.gammas))}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for dataset in args.datasets:
        for gamma in args.gammas:
            try:
                results = run_experiment_with_gamma(dataset, gamma)
                results['experiment_time'] = time.time() - start_time
                all_results.append(results)
            except Exception as e:
                print(f"\n[ERROR] Ошибка в эксперименте {dataset} gamma={gamma}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Сохранение сводных результатов
    summary = {
        'experiments': all_results,
        'total_time': time.time() - start_time,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Создание сводной таблицы
    summary_data = []
    for r in all_results:
        row = {
            'dataset': r['dataset'],
            'gamma': r['gamma'],
            'vanilla_accuracy': r['vanilla']['metrics'].get('accuracy', np.nan),
            'regularized_accuracy': r['regularized']['metrics'].get('accuracy', np.nan),
            'vanilla_roc_auc': r['vanilla']['metrics'].get('roc_auc', np.nan),
            'regularized_roc_auc': r['regularized']['metrics'].get('roc_auc', np.nan),
            'vanilla_time': r['vanilla']['training_time'],
            'regularized_time': r['regularized']['training_time'],
            'improvement_roc_auc': r['regularized']['metrics'].get('roc_auc', 0) - r['vanilla']['metrics'].get('roc_auc', 0),
            'speedup': r['vanilla']['training_time'] / r['regularized']['training_time'] if r['regularized']['training_time'] > 0 else np.nan
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"Общее время: {time.time() - start_time:.2f} сек")
    print(f"Результаты сохранены в {output_dir}")
    print(f"{'='*60}")
    
    # Вывод сводной таблицы
    print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()

