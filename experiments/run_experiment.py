"""
Главный скрипт для запуска экспериментов
"""
import sys
import os
from pathlib import Path
import argparse
import yaml

# Добавляем src в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from data.loader import DataLoader
from models.anfis_manager import ANFISManager
from models.shap_trainer import ShapAwareANFISTrainer
from analysis.shap_analyzer import PostHocSHAPAnalyzer


def main():
    parser = argparse.ArgumentParser(description='ANFIS+SHAP Experiments')
    parser.add_argument('--dataset', default='breast_cancer',
                        help='Название датасета')
    parser.add_argument('--experiment',
                        choices=['vanilla', 'posthoc', 'regularized', 'all'],
                        default='all',
                        help='Какой эксперимент запустить')
    parser.add_argument('--save-results', action='store_true',
                        help='Сохранить результаты')

    args = parser.parse_args()

    print(f"🚀 Запуск экспериментов для: {args.dataset}")
    print("=" * 60)

    # Загрузка конфигурации
    config_path = project_root / 'configs' / f'{args.dataset}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Создание папки для результатов
    results_dir = project_root / 'results' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка и подготовка данных
    data_loader = DataLoader(config_path)
    data = data_loader.load_and_prepare_data()

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # Менеджер ANFIS
    anfis_manager = ANFISManager(config)

    results = {}

    # Vanilla ANFIS
    if args.experiment in ['vanilla', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 1: Vanilla ANFIS")
        print("=" * 40)
        results['vanilla'] = anfis_manager.train_vanilla_model(
            X_train, y_train, X_test, y_test
        )

    # Post-hoc SHAP
    if args.experiment in ['posthoc', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 2: Post-hoc SHAP")
        print("=" * 40)

        if 'vanilla' not in results:
            results['vanilla'] = anfis_manager.train_vanilla_model(
                X_train, y_train, X_test, y_test
            )

        shap_analyzer = PostHocSHAPAnalyzer(results['vanilla']['model'], config)
        results['posthoc'] = shap_analyzer.analyze(X_test, feature_names)

    # SHAP Regularization
    if args.experiment in ['regularized', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 3: SHAP Regularization")
        print("=" * 40)

        # Создание базовой модели
        base_model = anfis_manager.create_model(verbose=False)
        y_init = y_train.values[:50] if hasattr(y_train, 'values') else y_train.iloc[:50]
        base_model.fit(X_train[:50], y_init)

        # Обучение с SHAP-регуляризацией
        gamma = config['shap']['gamma']
        epochs = config['shap']['training_epochs']

        trainer = ShapAwareANFISTrainer(base_model,config, gamma=gamma, verbose=True)
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        history = trainer.fit(X_train, y_train_values, epochs=epochs)

        # Оценка результатов
        y_pred = trainer.predict(X_test)
        import numpy as np
        from sklearn.metrics import (
            confusion_matrix,
            mean_squared_error, mean_absolute_error, r2_score)
        from sklearn.metrics import precision_score,accuracy_score, recall_score,roc_auc_score, f1_score

        feature_importance = trainer.get_global_shap_importance(X_test)


        if config['dataset']['task_type'] == 'regression':
            # Регрессионные метрики
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

            results['regularized'] = {
                'trainer': trainer,
                'probabilities': y_pred,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': trainer.training_time,
                'history': history}
        else:
            # Классификационные метрики
            y_pred_bin = (y_pred > 0.5).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_bin),
                'precision': precision_score(y_test, y_pred_bin),
                'recall': recall_score(y_test, y_pred_bin),
                'f1_score': f1_score(y_test, y_pred_bin),
                'roc_auc': roc_auc_score(y_test, y_pred)
                }

            results['regularized'] = {
                'trainer': trainer,
                'predictions': y_pred_bin,
                'probabilities': y_pred,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': trainer.training_time,
                'history': history
            }

        print(f"✅ SHAP-регуляризация завершена:")
        for metric_name, metric_value in metrics.items():
            print(f"   📊 {metric_name}: {metric_value:.4f}")
        print(f"   ⏱️ Время: {trainer.training_time:.2f} сек")

    # Сравнительный анализ
    if len(results) > 1:
        print("\n" + "=" * 40)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print("=" * 40)

        for exp_name, exp_results in results.items():
            if 'metrics' in exp_results:
                print(f"\n{exp_name.upper()}:")
                for metric, value in exp_results['metrics'].items():
                    print(f"  {metric}: {value:.4f}")

    print("\n🎉 Все эксперименты завершены успешно!")

    # Визуализация результатов
    if results:
        print("\n" + "=" * 40)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
        print("=" * 40)

        # Импортируем визуализатор
        from visualization.visualizer import ResultsVisualizer

        visualizer = ResultsVisualizer(config, save_dir=results_dir if args.save_results else None)
        visualizer.create_comprehensive_analysis(results, feature_names, X_test, y_test)

    if args.save_results:
        print(f"📁 Результаты сохранены в: {results_dir}")


if __name__ == "__main__":
    main()
