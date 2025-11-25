"""
Главный скрипт для запуска экспериментов
"""
import sys
import os
from pathlib import Path
import argparse
import yaml
import json
import numpy as np

# Добавляем src в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from data.loader import DataLoader
from models.anfis_manager import ANFISManager
from models.shap_trainer import ShapAwareANFISTrainer
from models.anfis_rules import ANFISRulesExtractor
from analysis.shap_analyzer import PostHocSHAPAnalyzer
from utils.system_audit import SystemAuditor
from utils.method_labels import METHOD_LABEL_INLINE, METHOD_MODEL_LABEL_RU


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
    parser.add_argument('--mode',
                        choices=['fast', 'article'],
                        default='fast',
                        help='Режим конфигурации: fast (по умолчанию) или article')

    args = parser.parse_args()

    print(f"Запуск экспериментов для: {args.dataset}")
    print(f"Режим конфигурации: {args.mode}")
    print("=" * 60)

    # Аудит системы
    print("\n" + "=" * 40)
    print("АУДИТ СИСТЕМЫ")
    print("=" * 40)
    auditor = SystemAuditor()
    audit_data = auditor.run_full_audit()
    auditor.print_audit_report()

    # Загрузка конфигурации
    config_name = args.dataset if args.mode == 'fast' else f"{args.dataset}_{args.mode}"
    config_path = project_root / 'configs' / f'{config_name}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(
            f"Конфигурация '{config_name}.yaml' не найдена в папке configs. "
            f"Создайте её или запустите с --mode fast."
        )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Создание папки для результатов
    if args.mode == 'fast':
        results_dir = project_root / 'results' / args.dataset
    else:
        results_dir = project_root / 'results' / args.dataset / args.mode
    results_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение аудита системы
    if args.save_results:
        audit_file = results_dir / 'system_audit.json'
        auditor.save_audit_report(audit_file)

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
    posthoc_params = {}

    # Vanilla ANFIS
    if args.experiment in ['vanilla', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 1: Vanilla ANFIS")
        print("=" * 40)
        results['vanilla'] = anfis_manager.train_vanilla_model(
            X_train, y_train, X_test, y_test
        )

    # Post-hoc SHAP для Vanilla ANFIS
    if args.experiment in ['posthoc', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 2: Post-hoc SHAP (Vanilla ANFIS)")
        print("=" * 40)

        if 'vanilla' not in results:
            results['vanilla'] = anfis_manager.train_vanilla_model(
                X_train, y_train, X_test, y_test
            )

        shap_analyzer = PostHocSHAPAnalyzer(results['vanilla']['model'], config)
        
        # Сохранение параметров post-hoc
        posthoc_params['vanilla'] = {
            'sample_size': config['shap']['sample_size'],
            'explainer_type': 'KernelExplainer',
            'background_size': config['shap']['sample_size']
        }
        
        results['posthoc_vanilla'] = shap_analyzer.analyze(X_test, feature_names)
        posthoc_params['vanilla']['analysis_time'] = results['posthoc_vanilla']['analysis_time']

    # Baseline-Masked Feature Sensitivity Regularization
    if args.experiment in ['regularized', 'all']:
        print("\n" + "=" * 40)
        print("ЭКСПЕРИМЕНТ 3: Baseline-Masked Feature Sensitivity Regularization")
        print("=" * 40)

        # Создание базовой модели
        # Используем больше данных для инициализации при полном обучении
        init_size = min(100, len(X_train) // 2)  # Используем до 100 образцов или половину данных
        base_model = anfis_manager.create_model(verbose=True)
        y_init = y_train.values[:init_size] if hasattr(y_train, 'values') else y_train.iloc[:init_size]
        base_model.fit(X_train[:init_size], y_init)

        # Обучение с Baseline-Masked Feature Sensitivity Regularization
        gamma = config['shap']['gamma']
        epochs = config['shap']['training_epochs']
        batch_size = config['shap']['batch_size']
        lr = config['shap']['learning_rate']

        trainer = ShapAwareANFISTrainer(base_model, config, gamma=gamma, verbose=True)
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Разделение на train/val для отслеживания метрик
        from sklearn.model_selection import train_test_split
        X_train_shap, X_val_shap, y_train_shap, y_val_shap = train_test_split(
            X_train, y_train_values, test_size=0.2, random_state=42, 
            stratify=y_train_values if config['dataset']['task_type'] == 'classification' else None
        )
        
        history = trainer.fit(X_train_shap, y_train_shap, X_val_shap, y_val_shap, 
                             epochs=epochs, batch_size=batch_size, lr=lr)

        # Оценка результатов
        y_pred = trainer.predict(X_test)
        from sklearn.metrics import (
            confusion_matrix,
            mean_squared_error, mean_absolute_error, r2_score)
        from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, f1_score

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
                'model': base_model,  # base_model.network обновлен через trainer.model
                'predictions': y_pred,  # Для регрессии predictions = y_pred
                'probabilities': y_pred,  # Для совместимости
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': trainer.training_time,
                'history': history
            }
        else:
            # Классификационные метрики
            y_pred_bin = (y_pred > 0.5).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_bin),
                'precision': precision_score(y_test, y_pred_bin, zero_division=0),
                'recall': recall_score(y_test, y_pred_bin, zero_division=0),
                'f1_score': f1_score(y_test, y_pred_bin, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
                }

            results['regularized'] = {
                'trainer': trainer,
                'model': base_model,  # base_model.network обновлен через trainer.model
                'predictions': y_pred_bin,
                'probabilities': y_pred,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': trainer.training_time,
                'history': history
            }

        print(f"[OK] {METHOD_LABEL_INLINE} завершена:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        print(f"   Время: {trainer.training_time:.2f} сек")

        # Post-hoc SHAP для модели с Baseline-Masked Feature Sensitivity Regularization
        print("\n" + "=" * 40)
        print(f"ЭКСПЕРИМЕНТ 4: Post-hoc SHAP ({METHOD_LABEL_INLINE})")
        print("=" * 40)
        
        # Создаем обертку модели для post-hoc анализа
        class ModelWrapper:
            def __init__(self, trainer, task_type):
                self.trainer = trainer
                self.network = trainer.model
                self.task_type = task_type
                
            def predict(self, X):
                return self.trainer.predict(X)
                
            def predict_proba(self, X):
                pred = self.predict(X)
                if self.task_type == 'classification':
                    # Для бинарной классификации возвращаем вероятности
                    pred = np.clip(pred, 0, 1)
                    return np.column_stack([1 - pred, pred])
                else:
                    return pred
        
        wrapped_model = ModelWrapper(trainer, config['dataset']['task_type'])
        shap_analyzer_reg = PostHocSHAPAnalyzer(wrapped_model, config)
        
        # Сохранение параметров post-hoc для регуляризованной модели
        posthoc_params['regularized'] = {
            'sample_size': config['shap']['sample_size'],
            'explainer_type': 'KernelExplainer',
            'background_size': config['shap']['sample_size']
        }
        
        results['posthoc_regularized'] = shap_analyzer_reg.analyze(X_test, feature_names)
        posthoc_params['regularized']['analysis_time'] = results['posthoc_regularized']['analysis_time']

        # Сохранение параметров post-hoc
        if args.save_results:
            posthoc_params_file = results_dir / 'posthoc_parameters.json'
            with open(posthoc_params_file, 'w', encoding='utf-8') as f:
                json.dump(posthoc_params, f, indent=2, ensure_ascii=False)
            print(f"[OK] Параметры post-hoc сохранены: {posthoc_params_file}")

    # Извлечение и сравнение правил
    if 'vanilla' in results and 'regularized' in results:
        print("\n" + "=" * 40)
        print("СРАВНЕНИЕ ПРАВИЛ ANFIS")
        print("=" * 40)
        
        try:
            vanilla_extractor = ANFISRulesExtractor(results['vanilla']['model'], feature_names)
            regularized_extractor = ANFISRulesExtractor(results['regularized']['model'], feature_names)
            
            vanilla_rules = vanilla_extractor.extract_rules()
            regularized_rules = regularized_extractor.extract_rules()
            
            rules_comparison = vanilla_extractor.compare_rules(
                vanilla_rules, regularized_rules,
                model1_name="Vanilla ANFIS",
                model2_name=METHOD_MODEL_LABEL_RU
            )
            
            results['rules_comparison'] = rules_comparison
            
            print(f"[OK] Правила извлечены и сравнены:")
            print(f"   Среднее сходство: {rules_comparison['average_similarity']:.4f}")
            print(f"   Среднее различие коэффициентов: {rules_comparison['average_coeff_diff']:.4f}")
            print(f"   Среднее различие ФП: {rules_comparison['average_mf_diff']:.4f}")
            
        except Exception as e:
            print(f"[WARNING] Ошибка при извлечении правил: {e}")

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
                if 'training_time' in exp_results:
                    print(f"  Время обучения: {exp_results['training_time']:.2f} сек")
                if 'analysis_time' in exp_results:
                    print(f"  Время анализа: {exp_results['analysis_time']:.2f} сек")

    print("\n[OK] Все эксперименты завершены успешно!")

    # Визуализация результатов
    if results:
        print("\n" + "=" * 40)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
        print("=" * 40)

        # Импортируем визуализатор
        from visualization.visualizer import ResultsVisualizer
        from visualization.individual_plots import IndividualPlotCreator

        visualizer = ResultsVisualizer(config, save_dir=results_dir if args.save_results else None)
        
        # Создание отдельных графиков высокого качества
        if args.save_results:
            print("\n" + "=" * 40)
            print("СОЗДАНИЕ ОТДЕЛЬНЫХ ГРАФИКОВ ВЫСОКОГО КАЧЕСТВА")
            print("=" * 40)
            individual_plotter = IndividualPlotCreator(config, save_dir=results_dir)
            
            # Отдельные графики для Vanilla модели
            if 'vanilla' in results:
                individual_plotter.create_all_vanilla_plots(
                    results['vanilla'], feature_names, X_test, y_test
                )
        
        # Основная визуализация (сохраняет объединенные графики для совместимости)
        visualizer.create_comprehensive_analysis(results, feature_names, X_test, y_test)
        
        # Графики обучения для Baseline-Masked Feature Sensitivity Regularization
        if 'regularized' in results and 'history' in results['regularized']:
            print("\nСоздание графиков обучения...")
            visualizer.create_training_plots(
                results['regularized']['history'],
                save_dir=results_dir if args.save_results else None
            )
        
        # Сравнение правил
        if 'rules_comparison' in results:
            print("\nСоздание графиков сравнения правил...")
            visualizer.create_rules_comparison(
                results['rules_comparison'],
                save_dir=results_dir if args.save_results else None
            )
        
        # Сравнение post-hoc SHAP и embedded SHAP
        if 'posthoc_vanilla' in results and 'regularized' in results:
            print("\nСоздание графиков сравнения SHAP методов...")
            posthoc_shap = results['posthoc_vanilla']['global_importance']
            embedded_shap = results['regularized']['feature_importance']
            visualizer.create_shap_comparison(
                posthoc_shap, embedded_shap, feature_names,
                save_dir=results_dir if args.save_results else None
            )
        
        # Сравнение post-hoc для vanilla и regularized
        if 'posthoc_vanilla' in results and 'posthoc_regularized' in results:
            print("\nСоздание графиков сравнения post-hoc SHAP...")
            posthoc_vanilla_shap = results['posthoc_vanilla']['global_importance']
            posthoc_reg_shap = results['posthoc_regularized']['global_importance']
            visualizer.create_shap_comparison(
                posthoc_vanilla_shap, posthoc_reg_shap, feature_names,
                save_dir=results_dir if args.save_results else None
            )
        
        # Создание отдельных качественных графиков для презентации
        print("\n" + "=" * 40)
        print("СОЗДАНИЕ ОТДЕЛЬНЫХ ГРАФИКОВ ДЛЯ ПРЕЗЕНТАЦИИ")
        print("=" * 40)
        
        from visualization.presentation_plots import PresentationPlotter
        
        presentation_plotter = PresentationPlotter(config, save_dir=results_dir if args.save_results else None)
        
        # 1. График сравнения метрик обучения
        if 'regularized' in results and 'history' in results['regularized']:
            print("\n[INFO] Создание графика сравнения метрик обучения...")
            # Создаем фиктивную историю для vanilla (только финальные метрики)
            vanilla_history = {}
            if 'vanilla' in results:
                vanilla_metrics = results['vanilla']['metrics']
                # Создаем историю с одной точкой (финальные метрики)
                vanilla_history = {
                    'train_metrics': [vanilla_metrics],
                    'val_metrics': [None],
                    'epoch_times': [results['vanilla']['training_time']]
                }
            
            regularized_history = results['regularized']['history']
            presentation_plotter.create_training_metrics_comparison(
                vanilla_history, regularized_history,
                save_name='01_training_metrics_comparison.png'
            )
        
        # 2. График сравнения предсказаний и post-hoc важности
        if ('vanilla' in results and 'regularized' in results and 
            'posthoc_vanilla' in results and 'posthoc_regularized' in results):
            print("[INFO] Создание графика сравнения предсказаний и важности признаков...")
            vanilla_pred = results['vanilla'].get('probabilities', results['vanilla'].get('predictions', []))
            regularized_pred = results['regularized'].get('probabilities', results['regularized'].get('predictions', []))
            posthoc_vanilla_shap = results['posthoc_vanilla']['global_importance']
            posthoc_reg_shap = results['posthoc_regularized']['global_importance']
            
            presentation_plotter.create_predictions_comparison(
                vanilla_pred, regularized_pred,
                posthoc_vanilla_shap, posthoc_reg_shap,
                feature_names,
                save_name='03_predictions_comparison.png'
            )
        
        # 3. График сравнения ROC кривых
        if config['dataset']['task_type'] == 'classification' and 'vanilla' in results and 'regularized' in results:
            print("[INFO] Создание графика сравнения ROC кривых...")
            presentation_plotter.create_roc_comparison(
                results['vanilla'], results['regularized'], y_test,
                save_name='04_roc_comparison.png'
            )
        
        # 4. График сравнения метрик (столбчатая диаграмма)
        if config['dataset']['task_type'] == 'classification' and 'vanilla' in results and 'regularized' in results:
            print("[INFO] Создание графика сравнения метрик...")
            presentation_plotter.create_metrics_bar_comparison(
                results['vanilla'], results['regularized'],
                save_name='05_metrics_bar_comparison.png'
            )
        
        # 5. График сравнения важности признаков (встроенная vs post-hoc)
        if ('vanilla' in results and 'regularized' in results and 
            'posthoc_vanilla' in results and 'posthoc_regularized' in results):
            print("[INFO] Создание графика сравнения важности признаков...")
            vanilla_importance = results['vanilla']['feature_importance']
            regularized_importance = results['regularized']['feature_importance']
            posthoc_vanilla_shap = results['posthoc_vanilla']['global_importance']
            posthoc_reg_shap = results['posthoc_regularized']['global_importance']
            
            presentation_plotter.create_feature_importance_comparison(
                vanilla_importance, regularized_importance,
                posthoc_vanilla_shap, posthoc_reg_shap,
                feature_names,
                save_name='06_feature_importance_comparison.png'
            )
        
        # 6. Детальное сравнение правил
        if 'rules_comparison' in results:
            print("[INFO] Создание детального графика сравнения правил...")
            presentation_plotter.create_rules_comparison_detailed(
                results['rules_comparison'], feature_names,
                save_name='07_rules_comparison_detailed.png'
            )
        
        # 7. График сравнения времени обучения
        print("[INFO] Создание графика сравнения времени обучения...")
        presentation_plotter.create_training_time_comparison(
            save_name='08_training_time_comparison.png'
        )

    # Анализ времени обучения
    if 'vanilla' in results and 'regularized' in results:
        print("\n" + "=" * 40)
        print("АНАЛИЗ ВРЕМЕНИ ОБУЧЕНИЯ")
        print("=" * 40)
        vanilla_time = results['vanilla']['training_time']
        regularized_time = results['regularized']['training_time']
        speedup = vanilla_time / regularized_time if regularized_time > 0 else 0
        
        print(f"Vanilla ANFIS: {vanilla_time:.2f} сек")
        print(f"{METHOD_LABEL_INLINE}: {regularized_time:.2f} сек")
        print(f"Ускорение: {speedup:.2f}x")
        
        if 'posthoc_vanilla' in results:
            posthoc_time = results['posthoc_vanilla']['analysis_time']
            total_posthoc_time = vanilla_time + posthoc_time
            speedup_vs_posthoc = total_posthoc_time / regularized_time if regularized_time > 0 else 0
            print(f"\nVanilla + Post-hoc SHAP: {total_posthoc_time:.2f} сек")
            print(f"Ускорение относительно post-hoc: {speedup_vs_posthoc:.2f}x")
        
        # Сохранение анализа времени
        if args.save_results:
            time_analysis = {
                'vanilla_time': vanilla_time,
                'regularized_time': regularized_time,
                'speedup': speedup,
                'posthoc_time': results.get('posthoc_vanilla', {}).get('analysis_time', 0),
                'total_posthoc_time': vanilla_time + results.get('posthoc_vanilla', {}).get('analysis_time', 0),
                'posthoc_regularized_time': results.get('posthoc_regularized', {}).get('analysis_time', 0),
                'total_posthoc_regularized_time': regularized_time + results.get('posthoc_regularized', {}).get('analysis_time', 0),
                'speedup_vs_posthoc': speedup_vs_posthoc if 'posthoc_vanilla' in results else 0
            }
            time_file = results_dir / 'time_analysis.json'
            with open(time_file, 'w', encoding='utf-8') as f:
                json.dump(time_analysis, f, indent=2, ensure_ascii=False)
            print(f"[OK] Анализ времени сохранен: {time_file}")

    if args.save_results:
        print(f"\n[INFO] Результаты сохранены в: {results_dir}")
        
        # Автоматическая проверка метрик после завершения
        print("\n" + "=" * 60)
        print("АВТОМАТИЧЕСКАЯ ПРОВЕРКА МЕТРИК")
        print("=" * 60)
        
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.check_metrics import check_metrics
            check_metrics(results_dir)
        except Exception as e:
            print(f"[WARNING] Ошибка при проверке метрик: {e}")
            print("   Можно запустить вручную: python scripts/check_metrics.py")


if __name__ == "__main__":
    main()
