"""
Скрипт для перегенерации графиков из сохраненных данных без переобучения моделей
"""
import sys
import json
import argparse
import numpy as np
import yaml
from pathlib import Path

# Добавляем src в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from visualization.presentation_plots import PresentationPlotter


def regenerate_shap_plots(dataset: str, results_dir: Path = None):
    """Перегенерация только SHAP графиков из сохраненных данных"""
    
    if results_dir is None:
        results_dir = project_root / 'results' / dataset
    
    # Загрузка конфигурации
    config_path = project_root / 'configs' / f'{dataset}.yaml'
    if not config_path.exists():
        print(f"[ERROR] Конфигурация не найдена: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Загрузка SHAP данных
    shap_data_path = results_dir / 'shap_data.json'
    if not shap_data_path.exists():
        print(f"[ERROR] SHAP данные не найдены: {shap_data_path}")
        print("       Сначала запустите эксперимент: python experiments/run_experiment.py --dataset {dataset} --save-results")
        return False
    
    with open(shap_data_path, 'r') as f:
        shap_data = json.load(f)
    
    feature_names = shap_data['feature_names']
    
    # Преобразуем в numpy arrays
    vanilla_importance = np.array(shap_data['vanilla_importance']) if shap_data['vanilla_importance'] else None
    regularized_importance = np.array(shap_data['regularized_importance']) if shap_data['regularized_importance'] else None
    posthoc_vanilla = np.array(shap_data['posthoc_vanilla']) if shap_data['posthoc_vanilla'] else None
    posthoc_regularized = np.array(shap_data['posthoc_regularized']) if shap_data['posthoc_regularized'] else None
    
    print(f"[OK] SHAP данные загружены для {dataset}")
    print(f"     Признаков: {len(feature_names)}")
    
    # Создание визуализатора
    plotter = PresentationPlotter(config, save_dir=results_dir)
    
    # Генерация графика сравнения Post-hoc SHAP
    if posthoc_vanilla is not None and posthoc_regularized is not None:
        print("[INFO] Создание графика Post-hoc SHAP сравнения...")
        plotter.create_posthoc_shap_comparison(
            posthoc_vanilla, posthoc_regularized, feature_names,
            save_name='09_posthoc_shap_comparison.png'
        )
    
    # Генерация графика сравнения важности признаков (если есть все данные)
    if all([vanilla_importance is not None, regularized_importance is not None,
            posthoc_vanilla is not None, posthoc_regularized is not None]):
        print("[INFO] Создание графика сравнения важности признаков...")
        plotter.create_feature_importance_comparison(
            vanilla_importance, regularized_importance,
            posthoc_vanilla, posthoc_regularized,
            feature_names,
            save_name='06_feature_importance_comparison.png'
        )
    
    print(f"\n[OK] Графики сохранены в: {results_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Перегенерация графиков из сохраненных SHAP данных')
    parser.add_argument('--dataset', default='breast_cancer',
                        help='Название датасета')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Путь к директории с результатами (опционально)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) if args.results_dir else None
    
    print(f"Перегенерация графиков для: {args.dataset}")
    print("=" * 50)
    
    success = regenerate_shap_plots(args.dataset, results_dir)
    
    if success:
        print("\n[OK] Перегенерация завершена успешно!")
    else:
        print("\n[ERROR] Перегенерация не удалась")
        sys.exit(1)


if __name__ == "__main__":
    main()

