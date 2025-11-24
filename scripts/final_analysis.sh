#!/bin/bash
# Финальный анализ результатов после завершения обучения

cd /home/lebedeffson/PycharmProjects/шап-рег/XAI-2.0-SHAP-regularized-ANFIS
source ~/venv/bin/activate 2>/dev/null || true

echo "="*70
echo "ФИНАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ"
echo "="*70
echo ""

# Проверка завершения
python scripts/check_training_progress.py

# Анализ результатов
echo ""
echo "Запуск детального анализа..."
echo ""
python scripts/analyze_final_results.py

# Создание графиков для каждого датасета
echo ""
echo "Создание графиков сравнения gamma..."
echo ""

for dataset in breast_cancer heart_disease pima_diabetes wine_quality; do
    if [ -f "results/${dataset}/gamma_experiments/summary.json" ]; then
        echo "Обработка ${dataset}..."
        python -c "
import sys
sys.path.insert(0, 'src')
from visualization.dataset_gamma_plots import DatasetGammaPlotter
from pathlib import Path
import json

dataset = '${dataset}'
gamma_dir = Path(f'results/{dataset}/gamma_experiments')
summary_path = gamma_dir / 'summary.json'

if summary_path.exists():
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    plotter = DatasetGammaPlotter(gamma_dir, summary)
    plotter.create_all_plots()
    print(f'  ✅ Графики созданы для {dataset}')
" 2>&1 | grep -E "(✅|ERROR|Ошибка)" || echo "  ⚠️  Проблема с созданием графиков для ${dataset}"
    fi
done

echo ""
echo "="*70
echo "АНАЛИЗ ЗАВЕРШЕН"
echo "="*70
echo ""
echo "Результаты сохранены в:"
echo "  - results/*/gamma_experiments/summary.json"
echo "  - results/*/gamma_experiments/*.png"
echo ""
