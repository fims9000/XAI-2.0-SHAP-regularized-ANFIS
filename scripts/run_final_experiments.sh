#!/bin/bash
# Скрипт для запуска финальных экспериментов на всех датасетах

cd "$(dirname "$0")/.."

echo "=== ЗАПУСК ФИНАЛЬНЫХ ЭКСПЕРИМЕНТОВ ==="
echo ""
echo "Датасеты: breast_cancer, heart_disease, pima_diabetes"
echo "Gamma: 0.1, 0.3, 0.5, 0.7"
echo ""

source ~/venv/bin/activate 2>/dev/null || true

python experiments/gamma_experiments.py \
    --datasets breast_cancer heart_disease pima_diabetes \
    --gammas 0.1 0.3 0.5 0.7 \
    --output-dir results/final_experiments 2>&1 | tee logs/final_experiments.log

echo ""
echo "=== СОЗДАНИЕ ГРАФИКОВ ==="
echo ""

for dataset in breast_cancer heart_disease pima_diabetes; do
    echo "Создание графиков для $dataset..."
    python -c "
from src.visualization.dataset_gamma_plots import DatasetGammaPlotter
from pathlib import Path

# Проверяем наличие результатов
gamma_dir = Path(f'results/{dataset}/gamma_experiments')
if (gamma_dir / 'summary.json').exists():
    plotter = DatasetGammaPlotter('$dataset')
    plotter.create_all_plots()
    print(f'  ✅ Графики созданы для $dataset')
else:
    print(f'  ⚠️  Результаты не найдены для $dataset')
"
done

echo ""
echo "✅ Все эксперименты завершены!"
echo "Результаты сохранены в: results/{dataset}/gamma_experiments/"
