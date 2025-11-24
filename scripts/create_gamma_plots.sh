#!/bin/bash
# Скрипт для создания графиков сравнения gamma после завершения экспериментов

cd "$(dirname "$0")/.."

echo "=== СОЗДАНИЕ ГРАФИКОВ СРАВНЕНИЯ GAMMA ==="
echo ""

source ~/venv/bin/activate 2>/dev/null || true

python -c "
from src.visualization.gamma_comparison_plots import GammaComparisonPlotter
from pathlib import Path

results_dir = 'results/gamma_experiments'
if Path(results_dir / 'summary.json').exists():
    plotter = GammaComparisonPlotter(results_dir)
    plotter.create_all_plots()
    print('\n[OK] Все графики созданы!')
else:
    print(f'[ERROR] Файл {results_dir}/summary.json не найден')
    print('Сначала запустите эксперименты: python experiments/gamma_experiments.py')
"

echo ""
echo "Графики сохранены в: results/gamma_experiments/"

