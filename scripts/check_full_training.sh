#!/bin/bash
# Скрипт для проверки прогресса полного обучения

cd /home/lebedeffson/PycharmProjects/шап-рег/XAI-2.0-SHAP-regularized-ANFIS

echo "=== ПРОВЕРКА ПРОГРЕССА ПОЛНОГО ОБУЧЕНИЯ ==="
echo ""

datasets=("breast_cancer" "heart_disease" "pima_diabetes" "wine_quality")

for dataset in "${datasets[@]}"; do
    echo "$dataset:"
    
    # Проверка процессов
    if ps aux | grep -q "run_experiment.py.*${dataset}" | grep -v grep; then
        echo "  ⏳ Обучение идет..."
    else
        # Проверка результатов
        count=$(ls -1 results/${dataset}/*.png 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "  ✅ Завершено ($count графиков)"
            
            # Проверка нужного графика
            if [ -f "results/${dataset}/feature_importance_comparison.png" ] || [ -f "results/${dataset}/06_feature_importance_comparison.png" ]; then
                echo "     ✅ График важности создан"
            else
                echo "     ⚠️  График важности отсутствует"
            fi
        else
            echo "  ⏳ Ожидание..."
        fi
    fi
done

echo ""
echo "Для просмотра логов:"
echo "  tail -f logs/*_full_*.log"
