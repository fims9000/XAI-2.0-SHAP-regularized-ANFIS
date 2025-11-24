#!/bin/bash
# Скрипт для запуска полных экспериментов с SHAP анализом

cd /home/lebedeffson/PycharmProjects/шап-рег/XAI-2.0-SHAP-regularized-ANFIS
source ~/venv/bin/activate 2>/dev/null || true

datasets=("breast_cancer" "heart_disease" "pima_diabetes" "wine_quality")

echo "======================================================================"
echo "ЗАПУСК ПОЛНЫХ ЭКСПЕРИМЕНТОВ С SHAP АНАЛИЗОМ"
echo "======================================================================"
echo ""

for dataset in "${datasets[@]}"; do
    echo "======================================================================"
    echo "ОБРАБОТКА: $dataset"
    echo "======================================================================"
    echo ""
    
    python experiments/run_experiment.py --dataset ${dataset} --experiment all --save-results 2>&1 | tee logs/${dataset}_full_$(date +%Y%m%d_%H%M%S).log
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ $dataset завершен успешно"
        echo "Проверка созданных файлов:"
        count=$(ls -1 results/${dataset}/*.png 2>/dev/null | wc -l)
        echo "  Графиков: $count"
        
        # Проверка нужного графика
        if [ -f "results/${dataset}/feature_importance_comparison.png" ] || [ -f "results/${dataset}/06_feature_importance_comparison.png" ]; then
            echo "  ✅ График важности создан"
        fi
        if [ -f "results/${dataset}/shap_methods_comparison.png" ]; then
            echo "  ✅ График сравнения SHAP создан"
        fi
    else
        echo ""
        echo "❌ Ошибка при обработке $dataset"
    fi
    
    echo ""
    echo "======================================================================"
    echo ""
done

echo "======================================================================"
echo "ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ"
echo "======================================================================"
