#!/bin/bash
# Скрипт для ожидания завершения обучения и анализа результатов

cd /home/lebedeffson/PycharmProjects/шап-рег/XAI-2.0-SHAP-regularized-ANFIS

echo "Ожидание завершения обучения..."
echo ""

# Активация виртуального окружения
source ~/venv/bin/activate 2>/dev/null || true

# Проверка каждые 30 секунд
while true; do
    python scripts/check_training_progress.py
    
    # Проверяем, завершено ли обучение
    if python scripts/check_training_progress.py 2>&1 | grep -q "16/16"; then
        echo ""
        echo "✅ Обучение завершено!"
        echo ""
        break
    fi
    
    echo "Ожидание 30 секунд..."
    sleep 30
done

echo "Запуск анализа результатов..."
echo ""
python scripts/analyze_final_results.py

echo ""
echo "✅ Анализ завершен!"

