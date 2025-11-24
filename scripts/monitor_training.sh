#!/bin/bash
# Скрипт для мониторинга процесса обучения и проверки метрик после завершения

RESULTS_DIR="results/breast_cancer_test"
LOG_FILE="full_training.log"

echo "Мониторинг обучения..."
echo "Лог файл: $LOG_FILE"
echo "Результаты: $RESULTS_DIR"
echo ""

# Проверка процесса
if pgrep -f "run_experiment.py" > /dev/null; then
    echo "✅ Процесс обучения запущен"
    echo ""
    echo "Последние строки лога:"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "Лог еще не создан"
else
    echo "⚠️ Процесс обучения не найден"
    echo ""
    
    # Проверка результатов
    if [ -f "$RESULTS_DIR/comparison_results.csv" ]; then
        echo "✅ Обучение завершено! Проверяем метрики..."
        echo ""
        python scripts/check_metrics.py "$RESULTS_DIR"
    else
        echo "⚠️ Результаты еще не созданы"
    fi
fi

