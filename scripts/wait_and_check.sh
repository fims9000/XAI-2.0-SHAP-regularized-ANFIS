#!/bin/bash
# Скрипт для ожидания завершения обучения и проверки результатов

RESULTS_DIR="results/breast_cancer_test"
LOG_FILE="training_run.log"
MAX_WAIT=1800  # Максимум 30 минут

echo "⏳ Ожидание завершения обучения..."
echo "Максимальное время ожидания: $MAX_WAIT секунд"
echo ""

start_time=$(date +%s)

while true; do
    # Проверяем, запущен ли процесс
    if ! pgrep -f "run_experiment.py" > /dev/null; then
        echo "✅ Процесс обучения завершен!"
        echo ""
        break
    fi
    
    # Проверяем таймаут
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $MAX_WAIT ]; then
        echo "⚠️ Превышено максимальное время ожидания"
        break
    fi
    
    # Показываем прогресс каждые 30 секунд
    if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo "⏱️ Прошло: ${elapsed} сек. Обучение продолжается..."
        tail -3 "$LOG_FILE" 2>/dev/null | grep -E "(Эпоха|✅|📊)" || echo "   (ожидание...)"
    fi
    
    sleep 5
done

echo ""
echo "=========================================="
echo "ПРОВЕРКА РЕЗУЛЬТАТОВ"
echo "=========================================="
echo ""

# Проверяем наличие результатов
if [ -f "$RESULTS_DIR/comparison_results.csv" ]; then
    echo "✅ CSV файл с метриками найден"
    echo ""
    echo "📊 МЕТРИКИ:"
    cat "$RESULTS_DIR/comparison_results.csv"
    echo ""
else
    echo "⚠️ CSV файл не найден"
fi

# Проверяем графики
PNG_COUNT=$(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l)
if [ "$PNG_COUNT" -gt 0 ]; then
    echo "✅ Найдено графиков: $PNG_COUNT"
    ls -lh "$RESULTS_DIR"/*.png 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
else
    echo "⚠️ Графики не найдены"
fi

echo ""
echo "Запуск детальной проверки метрик..."
echo ""

# Запускаем проверку метрик
if [ -f "scripts/check_metrics.py" ]; then
    source ~/venv/bin/activate 2>/dev/null || true
    python scripts/check_metrics.py "$RESULTS_DIR" 2>&1
else
    echo "⚠️ Скрипт проверки метрик не найден"
fi

