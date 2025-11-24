#!/bin/bash
# Скрипт для проверки результатов обучения

RESULTS_DIR="results/breast_cancer_test"

echo "=========================================="
echo "ПРОВЕРКА РЕЗУЛЬТАТОВ ОБУЧЕНИЯ"
echo "=========================================="
echo ""

# Проверка процесса
if pgrep -f "run_experiment.py" > /dev/null; then
    echo "⏳ Обучение еще идет..."
    echo ""
    echo "Последние строки лога:"
    tail -10 training_run.log 2>/dev/null | grep -E "(Эпоха|✅|📊|accuracy|roc_auc)" || tail -5 training_run.log 2>/dev/null
    echo ""
    echo "Для полной проверки дождитесь завершения обучения"
    exit 0
fi

echo "✅ Обучение завершено!"
echo ""

# Проверка CSV файла
if [ -f "$RESULTS_DIR/comparison_results.csv" ]; then
    echo "📊 МЕТРИКИ ИЗ CSV:"
    echo "----------------------------------------"
    cat "$RESULTS_DIR/comparison_results.csv" | column -t -s,
    echo ""
else
    echo "⚠️ CSV файл не найден"
fi

# Проверка JSON файлов
echo "📄 JSON ФАЙЛЫ:"
echo "----------------------------------------"
for json_file in "$RESULTS_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        echo "✅ $(basename $json_file)"
    fi
done
echo ""

# Проверка графиков
echo "📈 ГРАФИКИ:"
echo "----------------------------------------"
PNG_COUNT=$(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l)
if [ "$PNG_COUNT" -gt 0 ]; then
    echo "Найдено графиков: $PNG_COUNT"
    ls -lh "$RESULTS_DIR"/*.png 2>/dev/null | awk '{printf "  ✅ %-50s %6s\n", $9, $5}'
else
    echo "⚠️ Графики не найдены"
fi
echo ""

# Детальная проверка метрик
if [ -f "scripts/check_metrics.py" ] && [ -f "$RESULTS_DIR/comparison_results.csv" ]; then
    echo "=========================================="
    echo "ДЕТАЛЬНАЯ ПРОВЕРКА МЕТРИК"
    echo "=========================================="
    echo ""
    source ~/venv/bin/activate 2>/dev/null || true
    python scripts/check_metrics.py "$RESULTS_DIR" 2>&1
fi

