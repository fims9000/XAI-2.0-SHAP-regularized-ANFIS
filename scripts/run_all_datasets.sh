#!/bin/bash
# Скрипт для запуска обучения на всех датасетах

set -e  # Остановка при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$HOME/venv"

cd "$PROJECT_DIR"

# Активация виртуального окружения
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "[INFO] Виртуальное окружение активировано"
else
    echo "[WARNING] Виртуальное окружение не найдено: $VENV_PATH"
fi

# Список датасетов
DATASETS=(
    "breast_cancer"
    "heart_disease"
    "banknote"
    "pima_diabetes"
    "wine_quality"
)

# Логирование
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/all_datasets_${TIMESTAMP}.log"

echo "=========================================="
echo "ЗАПУСК ОБУЧЕНИЯ НА ВСЕХ ДАТАСЕТАХ"
echo "=========================================="
echo "Дата: $(date)"
echo "Лог: $MAIN_LOG"
echo ""

# Функция для запуска на одном датасете
run_dataset() {
    local dataset=$1
    local log_file="$LOG_DIR/${dataset}_${TIMESTAMP}.log"
    
    echo "[INFO] Запуск обучения на датасете: $dataset"
    echo "Лог: $log_file"
    
    python experiments/run_experiment.py \
        --dataset "$dataset" \
        --experiment all \
        --save-results \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] Датасет $dataset завершен успешно"
        return 0
    else
        echo "[ERROR] Ошибка при обучении на датасете $dataset (код: $exit_code)"
        return $exit_code
    fi
}

# Запуск на всех датасетах
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Обработка: $dataset"
    echo "=========================================="
    
    if run_dataset "$dataset"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        FAILED_DATASETS+=("$dataset")
    fi
    
    echo ""
done

# Итоговая сводка
echo ""
echo "=========================================="
echo "ИТОГОВАЯ СВОДКА"
echo "=========================================="
echo "Успешно: $SUCCESS_COUNT"
echo "Ошибок:  $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "[WARNING] Датасеты с ошибками:"
    for ds in "${FAILED_DATASETS[@]}"; do
        echo "  - $ds"
    done
    echo ""
fi

# Проверка результатов
echo "Проверка результатов..."
for dataset in "${DATASETS[@]}"; do
    results_dir="$PROJECT_DIR/results/$dataset"
    if [ -d "$results_dir" ]; then
        png_count=$(ls -1 "$results_dir"/*.png 2>/dev/null | wc -l)
        csv_count=$(ls -1 "$results_dir"/*.csv 2>/dev/null | wc -l)
        echo "  $dataset: $png_count графиков, $csv_count CSV файлов"
    else
        echo "  $dataset: результаты не найдены"
    fi
done

echo ""
echo "=========================================="
echo "ЗАВЕРШЕНО"
echo "=========================================="
echo "Время: $(date)"
echo "Лог: $MAIN_LOG"

