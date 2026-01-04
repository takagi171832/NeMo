#!/bin/bash
# FlexCTC Training and Evaluation Script for All GigaSpeech Categories
# This script trains n-gram LMs and evaluates FlexCTC on 6 categories

set -e

# Activate virtual environment
source /home/takagi/NeMo/.venv/bin/activate

# Configuration
NEMO_ROOT="/home/takagi/NeMo"
SCRIPT_DIR="${NEMO_ROOT}/flexctc_scripts"
MODEL="stt_en_conformer_ctc_large_ls"
KENLM_BIN="/home/takagi/kenlm/build/bin"
MANIFEST_BASE="/home/takagi/NeMo_arxiv/manifests/GigaSpeech"
OUTPUT_DIR="${NEMO_ROOT}/models/LM/GigaSpeech_FlexCTC"
RESULTS_DIR="${NEMO_ROOT}/results/FlexCTC"
BEAM_SIZE=20
LM_ALPHA_VALUES=(0.8 0.9 1.0 1.1)  # Array of LM alpha values to test
NGRAM_ORDER=4
BATCH_SIZE=256
SKIP_LM_TRAINING=true  # Set to false to train LMs

# Categories
CATEGORIES=(0 6 12 17 21 28)
CATEGORY_NAMES=("y-people" "p-news" "y-news" "y-science" "y-education" "audiobook")

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "FlexCTC Training and Evaluation"
echo "======================================================================"
echo "Model: $MODEL"
echo "Beam size: $BEAM_SIZE"
echo "LM alpha: $LM_ALPHA"
echo "N-gram order: $NGRAM_ORDER"
echo "Categories: ${CATEGORIES[*]}"
echo "======================================================================"
echo ""

# Phase 1: Train N-gram LMs for all categories (optional)
if [ "$SKIP_LM_TRAINING" = false ]; then
    echo "======================================================================"
    echo "Phase 1: Training N-gram Language Models"
    echo "======================================================================"

    for i in "${!CATEGORIES[@]}"; do
        CATEGORY=${CATEGORIES[$i]}
        CATEGORY_NAME=${CATEGORY_NAMES[$i]}

        TRAIN_MANIFEST="${MANIFEST_BASE}/category_${CATEGORY}/split/train_manifest.json"
        LM_OUTPUT="${OUTPUT_DIR}/category_${CATEGORY}_${NGRAM_ORDER}gram"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "Training LM for category ${CATEGORY} (${CATEGORY_NAME})"
        echo "  Train manifest: ${TRAIN_MANIFEST}"
        echo "  Output: ${LM_OUTPUT}"
        echo "----------------------------------------------------------------------"

        # Check if LM already exists
        if [ -f "${LM_OUTPUT}.nemo" ]; then
            echo "  LM already exists, skipping training..."
        else
            cd ${NEMO_ROOT}
            python scripts/asr_language_modeling/ngram_lm/train_kenlm.py \
                nemo_model_file=${MODEL} \
                "train_paths=[${TRAIN_MANIFEST}]" \
                kenlm_bin_path=${KENLM_BIN} \
                kenlm_model_file=${LM_OUTPUT}.bin \
                ngram_length=${NGRAM_ORDER} \
                preserve_arpa=true \
                save_nemo=True
            cd ${SCRIPT_DIR}

            echo "  LM training completed!"
        fi
    done
else
    echo "======================================================================"
    echo "Phase 1: Skipping N-gram LM Training (SKIP_LM_TRAINING=true)"
    echo "======================================================================"
fi

echo ""
echo "======================================================================"
echo "Evaluating FlexCTC on All Categories"
echo "======================================================================"

# Create results file
RESULTS_FILE="${RESULTS_DIR}/flexctc_results_$(date +%Y%m%d_%H%M%S).txt"
echo "FlexCTC Evaluation Results - LM Alpha Sweep" > "$RESULTS_FILE"
echo "===========================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Model: $MODEL" >> "$RESULTS_FILE"
echo "Beam size: $BEAM_SIZE" >> "$RESULTS_FILE"
echo "Batch size: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "LM alpha values tested: ${LM_ALPHA_VALUES[*]}" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Arrays to store best results for each category
declare -A best_wer
declare -A best_alpha
declare -A best_rtfx

for i in "${!CATEGORIES[@]}"; do
    CATEGORY=${CATEGORIES[$i]}
    CATEGORY_NAME=${CATEGORY_NAMES[$i]}

    TEST_MANIFEST="${MANIFEST_BASE}/category_${CATEGORY}/split/test_manifest.json"
    LM_FILE="${OUTPUT_DIR}/category_${CATEGORY}_${NGRAM_ORDER}gram.bin.nemo"

    echo ""
    echo "======================================================================"
    echo "Category ${CATEGORY} (${CATEGORY_NAME})"
    echo "======================================================================"
    echo "  Test manifest: ${TEST_MANIFEST}"
    echo "  LM file: ${LM_FILE}"
    echo ""

    # Initialize best WER with a high value
    best_wer[$CATEGORY]=999.99
    best_alpha[$CATEGORY]=0.0
    best_rtfx[$CATEGORY]=0.0

    # Test each LM alpha value
    for LM_ALPHA in "${LM_ALPHA_VALUES[@]}"; do
        echo "----------------------------------------------------------------------"
        echo "[${i}/${#CATEGORIES[@]}] Category ${CATEGORY} (${CATEGORY_NAME}) - LM alpha: ${LM_ALPHA}"
        echo "----------------------------------------------------------------------"

        LOG_FILE="${RESULTS_DIR}/category_${CATEGORY}_${CATEGORY_NAME}_alpha${LM_ALPHA}.log"

        # Run evaluation
        python ${SCRIPT_DIR}/evaluate.py \
            --model ${MODEL} \
            --manifest ${TEST_MANIFEST} \
            --lm ${LM_FILE} \
            --lm_alpha ${LM_ALPHA} \
            --beam_size ${BEAM_SIZE} \
            --batch_size ${BATCH_SIZE} \
            2>&1 | tee "${LOG_FILE}"

        # Extract WER and RTFx from log
        WER=$(grep "FlexCTC (beam_batch):" -A 10 "${LOG_FILE}" | grep "WER:" | head -1 | awk '{print $2}' | sed 's/%//')
        RTFX=$(grep "FlexCTC (beam_batch):" -A 10 "${LOG_FILE}" | grep "RTFx:" | head -1 | awk '{print $2}' | sed 's/x//')

        echo "  Result: WER=${WER}%, RTFx=${RTFX}x"

        # Update best result if this is better
        if (( $(echo "$WER < ${best_wer[$CATEGORY]}" | bc -l) )); then
            best_wer[$CATEGORY]=$WER
            best_alpha[$CATEGORY]=$LM_ALPHA
            best_rtfx[$CATEGORY]=$RTFX
            echo "  ✓ New best WER for this category!"
        fi
        echo ""
    done

    # Save best result for this category
    echo "" >> "$RESULTS_FILE"
    echo "Category ${CATEGORY} (${CATEGORY_NAME}):" >> "$RESULTS_FILE"
    echo "  Best LM alpha: ${best_alpha[$CATEGORY]}" >> "$RESULTS_FILE"
    echo "  Best WER: ${best_wer[$CATEGORY]}%" >> "$RESULTS_FILE"
    echo "  RTFx: ${best_rtfx[$CATEGORY]}x" >> "$RESULTS_FILE"

    echo "  Evaluation completed for all alpha values!"
done

echo ""
echo "======================================================================"
echo "All evaluations completed!"
echo "======================================================================"
echo ""
echo "Best Results Summary:"
echo "----------------------------------------------------------------------"
for i in "${!CATEGORIES[@]}"; do
    CATEGORY=${CATEGORIES[$i]}
    CATEGORY_NAME=${CATEGORY_NAMES[$i]}
    echo "Category ${CATEGORY} (${CATEGORY_NAME}):"
    echo "  Best LM alpha: ${best_alpha[$CATEGORY]}"
    echo "  Best WER: ${best_wer[$CATEGORY]}%"
    echo "  RTFx: ${best_rtfx[$CATEGORY]}x"
    echo ""
done
echo "======================================================================"
echo "Results saved to: ${RESULTS_FILE}"
echo "======================================================================"
echo ""

cat "$RESULTS_FILE"
