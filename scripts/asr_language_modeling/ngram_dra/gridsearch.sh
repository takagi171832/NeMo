#!/bin/bash

# 2gramから4gramまで、--use_unigram_subを含めて実行
for n in {2..4}
do
    lm_path="/home/takagi/NeMo/models/LM/CSJ/APS/aps_${n}gram.bin"
    python3 /home/takagi/NeMo/scripts/asr_language_modeling/ngram_dra/eval_ngram_dra_ctc_multi_beam.py \
    --asr_model="/home/takagi/NeMo/models/ASR/CSJ_SPS/LSTM-CTC-SPS.nemo" \
    --cache_path="/home/takagi/NeMo/dataset/output/SPS_to_APS/SPS_to_APS_cache" \
    --use_unigram_sub \
    --add_lm="$lm_path" \
    --test_manifest="/home/takagi/NeMo/manifests/CSJ/eval1/eval1_manifest.json" \
    --output_folder="/home/takagi/NeMo/dataset/output/SPS_to_APS/dra_result/"
done

# 2gramから4gramまでの全ての組み合わせで --add_lm と --sub_lm を使用して実行
for add_n in {2..4}
do
    for sub_n in {2..4}
    do
        add_lm_path="/home/takagi/NeMo/models/LM/CSJ/APS/aps_${add_n}gram.bin"
        sub_lm_path="/home/takagi/NeMo/models/LM/CSJ/SPS/sps_${sub_n}gram.bin"
        echo "Running with --add_lm $add_lm_path and --sub_lm $sub_lm_path"
        python3 /home/takagi/NeMo/scripts/asr_language_modeling/ngram_dra/eval_ngram_dra_ctc_multi_beam.py \
        --asr_model="/home/takagi/NeMo/models/ASR/CSJ_SPS/LSTM-CTC-SPS.nemo" \
        --cache_path="/home/takagi/NeMo/dataset/output/SPS_to_APS/SPS_to_APS_cache" \
        --sub_lm="$sub_lm_path" \
        --add_lm="$add_lm_path" \
        --test_manifest="/home/takagi/NeMo/manifests/CSJ/eval1/eval1_manifest.json" \
        --output_folder="/home/takagi/NeMo/dataset/output/SPS_to_APS/dra_result/"
    done
done
