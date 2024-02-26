#!/bin/bash

# for i in {2..6}
# do
#     for j in {2..6}
#     do
#         python3 /home/takagi/NeMo/scripts/asr_language_modeling/ngram_dra/eval_ngram_dra_ctc_multi.py --add_lm="/home/takagi/NeMo/models/LM/laboro${i}gram.bin" --sub_lm="/home/takagi/NeMo/models/LM/csj${j}gram.bin"
#     done
# done


for j in {5..6}
do
    python3 /home/takagi/NeMo/scripts/asr_language_modeling/ngram_dra/eval_ngram_dra_ctc_multi_beam.py \
            --test_manifest="/home/takagi/NeMo/manifests/laboroTV/tedx-jp-10k/tedx-jp-10k_manifest.json" \
            --add_lm="/home/takagi/NeMo/models/LM/laboro${j}gram.bin" \
            --sub_lm="/home/takagi/NeMo/models/LM/csj5gram.bin"
done

python3 /home/takagi/NeMo/scripts/asr_language_modeling/ngram_dra/eval_ngram_dra_ctc_multi_beam.py \
--test_manifest="/home/takagi/NeMo/manifests/laboroTV/tedx-jp-10k/tedx-jp-10k_manifest.json" \
--add_lm="/home/takagi/NeMo/models/LM/laboro2gram.bin" \
--sub_lm="/home/takagi/NeMo/models/LM/csj2gram.bin"