import os
import json

# manifestを読み込んで、半角スペース区切りでテキストファイルに書き出す

input_file = "/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train.json"
output_file = "/home/takagi/NeMo/models/LM/CSJ/SPS/sps_lm_train_text.txt"

spaced_texts = []

with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            spaced_text = " ".join(text)
            spaced_texts.append(spaced_text)

with open(output_file, 'w', encoding='utf-8') as f:
    for text in spaced_texts:
        f.write(text + "\n")