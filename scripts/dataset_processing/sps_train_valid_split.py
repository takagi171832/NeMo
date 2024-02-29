import json
import random

random.seed(42)

def extract_validation_set(input_file, total_samples=7000):
    # ランダムシードを固定

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ランダムに7000行を選択するためのインデックスを生成
    selected_indices = random.sample(range(len(lines)), total_samples)

    # 検証セット用の行を選択
    validation_lines = [lines[i] for i in selected_indices]

    # 選択されなかった行を訓練セットにする
    train_lines = [lines[i] for i in range(len(lines)) if i not in selected_indices]

    # 検証データセットをファイルに保存
    with open('/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train_tmp.json', 'w', encoding='utf-8') as f_valid:
        for line in validation_lines:
            f_valid.write(line)

    # 訓練データセットをファイルに保存
    with open('/home/takagi/NeMo/manifests/CSJ/SPS/SPS_valid.json', 'w', encoding='utf-8') as f_train:
        for line in train_lines:
            f_train.write(line)

    print(f"Validation set created with {len(validation_lines)} samples.")
    print(f"Training set created with {len(train_lines)} samples.")

# 入力ファイルのパス（例）
input_file = '/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train copy.json'
extract_validation_set(input_file)
