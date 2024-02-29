import json
import os

def classify_and_save_json(input_file):
    # 分類されたデータを格納するリスト
    data_a = []
    data_s = []

    # 元のJSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            filename = os.path.basename(data["audio_filepath"])
            # ファイル名をチェックしてAまたはSに分類
            if 'A' in filename:
                data_a.append(data)
            elif 'S' in filename:
                data_s.append(data)

    # Aに該当するデータをJSONファイルに保存
    with open('/home/takagi/NeMo/manifests/CSJ/APS/APS_train.json', 'w', encoding='utf-8') as f_a:
        for item in data_a:
            f_a.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Sに該当するデータをJSONファイルに保存
    with open('/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train.json', 'w', encoding='utf-8') as f_s:
        for item in data_s:
            f_s.write(json.dumps(item, ensure_ascii=False) + '\n')

# 入力ファイルのパス（例）
input_file = '/home/takagi/NeMo/manifests/CSJ/train_nodup_sp/train_nodup_sp_manifest.json'
classify_and_save_json(input_file)

