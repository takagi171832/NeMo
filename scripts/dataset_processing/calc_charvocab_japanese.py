import json

def create_unique_character_list(input_file):
    unique_chars = set()  # 文字の重複を避けるためにセットを使用

    # JSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            # テキストの各文字をセットに追加
            unique_chars.update(text)

    # セットをリストに変換して返す（任意でソートも可能）
    return sorted(list(unique_chars))

# 入力ファイルのパスのリスト
input_file_list = ['/home/takagi/NeMo/manifests/CSJ/APS/APS_train.json',
'/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train.json',
'/home/takagi/NeMo/manifests/CSJ/APS/APS_valid.json',
'/home/takagi/NeMo/manifests/CSJ/SPS/SPS_valid.json',
'/home/takagi/NeMo/manifests/CSJ/eval1/eval1_manifest.json',
'/home/takagi/NeMo/manifests/CSJ/eval2/eval2_manifest.json',
'/home/takagi/NeMo/manifests/CSJ/eval3/eval3_manifest.json']

#　文字のリストを作成
unique_chars = set()
for input_file in input_file_list:
    unique_chars.update(create_unique_character_list(input_file))

# リストの内容を表示
# ''や,の後のスペースはなしで表示
# ソートした後に表示
print(sorted(list(unique_chars)))
print(len(unique_chars))
