import pickle
import json
from unigram_lm import JapaneseCharacterNgramModel
import nemo.collections.asr as nemo_asr

# 言語モデルのボキャブラリを取得します。
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("/home/takagi/NeMo/models/ASR/CSJ_APS/LSTM-CTC-APS.nemo")
vocab = asr_model.cfg.labels

# 訓練データを用意します。
train_path = "/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train.json"
with open(train_path, "r") as json_r:
    training_text = ""
    for line in json_r:
        training_text += json.loads(line)["text"]

# 言語モデルを作成します。
lm = JapaneseCharacterNgramModel(vocab, training_text)

# 入力文字列の尤度を計算します。
input_csj_text = "という結果が得られました"
likelihood = lm.likelihood(input_csj_text)
print(f"入力文字列input_csj_text '{input_csj_text}' の尤度: {likelihood}")

input_char = ["と", "い", "う", "結", "果", "が", "得", "ら", "れ", "ま", "し", "た"]
for char in input_char:
    print(f"文字 '{char}' の確率: {lm.get_probability(char)}")

input_laboro_text = "レバノンの大爆発"
likelihood = lm.likelihood(input_laboro_text)
print(f"入力文字列input_laboro_text '{input_laboro_text}' の尤度: {likelihood}")

input_char = ["レ", "バ", "ノ", "ン", "の", "大", "爆", "発"]
for char in input_char:
    print(f"文字 '{char}' の確率: {lm.get_probability(char)}")



with open("/home/takagi/NeMo/models/LM/CSJ/SPS/sps_1gram.pkl", "wb") as f:
    pickle.dump(lm, f)