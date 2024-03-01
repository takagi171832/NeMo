# train_and_save.py
from cy_unigram_lm import FastOneGramModelCython
import pickle
import json
import nemo.collections.asr as nemo_asr

# 言語モデルのボキャブラリを取得します。
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("/home/takagi/NeMo/models/ASR/CSJ_APS/LSTM-CTC-APS.nemo", map_location="cpu")
vocab = asr_model.cfg.labels
# 訓練データを用意します。
aps_train_path = "/home/takagi/NeMo/manifests/CSJ/APS/APS_train.json"
with open(aps_train_path, "r") as json_r:
    aps_training_text = ""
    for line in json_r:
        aps_training_text += json.loads(line)["text"]

sps_train_path = "/home/takagi/NeMo/manifests/CSJ/SPS/SPS_train.json"
with open(sps_train_path, "r") as json_rr:
    sps_training_text = ""
    for line in json_rr:
        sps_training_text += json.loads(line)["text"]

# モデルの初期化と学習
aps_lm = FastOneGramModelCython(aps_training_text, list(vocab))
sps_lm = FastOneGramModelCython(sps_training_text, list(vocab))

#モデルを保存
aps_lm.save_model("/home/takagi/NeMo/models/LM/CSJ/APS/aps_1gram.pkl")
sps_lm.save_model("/home/takagi/NeMo/models/LM/CSJ/SPS/sps_1gram.pkl")
