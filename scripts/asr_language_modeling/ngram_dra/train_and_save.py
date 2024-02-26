# train_and_save.py
from cy_unigram_lm import FastOneGramModelCython
import pickle
import json
import nemo.collections.asr as nemo_asr

# 言語モデルのボキャブラリを取得します。
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from("/home/takagi/NeMo/models/ASR/uni_LSTM_8_bpe_CSJ.nemo")
vocab = asr_model.decoder.vocabulary

# 訓練データを用意します。
laboro_train_path = "/home/takagi/NeMo/manifests/laboroTV/train_nodev_sp/train_nodev_sp_manifest.json"
with open(laboro_train_path, "r") as json_r:
    laboro_training_text = ""
    for line in json_r:
        laboro_training_text += json.loads(line)["text"]

csj_train_path = "/home/takagi/NeMo/manifests/CSJ/train_nodup_sp/train_nodup_sp_manifest.json"
with open(csj_train_path, "r") as json_rr:
    csj_training_text = ""
    for line in json_rr:
        csj_training_text += json.loads(line)["text"]

# モデルの初期化と学習
laboro_lm = FastOneGramModelCython(laboro_training_text, list(vocab))
csj_lm = FastOneGramModelCython(csj_training_text, list(vocab))

#モデルを保存
laboro_lm.save_model("/home/takagi/NeMo/models/LM/cy_laboro1gram.pkl")
csj_lm.save_model("/home/takagi/NeMo/models/LM/cy_csj1gram.pkl")
