import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER, word_error_rate_detail
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import NeuralType
import kenlm
import pickle
import json
import numpy as np
import os
from nemo.collections.asr.metrics.wer import word_error_rate
import time
import torch
import math
from collections import deque
import argparse
import logging
from typing import List, Optional
from tqdm.auto import tqdm
import multiprocessing

logging.basicConfig(level=logging.INFO)

# 引数を受け取る。
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="/home/takagi/NeMo/models/ASR/uni_LSTM_8_bpe_CSJ.nemo", help="NeMoモデルのパス")
    parser.add_argument("--cache_path", type=str, default="/home/takagi/NeMo/dataset/output/CSJ_to_laboro/tedx10k_chache", help="キャッシュファイルのパス")
    parser.add_argument("--use_unigram_add", action="store_true", help="加算用言語モデルにunigramを使用するかどうか")
    parser.add_argument("--use_unigram_sub", action="store_true", help="減算用言語モデルにunigramを使用するかどうか")
    parser.add_argument("--add_lm", type=str, default="/home/takagi/NeMo/models/LM/laboro1gram.pkl", help="加算用言語モデルのパス")
    parser.add_argument("--sub_lm", type=str, default="/home/takagi/NeMo/models/LM/csj1gram.pkl", help="減算用言語モデルのパス")
    parser.add_argument("--add_weight", type=List[float] , default=[0.1,0.3,0.5,0.7,0.9,1.1], help="加算用言語モデルの重み")
    parser.add_argument("--sub_weight", type=List[float] , default=[0.0], help="減算用言語モデルの重み")
    parser.add_argument("--batch_size", type=int, default=5, help="バッチサイズ")
    parser.add_argument("--test_manifest", type=str, default="/home/takagi/NeMo/manifests/laboroTV/tedx-jp-10k/tedx-jp-10k_manifest.json", help="テストデータのパス")
    parser.add_argument("--output_folder", type=str, default="/home/takagi/NeMo/dataset/output/dra_result/", help="出力先のフォルダ")
    return parser.parse_args()

def batch_ngram_lm_rescoring(logits_batch, vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram):
    """
    logits_batch: ASRモデルの出力 (T, V, B)
    vocab: ASRモデルの語彙
    lm_add: 加算用言語モデル
    lm_sub: 減算用言語モデル
    add_weight: 加算用言語モデルの重みのリスト
    sub_weight: 減算用言語モデルの重みのリスト
    add_ngram: 加算用言語モデルのngramの次数
    sub_ngram: 減算用言語モデルのngramの次数
    """
    cut_off_n = 40
    batch_new_scores = []
    # ngramの最大次数を取得する。
    max_ngram = max(add_ngram, sub_ngram)
    # blankのidを取得する。
    blank_idx = len(vocab)

    # バッチサイズ分のループを回す。
    for logits in logits_batch:
        T, V = logits.shape
        new_scores = np.zeros(logits.shape)
        # ngramの最大次数でpaddingする。
        context = deque(["<s>"] * max_ngram, maxlen=max_ngram)
        # 1つ前のフレームでの文字の最大確率のindexを記録する。
        pred_frame_index = blank_idx
        # Tの長さ分のループを回す。
        for t in range(T):

            new_scores[t] = logits[t]
            # logits[t]の値の中から、大きい方からcut_off_nまでのindexを取得する。
            top_n_idx = np.argsort(logits[t])[::-1][:cut_off_n]
            logging.debug(f"top_n_idx: {top_n_idx}")
            for v in top_n_idx:
                # 現在のフレームでの文字のindexを取得する。
                current_frame_index = v.item() - 1
                if not current_frame_index == pred_frame_index:
                    # 現在のフレームでの文字を取得する。
                    current_frame_char = vocab[current_frame_index]
                    # 加算のlmのスコアを取得する。
                    add_score = lm_add.score(" ".join(list(context) + [current_frame_char]))*math.log(10)
                    # 減算のlmのスコアを取得する。
                    #sub_score = lm_sub.score(" ".join(list(context) + [current_frame_char]))*math.log(10)
                    # 加算のlmのスコアと減算のlmのスコアをlogitに足す。
                    new_scores[t, current_frame_index] = new_scores[t, current_frame_index] + add_weight*add_score #- sub_weight*sub_score
                
                logging.debug(f'Loop iteration: {t}')
            # 現在のフレームでのnew_scoresの文字のindexを取得する。
            pred_frame_index = np.argmax(new_scores[t]).item()
            #連続で同じ文字でなく、かつ、blankでない場合、contextに追加する。
            if not pred_frame_index == blank_idx and not pred_frame_index == current_frame_index:
                context.append(vocab[pred_frame_index])
            pred_frame_index = pred_frame_index

        batch_new_scores.append(new_scores)
    
    return np.array(batch_new_scores, dtype=object)

def process_batch(batch_data):
    i, probs, vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram = batch_data
    rescored_probs = batch_ngram_lm_rescoring(
        probs, vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram
    )
    rescored_text = [prob_to_str(vocab, rescored_prob) for rescored_prob in rescored_probs]
    return i, rescored_text

def batch_prob_to_str(vocab, probs_batch):
    batch_predicted_text = []
    predicted_ids_batch = np.argmax(probs_batch, axis=-1)
    for predicted_ids in predicted_ids_batch:
        pred = -1
        predicted_ids_list = []
        for p in predicted_ids:
            if not p == pred:
                if p == len(vocab):
                    predicted_ids_list.append("<blank>")
                else:
                    predicted_ids_list.append(vocab[int(p)])
            pred = p
        predicted_text = "".join([i for i in predicted_ids_list if not i == "<blank>"])
        batch_predicted_text.append(predicted_text)
    return batch_predicted_text

def prob_to_str(vocab, probs):
    predicted_ids = np.argmax(probs, axis=-1)
    pred = -1
    predicted_ids_list = []
    for p in predicted_ids:
        if not p == pred:
            if p == len(vocab) :
                predicted_ids_list.append("<blank>")
            else:
                predicted_ids_list.append(vocab[int(p)])
        pred = p
    predicted_text = "".join([i for i in predicted_ids_list if not i == "<blank>"])
    #bpeのため、最初の文字を削除する。
    predicted_text = predicted_text[1:]
    return predicted_text

def main():
    args = get_parser()
    # モデルをロードする。
    asr_model = EncDecCTCModel.restore_from(args.asr_model)
    # テストデータのマニフェストからテキストを取得する。
    # 抜き出したテキストはListに格納する。
    target_transcript = []
    all_probs = []

    with open(args.test_manifest, "r") as f:
        for line in f:
            target_transcript.append(json.loads(line)["text"])
    
    # cache_pathから、音声認識結果を取得する。
    # もしcache_pathがない場合は、音声認識を行う。
    if not os.path.exists(args.cache_path):
        logging.info("cache_pathが存在しません。")
        logging.info("音声認識を行います。")
        # テストデータのマニフェストから音声認識を行う。
        with open(args.test_manifest, "r") as f:
            for line in f:
                # マニフェストから音声認識を行う。
                # 音声認識結果は、all_probsに格納する。
                all_probs.append(asr_model.transcribe([json.loads(line)["audio_filepath"]], logprobs=True)[0])
    else:
        with open(args.cache_path, "rb") as f:
            all_probs = pickle.load(f)

    # もし、音声認識結果がない場合は、音声認識を行う。
    
    #音声認識結果とテストの正解文の数が一致するか確認する。
    if len(all_probs) != len(target_transcript):
        logging.error("音声認識結果とテストの正解文の数が一致しません。")
        return
    
    # 言語モデルをロードする。
    if args.use_unigram_add:
        with open(args.add_lm, "rb") as f:
            lm_add = pickle.load(f)
    else:
        lm_add = kenlm.Model(args.add_lm)
    
    if args.use_unigram_sub:
        with open(args.sub_lm, "rb") as f:
            lm_sub = pickle.load(f)
    else:
        lm_sub = kenlm.Model(args.sub_lm)

    # 加算用言語モデルのngramの次数を取得する。
    if args.use_unigram_add:
        add_ngram = 1
    else:
        add_ngram = lm_add.order
    
    # 減算用言語モデルのngramの次数を取得する。
    if args.use_unigram_sub:
        sub_ngram = 1
    else:
        sub_ngram = lm_sub.order

    # 音声認識モデルの語彙を取得する。
    vocab = asr_model.decoder.vocabulary

    #結果を出力するファイルを作成する。
    #ファイル名はadd_lmとsub_lmのngramの次数とする。
    output_file_name = f"add_{add_ngram}.txt"
    output_file_path = os.path.join(args.output_folder, output_file_name)
    with open(output_file_path, "a") as f:
        # 使用したパラメータを出力する。
        f.write(f"asr_model: {args.asr_model}\n")
        f.write(f"test_manifest: {args.test_manifest}\n")
        f.write(f"add_weight: {args.add_weight}, sub_weight: {args.sub_weight}\n")
        f.write(f"add_ngram: {add_ngram}, sub_ngram: {sub_ngram}\n")
        f.write(f"add_lm: {args.add_lm}\n")
        f.write(f"sub_lm: {args.sub_lm}\n")
    
    # 加算重みのをグリッドサーチする。
    for add_weight in args.add_weight:
        # 減算重みのをグリッドサーチする。
        for sub_weight in args.sub_weight:
            results = []
            all_rescored_text = []
            batch_data = [
                (i, all_probs[i * args.batch_size : (i + 1) * args.batch_size], vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram)
                for i in range(int(math.ceil(len(all_probs) / args.batch_size)))
            ]
            # マルチプロセッシングとtqdmの組み合わせ
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                with tqdm(total=len(batch_data)) as pbar:
                    for index, result in pool.imap_unordered(process_batch, batch_data):
                        results.append((index, result))
                        pbar.update(1)

            # 結果を元の順序に並べ替え
            results.sort(key=lambda x: x[0])
            for index, result in results:
                    all_rescored_text += result
            # werを計算する。
            cer, _, ins_rate, del_rate, sub_rate = word_error_rate_detail(all_rescored_text, target_transcript, use_cer=True)
            # 結果を出力する。
            logging.info(
                f"add_weight: {add_weight}, sub_weight: {sub_weight}, cer: {cer}, ins_rate: {ins_rate}, del_rate: {del_rate}, sub_rate: {sub_rate}"
            )

            # 結果をファイルに保存する。
            with open(output_file_path, "a") as f:
                f.write(
                    f"add_weight: {add_weight}, sub_weight: {sub_weight}, cer: {cer}, ins_rate: {ins_rate}, del_rate: {del_rate}, sub_rate: {sub_rate}\n"
                )


if __name__ == "__main__":
    main()
