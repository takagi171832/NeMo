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
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple
from cy_unigram_lm import FastOneGramModelCython
from beamsearch_cython import beamsearch_cy

# ログの設定
logging.basicConfig(level=logging.DEBUG)

@dataclass
class BeamEntry:
    SCORE:float = np.log(1) #音声認識モデルと言語モデルのスコアを保存
    label_out: Tuple = ()
    context: Tuple = ()

class BeamList:
    def __init__(self) -> None:
        self.entries = defaultdict(BeamEntry)

    def normalize(self) -> None:
        for k, v in self.entries.items():
            context_len = len(v.context)
            self.entries[k].SCORE = v.SCORE / (context_len if context_len > 0 else 1)

    def sort(self) -> List[Tuple]:
        # まず、SCOREでソートします
        sorted_entries = sorted(self.entries.items(), key=lambda x: x[1].SCORE, reverse=True)
        # 次に、ソートされたエントリからlabel_outのみを抽出してリストを作成します
        sorted_label_outs = [entry[1].label_out for entry in sorted_entries]
        return sorted_label_outs

# 引数を受け取る。
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="/home/takagi/NeMo/models/ASR/uni_LSTM_8_bpe_CSJ.nemo", help="NeMoモデルのパス")
    parser.add_argument("--cache_path", type=str, default="/home/takagi/NeMo/dataset/output/CSJ_to_laboro/tedx10k_chache", help="キャッシュファイルのパス")
    parser.add_argument("--use_unigram_add", action="store_true", help="加算用言語モデルにunigramを使用するかどうか")
    parser.add_argument("--use_unigram_sub", action="store_true", help="減算用言語モデルにunigramを使用するかどうか")
    parser.add_argument("--add_lm", type=str, default="/home/takagi/NeMo/models/LM/cy_laboro1gram.pkl", help="加算用言語モデルのパス")
    parser.add_argument("--sub_lm", type=str, default="/home/takagi/NeMo/models/LM/cy_csj1gram.pkl", help="減算用言語モデルのパス")
    parser.add_argument("--add_weight", type=List[float] , default=[0.1], help="加算用言語モデルの重み")
    parser.add_argument("--sub_weight", type=List[float] , default=[0.3], help="減算用言語モデルの重み")
    parser.add_argument("--batch_size", type=int, default=25, help="バッチサイズ")
    parser.add_argument("--test_manifest", type=str, default="/home/takagi/NeMo/manifests/laboroTV/tedx-jp-10k/tedx-jp-10k_manifest.json", help="テストデータのパス")
    parser.add_argument("--output_folder", type=str, default="/home/takagi/NeMo/dataset/output/dra_result_beam/", help="出力先のフォルダ")
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
    cut_off_n = 50
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
                    add_score = lm_add.score(" ".join((list(context) + [current_frame_char])[-1*add_ngram:]))*math.log(10)
                    # 減算のlmのスコアを取得する。
                    sub_score = lm_sub.score(" ".join((list(context) + [current_frame_char])[-1*sub_ngram:]))*math.log(10)
                    # 加算のlmのスコアと減算のlmのスコアをlogitに足す。
                    new_scores[t, current_frame_index] = new_scores[t, current_frame_index] + add_weight*add_score - sub_weight*sub_score

                logging.debug(f'Loop iteration: {t}')
            # 現在のフレームでのnew_scoresの文字のindexを取得する。
            pred_frame_index = np.argmax(new_scores[t]).item()
            #連続で同じ文字でなく、かつ、blankでない場合、contextに追加する。
            if not pred_frame_index == blank_idx and not pred_frame_index == current_frame_index:
                context.append(vocab[pred_frame_index])
            pred_frame_index = pred_frame_index

        batch_new_scores.append(new_scores)

    return np.array(batch_new_scores, dtype=object)

def batch_beamsearch_ngram_lm_rescoring(logits_batch, vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram):
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
    cut_off_n = 50
    batch_new_scores = []
    # ngramの最大次数を取得する。
    max_ngram = max(add_ngram, sub_ngram)
    # blankのidを取得する。
    blank_idx = len(vocab)
    # blankの文字を取得する。

    # ビームサーチのビーム幅
    beam_width = 1

    # バッチサイズ分のループを回す。
    for logits in logits_batch:
        # ngramの最大次数でタプルにpaddingを行う。
        context = tuple(["<s>"] * max_ngram)

        # ビームを初期化する。
        last = BeamList()
        label_out = ()
        last.entries[label_out] = BeamEntry()
        last.entries[label_out].context = context

        T, V = logits.shape
        new_scores = np.zeros(logits.shape)

        # Tの長さ分のループを回す。
        for t in range(T):
            curr = BeamList()
            # logits[t]の値の中から、大きい方からcut_off_nまでのindexを取得する。
            top_n_idx = np.argsort(logits[t])[::-1][:cut_off_n]
            # logging.debug(f"top_n_idx: {top_n_idx}")
            #beam幅分のbest_label
            best_label = last.sort()[:beam_width]
            # logging.debug(f"best_label: {best_label}")
            # logging.debug(f"last.entries[best_label]: {last.entries[best_label[0]]}")
            # logging.debug(f"last.entries[best_label].context: {last.entries[best_label[0]].context}")

            for label in best_label:
                logging.debug(f"label.entries.context: {last.entries[label].context}")

            new_scores[t] = logits[t]
            for label in best_label:
                for v in top_n_idx:
                    # 現在のフレームでの文字のindexを取得する。
                    current_frame_index = v.item()
                    # v = 1で_が出る場合、次のループに入る。
                    if current_frame_index == 1:
                        continue
                    
                    # 現在のフレームでの文字を取得する。
                    if not current_frame_index == blank_idx:
                        current_frame_char = vocab[current_frame_index]
                    else:
                        current_frame_char = "<blank>"

                    # 一つ前のフレームでの文字のindexを取得する。
                    if label and not label[-1] == "<blank>":
                        pred_frame_index = vocab.index(label[-1])
                    else:
                        pred_frame_index = blank_idx

                    if not current_frame_index == pred_frame_index and not current_frame_index == blank_idx:
                        logging.debug(f"lm_text: {' '.join((list(last.entries[label].context) + [current_frame_char])[-2*add_ngram+1:])}")
                        # 加算のlmのスコアを取得する。
                        add_score = lm_add.score(" ".join((list(last.entries[label].context) + [current_frame_char])[-2*add_ngram+1:]))*math.log(10)
                        # 減算のlmのスコアを取得する。
                        sub_score = lm_sub.score(" ".join((list(last.entries[label].context) + [current_frame_char])[-2*sub_ngram+1:]))*math.log(10)
                        # 加算のlmのスコアと減算のlmのスコアとASRモデルのスコアを足す。
                        new_score = last.entries[label].SCORE + logits[t, current_frame_index] + add_weight*add_score - sub_weight*sub_score
                        new_label = label + (current_frame_char,)
                        if not new_label in curr.entries:
                            curr.entries[new_label].label_out = new_label
                            curr.entries[new_label].SCORE = new_score
                            curr.entries[new_label].context = tuple(last.entries[label].context) + (current_frame_char,)
                    else:
                        new_score = last.entries[label].SCORE + logits[t, current_frame_index]
                        new_label = label + (current_frame_char,)
                        if not new_label in curr.entries:
                            curr.entries[new_label].label_out = new_label
                            curr.entries[new_label].SCORE = new_score
                            curr.entries[new_label].context = last.entries[label].context
            last = curr
        last.normalize()
        # last.sort()の中から、最もSCOREが高いものを取得する。
        # paddingを削除する。
        best_label = last.sort()[:beam_width]

        for label in best_label:
            logging.debug(f"last.entries[label]: {last.entries[label]}")
        batch_new_scores.append("".join(last.entries[best_label[0]].context[max_ngram:]))

    #文字列のリストを返す。
    return batch_new_scores

def process_batch(batch_data):
    i, probs, vocab, lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram = batch_data
    rescored_text = beamsearch_cy(
        probs, list(vocab), lm_add, lm_sub, add_weight, sub_weight, add_ngram, sub_ngram
    )
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

    all_probs = all_probs[:len(target_transcript)]

    #音声認識結果とテストの正解文の数が一致するか確認する。
    if len(all_probs) != len(target_transcript):
        logging.error("音声認識結果とテストの正解文の数が一致しません。")
        return

    # 言語モデルをロードする。
    if args.use_unigram_add:
        lm_add = FastOneGramModelCython("", list(asr_model.decoder.vocabulary))
        lm_add.load_model(args.add_lm)
    else:
        lm_add = kenlm.Model(args.add_lm)

    if args.use_unigram_sub:
        lm_sub = FastOneGramModelCython("", list(asr_model.decoder.vocabulary))
        lm_sub.load_model(args.sub_lm)
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
    output_file_name = f"add_{add_ngram}_sub_{sub_ngram}.txt"
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
    for sub_weight in args.sub_weight:
        # 減算重みのをグリッドサーチする。
        for add_weight in args.add_weight:
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
            logging.debug(f"all_rescored_text: {all_rescored_text}")
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
