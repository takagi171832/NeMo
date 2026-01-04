# FlexCTC Scripts

FlexCTC (CTC Batched Beam Search) を使用したASRモデルの評価および言語モデル統合のためのスクリプト集です。

## 概要

このディレクトリには、NeMoのCTCモデルでFlexCTC（高速化されたバッチ化ビームサーチ）を使用するための4つのスクリプトが含まれています。

## スクリプト一覧

### 1. `test.py`
FlexCTCの基本的な動作確認を行うテストスクリプトです。

**用途:**
- FlexCTCの動作確認
- GPU環境のテスト
- 合成データを使用した簡易動作チェック

**使い方:**
```bash
python test.py
```

**動作:**
- 事前学習済みCTCモデル（stt_en_conformer_ctc_small）をロード
- 合成データでビームサーチを実行
- デコード結果を表示

### 2. `evaluate.py`
FlexCTCを使用してASRモデルを評価するメインスクリプトです。N-gram言語モデルとの統合に対応しています。

**用途:**
- 実データでのWER（Word Error Rate）評価
- 言語モデルとの統合評価
- グリーディデコードとの比較

**使い方:**
```bash
# 基本的な使い方（言語モデルなし）
python evaluate.py \
    --model stt_en_conformer_ctc_large \
    --manifest /path/to/test_manifest.json \
    --beam_size 16

# 言語モデルを使用する場合
python evaluate.py \
    --model stt_en_conformer_ctc_large \
    --manifest /path/to/test_manifest.json \
    --lm /path/to/language_model.bin.nemo \
    --lm_alpha 0.5 \
    --beam_size 20 \
    --compare_greedy
```

**主要なオプション:**
- `--model`: 使用する事前学習済みモデル名
- `--manifest`: 評価用マニフェストファイルのパス
- `--lm`: N-gram言語モデルのパス（.nemo形式）
- `--lm_alpha`: 言語モデルの重み（0.0～1.0）
- `--beam_size`: ビームサイズ（デフォルト: 16）
- `--beam_beta`: 単語挿入ペナルティ（デフォルト: 0.3）
- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--compare_greedy`: グリーディデコードとの比較を行う
- `--no_cuda_graphs`: CUDAグラフを無効化

### 3. `batch_evaluate.py`
複数のカテゴリに対して一括で評価を実行するスクリプトです。

**用途:**
- GigaSpeechの複数カテゴリを自動評価
- 複数の設定での一括評価

**使い方:**
1. スクリプト内の設定を編集：
   ```python
   CATEGORIES = [0, 6, 12, 17, 21, 28]
   CATEGORY_NAMES = ["y-people", "p-news", "y-news", "y-science", "y-education", "audiobook"]
   MANIFEST_BASE = "/path/to/manifests"
   LM_BASE = "/path/to/language_models"
   RESULTS_DIR = "/path/to/results"
   MODEL = "stt_en_conformer_ctc_large"
   BEAM_SIZE = 20
   LM_ALPHA = 0.5
   ```

2. スクリプトを実行：
   ```bash
   python batch_evaluate.py
   ```

**出力:**
- 各カテゴリのログファイル: `{RESULTS_DIR}/category_{n}_{name}.log`
- サマリーファイル: `{RESULTS_DIR}/summary.txt`

### 4. `train_and_eval.sh`
N-gram言語モデルの学習から評価までの全パイプラインを実行するシェルスクリプトです。

**用途:**
- 言語モデルの自動学習
- 学習から評価までの一貫したパイプライン実行

**使い方:**
1. スクリプト内の設定を編集：
   ```bash
   MODEL="stt_en_conformer_ctc_large"
   KENLM_BIN="/path/to/kenlm/build/bin"
   MANIFEST_BASE="/path/to/manifests"
   OUTPUT_DIR="/path/to/output/models"
   RESULTS_DIR="/path/to/results"
   BEAM_SIZE=20
   LM_ALPHA=0.5
   NGRAM_ORDER=4
   ```

2. スクリプトを実行：
   ```bash
   bash train_and_eval.sh
   ```

**フェーズ:**
1. Phase 1: N-gram言語モデルの学習（KenLM使用）
2. Phase 2: FlexCTCでの評価

## 環境要件

### 必須
- Python 3.8+
- PyTorch（CUDA対応推奨）
- NeMo (NVIDIA NeMo Toolkit)
- CUDA対応GPU（FlexCTCの高速化にはGPUが必要）

### オプション
- KenLM（N-gram言語モデルの学習用）

### インストール
```bash
# NeMoのインストール
pip install nemo_toolkit[asr]

# KenLMのインストール（言語モデル学習用）
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j4
```

## FlexCTCとは

FlexCTC（CTC Batched Beam Search）は、CTCベースのASRモデルにおいて、以下の特徴を持つ高速ビームサーチアルゴリズムです：

- **バッチ処理**: 複数の音声を同時に処理して高速化
- **CUDA高速化**: GPU上で最適化された実装
- **言語モデル統合**: N-gram言語モデルとの統合をサポート
- **柔軟性**: ビームサイズや言語モデルの重みを調整可能

## パフォーマンスのヒント

1. **ビームサイズ**: より大きなビームサイズは精度を向上させますが、計算コストが増加します（推奨: 16-32）
2. **言語モデルの重み (lm_alpha)**: 0.3-0.7の範囲で調整することを推奨
3. **バッチサイズ**: GPUメモリに応じて調整（推奨: 8-16）
4. **CUDAグラフ**: デフォルトで有効。さらなる高速化が可能

## トラブルシューティング

### CUDA out of memory
- バッチサイズを減らす
- ビームサイズを減らす

### 言語モデルが読み込めない
- `.nemo`形式の言語モデルファイルであることを確認
- KenLMでコンパイルされたバイナリファイルを使用

### 評価が遅い
- CUDAグラフが有効になっているか確認
- GPU使用率を確認（`nvidia-smi`）

## 参考情報

- [NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html)
- [KenLM](https://kheafield.com/code/kenlm/)

## ライセンス

このスクリプトはNeMoプロジェクトの一部として提供されています。
