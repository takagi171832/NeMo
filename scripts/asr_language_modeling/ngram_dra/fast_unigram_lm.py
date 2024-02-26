import collections
import math
import pickle

class OneGramModel:
    def __init__(self, training_text, smoothing=0.001):
        self.smoothing = smoothing
        self.model = self.build_model(training_text)

    def build_model(self, text):
        freq = collections.Counter(text)
        total_count = sum(freq.values())
        vocabulary_size = len(freq)

        # スムージングを適用し、確率を計算
        probabilities = {char: (count + self.smoothing) / (total_count + self.smoothing * vocabulary_size) for char, count in freq.items()}

        # log10 確率に変換
        log_probabilities = {char: math.log10(prob) for char, prob in probabilities.items()}

        return log_probabilities

    def score(self, char):
        # 与えられた文字の log10 確率を返す。存在しない文字の場合はデフォルト値を返す。
        return self.model.get(char, math.log10(self.smoothing / (sum(self.model.values()) + self.smoothing * len(self.model))))

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)

# 使用例
training_text = "トレーニング用のテキストデータ"
model = OneGramModel(training_text)
model.save_model('1gram_model.pkl')

# モデルを読み込んで確率を取得
model.load_model('1gram_model.pkl')
print(model.score('あ'))  # 例えば「あ」という文字に対する log10 確率を取得
