import nltk
from nltk import FreqDist
from nltk.util import ngrams
import json
import pickle
import torch

class JapaneseCharacterNgramModel:

    def __init__(self, vocabulary, training_text, n=1, smoothing=0.001):
        self.vocabulary = set(vocabulary)
        self.n = n
        self.smoothing = smoothing
        self.model = self.build_model(training_text)

    def build_model(self, training_text):
        ngrams_list = list(ngrams(training_text, self.n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ngram_freq = FreqDist(ngrams_list)
        total_count = sum(ngram_freq.values())

        model = {}
        for ngram, count in ngram_freq.items():
            probability = (count + self.smoothing) / (total_count + self.smoothing * len(self.vocabulary))
            model[ngram] = probability

        return model

    def likelihood(self, input_text):
        input_ngrams = list(ngrams(input_text, self.n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        likelihood = 1.0
        for ngram in input_ngrams:
            likelihood *= self.model.get(ngram, self.smoothing / (sum(self.model.values()) + self.smoothing * len(self.vocabulary)))

        return likelihood
    
    def raw_probability(self, character):
        ngram = (character,)
        count = self.model.get(ngram, 0) * (sum(self.model.values()) + self.smoothing * len(self.vocabulary)) - self.smoothing
        probability = count / sum(self.model.values())
        return probability
    
    def get_probability(self, char):
        char_ngram = tuple([char])
        probability = self.model.get(char_ngram, self.smoothing / (sum(self.model.values()) + self.smoothing * len(self.vocabulary)))
        return probability
    
    #textを受け取ったら、最後の文字のlog10ソフトマックスの値を返す。
    def score(self, text):
        #textから空白を除去
        text = text.replace(" ", "")
        char_ngram = tuple([text[-1]])
        probability = self.model.get(char_ngram, self.smoothing / (sum(self.model.values()) + self.smoothing * len(self.vocabulary)))
        return torch.log10(torch.tensor(probability))
    
    def calculate_ppl(self, text):
        total_log_prob = 0.0
        text = text.replace(" ", "")
        for char in text:
            total_log_prob += self.score(char).item()  # scoreメソッドから対数確率を取得

        avg_log_prob = total_log_prob / len(text)
        ppl = 10**(-avg_log_prob)
        return ppl