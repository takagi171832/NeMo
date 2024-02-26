# cy_unigram_lm.pyx
cimport numpy as np
import numpy as np
from libc.math cimport log10
import pickle

cdef class FastOneGramModelCython:
    cdef np.ndarray probabilities
    cdef double smoothing
    cdef int vocabulary_size
    cdef dict char_to_index

    def __init__(self, training_text, list vocabulary, double smoothing=0.001):
        self.smoothing = smoothing
        self.vocabulary_size = len(vocabulary)
        self.char_to_index = {char: index for index, char in enumerate(vocabulary)}
        self.probabilities = self.build_model(training_text)

    cdef np.ndarray build_model(self, text):
        cdef int total_count
        cdef np.ndarray freq = np.zeros(self.vocabulary_size, dtype=np.int)
        for char in text:
            index = self.char_to_index.get(char, -1)
            if index != -1:
                freq[index] += 1
        total_count = np.sum(freq)
        smoothed_freq = freq + self.smoothing
        return smoothed_freq / (total_count + self.smoothing * self.vocabulary_size)

    cpdef double get_probability(self, char):
        index = self.char_to_index.get(char, -1)
        if index != -1:
            return self.probabilities[index]
        else:
            return 0.0

    cpdef double score(self, char):
        cdef double probability = self.get_probability(char)
        if probability > 0:
            return log10(probability)
        else:
            return -float('inf')

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.probabilities, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.probabilities = pickle.load(f)
