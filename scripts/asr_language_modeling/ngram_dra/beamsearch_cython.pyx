# cython: language_level=3
from libc.math cimport log10
import numpy as np
cimport numpy as cnp
import math

cdef class BeamEntry:
    cdef public float SCORE
    cdef public tuple label_out
    cdef public tuple context

    def __cinit__(self):
        self.SCORE = math.log(1)
        self.label_out = ()
        self.context = ()
    
    def __reduce__(self):
        # __reduce__ メソッドで、オブジェクトを再構築するために必要な情報を返します
        return (self.__class__, (self.SCORE, self.label_out, self.context))

def compare_score(item):
    return item[1].SCORE

cdef class BeamList:
    cdef public dict entries

    def __init__(self):
        self.entries = {}

    cpdef normalize(self):
        cdef int context_len
        for k, v in self.entries.items():
            context_len = len(v.context)
            self.entries[k].SCORE = v.SCORE / (context_len if context_len > 0 else 1)

    cpdef sort(self):
        sorted_entries = sorted(self.entries.items(), key=compare_score, reverse=True)
        return [entry[1].label_out for entry in sorted_entries]

cpdef beamsearch_cy(list logits_batch, 
                                          list vocab, 
                                          lm_add, 
                                          lm_sub, 
                                          float add_weight, 
                                          float sub_weight, 
                                          int add_ngram, 
                                          int sub_ngram):
    cdef int cut_off_n = 100
    cdef list batch_new_scores = []
    cdef int max_ngram = max(add_ngram, sub_ngram)
    cdef int blank_idx = len(vocab)
    cdef int beam_width = 35
    cdef int T, V, t, v
    cdef float new_score, add_score, sub_score
    cdef BeamList last, curr
    cdef tuple context, new_label, label
    cdef cnp.ndarray top_n_idx
    cdef str current_frame_char
    cdef int current_frame_index, pred_frame_index

    for logits in logits_batch:
        logits = np.array(logits)
        context = ("<s>",) * max_ngram

        last = BeamList()
        last.entries[()] = BeamEntry()
        last.entries[()].context = context

        T, V = logits.shape
        for t in range(T):
            curr = BeamList()
            top_n_idx = np.argsort(logits[t])[::-1][:cut_off_n]
            for label in last.sort()[:beam_width]:
                for v in top_n_idx:
                    current_frame_index = v
                    if current_frame_index == 1:
                        continue
                    current_frame_char = "<blank>" if current_frame_index == blank_idx else vocab[current_frame_index]
                    pred_frame_index = blank_idx
                    if not label or label[-1] == "<blank>":
                        pred_frame_index = blank_idx
                    else:
                        # label[-1] が BeamEntry オブジェクトである場合、その label_out 属性を使用
                        label_str = label[-1].label_out if isinstance(label[-1], BeamEntry) else label[-1]
                        if label_str in vocab:
                            pred_frame_index = vocab.index(label_str)

                    if current_frame_index != pred_frame_index and current_frame_index != blank_idx:
                        add_score = lm_add.score(" ".join((list(last.entries[label].context) + [current_frame_char])[-2*add_ngram+1:])) * math.log(10)
                        sub_score = lm_sub.score(" ".join((list(last.entries[label].context) + [current_frame_char])[-2*sub_ngram+1:])) * math.log(10)
                        new_score = last.entries[label].SCORE + logits[t, current_frame_index] + add_weight * add_score - sub_weight * sub_score
                        new_label = label + (current_frame_char,)
                        if new_label not in curr.entries:
                            curr.entries[new_label] = BeamEntry()
                            curr.entries[new_label].label_out = new_label
                            curr.entries[new_label].SCORE = new_score
                            curr.entries[new_label].context = last.entries[label].context + (current_frame_char,)
                    else:
                        new_score = last.entries[label].SCORE + logits[t, current_frame_index]
                        new_label = label + (current_frame_char,)
                        if new_label not in curr.entries:
                            curr.entries[new_label] = BeamEntry()
                            curr.entries[new_label].label_out = new_label
                            curr.entries[new_label].SCORE = new_score
                            curr.entries[new_label].context = last.entries[label].context
            last = curr
        last.normalize()
        batch_new_scores.append("".join(last.entries[last.sort()[0]].context[max_ngram:]))

    return batch_new_scores
