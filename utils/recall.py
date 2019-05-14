import numpy as np
import pandas as pd
import numpy as np
import os
import pickle
import random
import pdb
from collections import defaultdict
from annoy import AnnoyIndex
from gensim import corpora,models,similarities
import sys,os

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

class Recall():
    #基于similarities.SparseMatrixSimilarity
    def __init__(self, data_list):
        data_list = self._check(data_list)
        self.dictionary = corpora.Dictionary(data_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in data_list]
        self.tfidf = models.TfidfModel(corpus) #文档建tfidf模型

    def _check(self, data_list):
        assert type(data_list) == list and len(data_list)>0,'type error or empty for data_list'
        if type(data_list[0]) != list:
            return [data.split() for data in data_list]
        return data_list

    def _init_query(self, query_list):
        query_list = self._check(query_list)
        #相似度模型
        corpus = [self.dictionary.doc2bow(doc) for doc in query_list]
        self.index = similarities.SparseMatrixSimilarity(self.tfidf[corpus],
                                                         num_features=len(self.dictionary.keys()))

    def _cal_similarity(self, text):
        if type(text) == str:
            text = text.split()
        doc_test_vec = self.dictionary.doc2bow(text)
        text_tfidf = self.tfidf[doc_test_vec]
        sim = self.index[text_tfidf]
        return sim

    def __call__(self, data, query_id_list, text_id, num, reverse = False):
        query_list = [data[idx] for idx in query_id_list]
        text = data[text_id]
        self._init_query(query_list)
        sim = self._cal_similarity(text)
        if reverse == True:
            sorted_idx = np.argsort(-np.array(sim))
        else:
            sorted_idx = np.argsort(np.array(sim))
        query_id_list = np.array(query_id_list)[sorted_idx][:num]
        return query_id_list

class Recall2():
    #基于倒排索引
    def __init__(self, data_list):
        data_list = self._check(data_list)
        self.dictionary = corpora.Dictionary(data_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in data_list]
        self.tfidf = models.TfidfModel(corpus) #文档建tfidf模型

    def _check(self, data_list):
        assert type(data_list) == list and len(data_list)>0,'type error or empty for data_list'
        if type(data_list[0]) != list:
            return [data.split() for data in data_list]
        return data_list

    def create_inverted_index(self, data_list):
        inverted_index = defaultdict(set)
        for idx, word_list in enumerate(data_list):
            for word in word_list:
                inverted_index[word].add(idx)
        return inverted_index

    def __call__(self, data, query_id_list, text_id, num, **kwargs):
        query_list = [data[idx].split() for idx in query_id_list]
        text = data[text_id].split()
        inverted_index = self.create_inverted_index(query_list)
        doc_test_vec = self.dictionary.doc2bow(text)
        text_tfidf = [item[1] for item in self.tfidf[doc_test_vec]]
        sorted_id = np.argsort(-np.array(text_tfidf))

        ret = set()
        for idx in sorted_id:
            word = text[idx]
            if word in inverted_index:
                ret.update([query_id_list[idx] for idx in inverted_index[word]])
            if len(ret) > num: break
        ret = (list(ret))[:num]
        if len(ret) < num:
            add_num = num - len(ret)
            if add_num <= len(query_id_list):
                ret += random.sample(query_id_list, add_num)
            else:
                ret += query_id_list
        return ret

class Recall3():
    #基于Annoy
    def __init__(self, vecs):
        assert len(vecs)>0, 'no vecs available to init AnnoyIndex'
        size = len(vecs[0])
        self.annoy_model = AnnoyIndex(size)
        for idx,vec in enumerate(vecs):
            self.annoy_model.add_item(idx, vec)
        self.annoy_model.build(50)

    def __call__(self, vec):
        return self.annoy_model.get_nns_by_vector(vec, 20, include_distances =
                                                  True)
