#refer: https://github.com/xionghhcs/chip2018/blob/c2bb9efc08eca521a9ef5d37d4b915fb4c2a69dc/src/feature_extractor.py
from nltk import ngrams
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors
import gensim
from gensim.models import KeyedVectors
from collections import Counter
import sys
import time
import datetime
import copy
import pdb

from gensim.summarization.bm25 import get_bm25_weights, BM25
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_similarity_score

class Similarity():
    def __init__(self, corpus = None):
        self.corpus = corpus
        if self.corpus != None:
            self.tfidf_vectorizer = self.get_tfidf_vectorizer(self.corpus)
            self.corpus_vec = self.tfidf_vectorizer.transform(self.corpus)
            self.bm25_model = BM25([s.split() for s in corpus])
            self.average_idf = sum(map(lambda k: float(self.bm25_model.idf[k]),
                                       self.bm25_model.idf.keys())) / len(self.bm25_model.idf.keys())

    def get_tfidf_vectorizer(self, corpus):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)
        return tfidf_vectorizer

    def get_vector(self, query):
        vec = self.tfidf_vectorizer.transform([query])
        return vec[0]

    def similarity(self, query, type):
        assert self.corpus != None, "self.corpus can't be None"
        ret = []
        if type == 'cosine':
            query = self.get_vector(query)
            for item in self.corpus_vec:
                sim = cosine_similarity(item, query)
                ret.append(sim[0][0])
        elif type == 'manhattan':
            query = self.get_vector(query)
            for item in self.corpus_vec:
                sim = manhattan_distances(item, query)
                ret.append(sim[0][0])
        elif type == 'euclidean':
            query = self.get_vector(query)
            for item in self.corpus_vec:
                sim = euclidean_distances (item, query)
                ret.append(sim[0][0])
        #elif type == 'jaccard':
        #    #query = query.split()
        #    query = self.get_vector(query)
        #    for item in self.corpus_vec:
        #        pdb.set_trace()
        #        sim = jaccard_similarity_score(item, query)
        #        ret.append(sim)
        elif type == 'bm25':
            query = query.split()
            ret = self.bm25_model.get_scores(query)
        else:
            raise ValueError(f'similarity type error:{type}')
        return ret

if __name__ == '__main__':
    corpus = ['帮我 打开 灯','打开 空调', '关闭 空调','关灯','音量 调高','声音 调高']
    sim = Similarity(corpus)
    print(sim.similarity('打开 灯','cosine'))
    print(sim.similarity('打开 灯','manhattan'))
    print(sim.similarity('打开 灯','euclidean'))
    print(sim.similarity('打开 灯','bm25'))


