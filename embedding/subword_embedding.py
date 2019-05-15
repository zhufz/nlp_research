import gensim
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
import numpy as np
from itertools import chain
import tensorflow as tf
from utils.preprocess import *
from common.layers import get_initializer
import collections
import pickle
import pandas as pd
import pdb

MAX_SUB_LEN = 8


class SubwordEmbedding():
    def __init__(self, text_list, dict_path, vocab_dict, random = False,\
                 maxlen = 20, embedding_size = 128, **kwargs):
        self.embedding_path = kwargs['conf']['subword_embedding_path']
        self.maxlen = maxlen
        self.dict_path = dict_path
        self.size = embedding_size
        self.embedding = tf.get_variable("embeddings",
                                         shape = [len(vocab_dict), self.size],
                                         initializer=get_initializer('xavier'),
                                         trainable = True)
        self.batch_size = kwargs['batch_size']
        self.indices = {}
        self.values = {}
        self.input_ids = {}


    def __call__(self, name = "subword_embedding"):
        """define placeholder"""
        self.indices[name] = tf.placeholder(dtype=tf.int64, shape=[None,2], name
                                            = name+"_indices")
        self.values[name] = tf.placeholder(dtype=tf.int64, shape=[None], name
                                            = name+"_values")

        self.input_ids[name] = tf.SparseTensor(indices=self.indices[name],
                                               values=self.values[name],
                                               dense_shape = (self.batch_size*self.maxlen,MAX_SUB_LEN))

        embed =  tf.nn.embedding_lookup_sparse(self.embedding,\
                                             self.input_ids[name],
                                             sp_weights = None,
                                             combiner="mean")
        return tf.reshape(embed, [-1, self.maxlen, self.size])


    def feed_dict(self, input_x, name = 'subword_embedding'):
        feed_dict = {}
        indices, values = zip(*input_x)
        feed_dict[self.indices[name]] = indices
        feed_dict[self.values[name]] = values
        return feed_dict

    def pb_feed_dict(self, graph, input_x, name = 'subword_embedding'):
        feed_dict = {}
        indices_node = graph.get_operation_by_name(name + "_indices").outputs[0]
        values_node = graph.get_operation_by_name(name + "_values").outputs[0]
        indices, values = zip(*input_x)
        feed_dict[indices_node] = indices
        feed_dict[values_node] = values
        return feed_dict

    @staticmethod
    def build_dict(dict_path, text_list = None,  mode = "train"):
        if not os.path.exists(dict_path) or mode == "train":
            assert text_list != None, "text_list can't be None in train mode"
            words = list()
            for content in text_list:
                for word in word_tokenize(clean_str(content)):
                    words.append(word)
            new_list = []
            for item in words:
                new_list.append(item)
                #对于空word或者带有尖括号特殊意义的不拆分
                if item == "" or (item[0] == '<' and item[-1] == '>'): continue
                item = "<"+item+">"
                for idx in range(0, len(item) -1 ):
                    if idx >= MAX_SUB_LEN: break
                    new_list.append(item[idx:idx+2])
            word_counter = collections.Counter(new_list).most_common()
            vocab_dict = dict()
            vocab_dict["<pad>"] = 0
            vocab_dict["<unk>"] = 1
            for word, _ in word_counter:
                vocab_dict[word] = len(vocab_dict)

            with open(dict_path, "wb") as f:
                pickle.dump(vocab_dict, f)
        else:
            with open(dict_path, "rb") as f:
                vocab_dict = pickle.load(f)

        return vocab_dict

    def words2indices(self, word_list, vocab_dict, index, maxlen):
        values_list = []
        indices_list = []
        start_x = index
        for idx in range(maxlen):
            if idx < len(word_list):
                item = word_list[idx]
                values_list.append(vocab_dict.get(item, vocab_dict['<unk>']))
                indices_list.append([start_x, 0])
                if (item[0] == '<' and item[-1] == '>') or item == "":
                    start_x += 1
                    continue
                item = "<"+item+">"
                start_y = 1

                for idy in range(0, len(item) -1):
                    if idy >= MAX_SUB_LEN: break
                    indices_list.append([start_x, start_y])
                    start_y += 1
                    values_list.append(vocab_dict.get(item[idy:idy+2],
                                                      vocab_dict['<unk>']))
                start_x += 1
            else:
                values_list.append(vocab_dict['<pad>'])
                indices_list.append([start_x, 0])
                start_x += 1
        return values_list, indices_list, start_x

    def text2id(self, text_list, vocab_dict,  need_preprocess = True):
        """
        文本id化
        """
        if need_preprocess:
            pre = Preprocess()
            text_list = [pre.get_dl_input_by_text(text) for text in text_list]
        x = list(map(lambda d: word_tokenize(clean_str(d)), text_list))
        x_len = [len(text) for text in x]
        values_list, indices_list = [],[]

        start_x = 0
        for idx in range(len(x)):
            values, indices, start_x = self.words2indices(x[idx], vocab_dict,
                                                             start_x,
                                                             self.maxlen)
            values_list += values
            indices_list += indices
        return text_list, zip(indices_list, values_list), x_len

if __name__ == '__main__':
    embedding = SubwordEmbedding()
