import gensim
import sys,os
import numpy as np
from itertools import chain
import tensorflow as tf
import pandas as pd
import collections
import pickle
import pdb

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.preprocess import *
from char_embedding import CharEmbedding
from word_embedding import WordEmbedding

class MixEmbedding():
    def __init__(self, text_list, dict_path, vocab_dict = None, random = False):
        self.char_embedding = CharEmbedding(text_list, dict_path, vocab_dict, random)
        self.word_embedding = WordEmbedding(text_list, dict_path, vocab_dict, random)

    def __call__(self, name = "mix_embedding"):
        """define placeholder"""
        return self.char_embeddibg('char_'+name)+self.word_embedding('word_'+name)

    @staticmethod
    def build_dict(dict_path, text_list = None, mode = "train"):
        vocab_dict = {}
        vocab_dict['char'] = CharEmbedding.build_dict(dict_path, text_lst, mode)
        vocab_dict['word'] = WordEmbedding.build_dict(dict_path, text_lst, mode)
        return vocab_dict

    def feed_dict(self, input_x, name = 'mix_embedding'):
        feed_dict = {}
        feed_dict.update(self.char_embedding.feed_dict(input_x))
        feed_dict.update(self.word_embedding.feed_dict(input_x))
        return feed_dict


    def text2id(self, text_list, vocab_dict, need_preprocess = True):
        """
        文本id化
        """
        text_list1, x1 = self.char_embedding.text2id(text_list, vocab_dict['char'], need_preprocess)
        text_list2, x2 = self.word_embedding.text2id(text_list, vocab_dict['word'], need_preprocess)
        res = {}
        res['char'] = (text_list1, x1)
        res['word'] = (text_list2, x2)
        return res

if __name__ == '__main__':
    pass
