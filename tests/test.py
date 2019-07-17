#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import predictor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pdb
import traceback
import pickle
import logging
import multiprocessing
from functools import partial
import os,sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

from embedding import embedding
from encoder import encoder
from utils.data_utils import *


class Test(object):
    def __init__(self, conf, **kwargs):
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.zdy = {}
        #init embedding
        self.init_embedding()
        #load train data
        csv = pd.read_csv(self.ori_path, header = 0, sep="\t", error_bad_lines=False)
        if 'text' in csv.keys() and 'target' in csv.keys():
            #format: text \t target
            #for this format, the size for each class should be larger than 2 
            self.text_list = list(csv['text'])
            self.label_list = list(csv['target'])
        elif 'text_a' in csv.keys() and 'text_b' in csv.keys() and'target' in csv.keys():
            #format: text_a \t text_b \t target
            #for this format, target value can only be choosen from 0 or 1
            self.text_a_list = list(csv['text_a'])
            self.text_b_list = list(csv['text_b'])
            self.text_list = self.text_a_list + self.text_b_list
            self.label_list = list(csv['target'])

        subdirs = [os.path.join(self.export_dir_path,x) for x in os.listdir(self.export_dir_path)
                if 'temp' not in(x)]

        latest = str(sorted(subdirs)[-1])
        self.predict_fn = predictor.from_saved_model(latest)

    def init_embedding(self):
        self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.dict_path,
                                            mode = 'test')
        self.text2id = partial(embedding[self.embedding_type].text2id,
                               vocab_dict = self.vocab_dict,
                               maxlen = self.maxlen,
                               )

