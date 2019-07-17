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
from tests.test import Test


class TestNER(Test):
    def __init__(self, conf, **kwargs):
        super(TestNER, self).__init__(conf, **kwargs)
        conf.update({
            "keep_prob": 1,
            "is_training": False
        })
        self.encoder = encoder[conf['encoder_type']](**conf)
        self.mp_label = pickle.load(open(self.label_path, 'rb'))
        self.mp_label_rev = {self.mp_label[item]:item for item in self.mp_label}


    def test(self, text_list):
        length = len(text_list[0])
        text_list_pred, x_query, x_query_length = self.text2id(text_list,
                                                               need_preprocess = False)
        input_dict = {'x_query': x_query, 
                      'x_query_length': x_query_length, 
                      }
        input_dict.update(self.encoder.encoder_fun(**input_dict))
        predictions = self.predict_fn(input_dict)
        pred_ids = [item for item in predictions['pred_ids']]
        for idx,item in enumerate(pred_ids):
            pred_ids[idx] = [self.mp_label_rev[id] for id in
                             pred_ids[idx] if id in self.mp_label_rev][:length]
        return pred_ids

    def test_file(self, file):
        with open(file) as f_in, open(file+'.out.txt','w') as f_out:
            lines = f_in.readlines()
            lines = [line.strip() for line in lines]
            #res = self.test(lines)
            res0 = self.test(lines[:1500])
            res1 = self.test(lines[1500:])
            res = res0 +res1
            for idx,line in enumerate(lines):
                pred = zip(line.split(),res[idx])
                for word,tag in pred:
                    f_out.write(word + "\t" + tag +'\n')
                f_out.write('\n')

    def __call__(self, text):
        text_list  = [text]
        return self.test(text_list)[0]
