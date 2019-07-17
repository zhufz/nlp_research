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


class TestMatch(Test):
    def __init__(self, conf, **kwargs):
        super(TestMatch, self).__init__(conf, **kwargs)
        conf.update({
            "keep_prob": 1,
            "is_training": False
        })
        self.encoder = encoder[conf['encoder_type']](**conf)
        if self.sim_mode == 'represent':
            #represent模式，预先缓存所有训练语料的encode结果
            self.vec_list = self._get_vecs(self.text_list, True)

    def test_file(self, text):
        raise ValueError('no implemented')

    def __call__(self, text):
        if self.tfrecords_mode == 'point':
            assert text.find('||') != -1,"input should cotain two sentences seperated by ||"
            text_a = text.split('||')[0]
            text_b = text.split('||')[-1]
            pred,score = self._get_label([text_a], [text_b], need_preprocess = True)
            return pred[0][0], score[0][0]

        #加载自定义问句(自定义优先)
        if self.sim_mode == 'cross':
            text_list = self.text_list
            label_list = self.label_list
            if self.zdy != {}:
                text_list = self.zdy['text_list'] + text_list
                label_list = self.zdy['label_list'] + label_list
            pred,score = self._get_label([text], self.text_list, need_preprocess = True)
            selected_id = np.argmax(score)
            out_score = score[selected_id]
        elif self.sim_mode == 'represent':
            text_list = self.text_list
            vec_list = self.vec_list
            label_list = self.label_list
            if self.zdy != {}:
                text_list = self.zdy['text_list'] + text_list
                vec_list = np.concatenate([self.zdy['vec_list'], self.vec_list], axis = 0)
                label_list = self.zdy['label_list'] + label_list
            vec = self._get_vecs([text], need_preprocess = True)
            if self.is_distance:
                scores = euclidean_distances(vec, vec_list)[0]
                selected_id = np.argmin(scores)
                out_score = 1 - scores[selected_id]
            else:
                scores = cosine_similarity(vec, vec_list)[0]
                selected_id = np.argmax(scores)
                out_score = scores[selected_id]
        else:
            raise ValueError('unknown sim mode, represent or cross?')
        ret = (label_list[selected_id], out_score, selected_id, \
               self.text_list[selected_id])
        return ret

    def set_zdy_labels(self, text_list, label_list):
        if len(text_list) == 0 or len(label_list) == 0: 
            self.zdy = {}
            return
        self.zdy['text_list'] = text_list
        self.zdy['vec_list'] = self._get_vecs(text_list,
                                              need_preprocess = True)
        self.zdy['label_list'] = label_list

    def _get_vecs(self, text_list, need_preprocess = False):
        #根据batches数据生成向量
        text_list_pred, x_query, x_query_length = self.text2id(text_list,
                                                               need_preprocess = True)
        label = [0 for _ in range(len(text_list))]
        input_dict = {'x_query': x_query, 
                      'x_query_raw': text_list_pred,
                      'x_query_length': x_query_length, 
                      'label': label}
        input_dict.update(self.encoder.encoder_fun(**input_dict))
        del input_dict['x_query_raw']
        predictions = self.predict_fn(input_dict)
        return predictions['encode']

    def _get_label(self, query_list, sample_list, need_preprocess = False):
        #计算query_list 与 sample_list的匹配分数
        x_query_pred, x_query, x_query_length = self.text2id(query_list,
                                                             need_preprocess = True)
        x_sample_pred, x_sample, x_sample_length = self.text2id(sample_list,
                                                                need_preprocess = True)
        label = [0 for _ in range(len(sample_list))]
        if len(x_query) != len(x_sample):
            x_query = np.tile(x_query[0],(len(x_sample),1))
            x_query_raw = np.tile(x_query_raw[0],(len(x_sample),1))
            x_query_length = np.tile(x_query_length[0],(len(x_sample),))
        input_dict = {'x_query': x_query, 
                      'x_query_raw': x_query_pred,
                      'x_query_length': x_query_length, 
                      'x_sample': x_sample,
                      'x_sample_raw': x_sample_pred,
                      'x_sample_length': x_sample_length, 
                      'label': label}
        input_dict.update(self.encoder.encoder_fun(**input_dict))
        del input_dict['x_query_raw']
        del input_dict['x_sample_raw']
        predictions = self.predict_fn(input_dict)
        return predictions['pred'], predictions['score']


