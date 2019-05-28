import tensorflow as tf
from tensorflow.contrib import predictor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
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
from utils.data_utils import *
from utils.recall import Annoy


class Test(object):
    def __init__(self, conf, **kwargs):
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.zdy = {}
        #init embedding
        self.init_embedding()
        #load train data
        csv = pd.read_csv(self.ori_path, header = 0, sep=",", error_bad_lines=False)
        self.text_list = list(csv['text'])
        self.label_list = list(csv['target'])
        #load model
        subdirs = [x for x in Path(self.export_dir_path).iterdir()
                if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        self.predict_fn = predictor.from_saved_model(latest)

    def init_embedding(self):
        self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.dict_path,
                                            mode = 'test')
        self.text2id = partial(embedding[self.embedding_type].text2id,
                               vocab_dict = self.vocab_dict,
                               maxlen = self.maxlen,
                               need_preprocess = True)

class TestClassify(Test):
    def __init__(self, conf, **kwargs):
        super(TestClassify, self).__init__(conf, **kwargs)
        self.mp_label = pickle.load(open(self.label_path, 'rb'))
        self.mp_label_rev = {self.mp_label[item]:item for item in self.mp_label}

    def __call__(self, text):
        text_list  = [text]
        text_list_pred, x_query, x_query_length = self.text2id(text_list)
        label = [0 for _ in range(len(text_list))]

        predictions = self.predict_fn({'x_query': x_query, 
                                  'x_query_length': x_query_length, 
                                  'label': label})
        scores = [item for item in predictions['pred']]
        max_scores = np.max(scores, axis = -1)
        max_ids = np.argmax(scores, axis = -1)
        ret =  (max_ids[0], max_scores[0], self.mp_label_rev[max_ids[0]])
        return ret

class TestMatch(Test):
    def __init__(self, conf, **kwargs):
        super(TestMatch, self).__init__(conf, **kwargs)
        if self.mode == 'represent':
            #represent模式，预先缓存所有训练语料的encode结果
            self.vec_list = self._get_vecs(self.text_list, True)

    def __call__(self, text):

        #加载自定义问句(自定义优先)
        if self.sim_mode == 'cross':
            text_list = self.text_list
            label_list = self.label_list
            if self.zdy != {}:
                text_list = self.zdy['text_list'] + text_list
                label_list = self.zdy['label_list'] + label_list
            pred,score = self._get_label([text], self.text_list, need_preprocess = True)
            max_id = np.argmax(score[0])
            max_score = np.max(score[0])
        elif self.sim_mode == 'represent':
            text_list = self.text_list
            vec_list = self.vec_list
            label_list = self.label_list
            if self.zdy != {}:
                text_list = self.zdy['text_list'] + text_list
                vec_list = np.concatenate([self.zdy['vec_list'], self.vec_list], axis = 0)
                label_list = self.zdy['label_list'] + label_list
            vec = self._get_vecs([text], need_preprocess = True)
            scores = cosine_similarity(vec, vec_list)[0]
            max_id = np.argmax(scores)
            max_score = scores[max_id]
        else:
            raise ValueError('unknown sim mode, represent or cross?')
        max_similar = text_list[max_id]
        ret = (label_list[max_id], max_score, max_id)
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
        text_list_pred, x_query, x_query_length = self.text2id(text_list)
        label = [0 for _ in range(len(text_list))]

        predictions = self.predict_fn({'x_query': x_query, 
                                  'x_query_length': x_query_length, 
                                  'label': label})
        return predictions['encode']


    def _get_label(self, query_list, sample_list, need_preprocess = False):
        #根据batches数据生成向量
        _, x_query, x_query_length = self.text2id(query_list)
        _, x_sample, x_sample_length = self.text2id(sample_list)
        label = [0 for _ in range(len(sample_list))]
        x_query = np.tile(x_query[0],(len(x_sample),1))
        x_query_length = np.tile(x_query_length[0],(len(x_sample),))
        predictions = self.predict_fn({'x_query': x_query, 
                                  'x_query_length': x_query_length, 
                                  'x_sample': x_sample,
                                  'x_sample_length': x_sample_length, 
                                  'label': label})
        return predictions['pred'], predictions['score']


