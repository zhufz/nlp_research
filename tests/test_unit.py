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
    def __init__(self, conf):
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.model_loaded = False
        self.zdy = {}
        self.init_embedding()
        csv = pd.read_csv(self.ori_path, header = 0, sep=",", error_bad_lines=False)
        self.text_list = list(csv['text'])
        self.label_list = list(csv['target'])


    def init_embedding(self):
        self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.dict_path,
                                            mode = 'test')
        self.text2id = partial(embedding[self.embedding_type].text2id,
                               vocab_dict = self.vocab_dict,
                               maxlen = self.maxlen,
                               need_preprocess = True)

    def test_unit(self,text):
        if self.task_type == 'classify':
            self.test_classify(text)
        elif self.task_type == 'match':
            self.test_match(text)
        else:
            raise ValueError('unknown task type!')

    def test_classify(self, text):
        if self.model_loaded == False:
            self.init_embedding()
            subdirs = [x for x in Path(self.export_dir_path).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
            self.mp_label = pickle.load(open(self.label_path, 'rb'))
            self.mp_label_rev = {self.mp_label[item]:item for item in self.mp_label}
            self.model_loaded = True
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
        print(ret)
        return ret

    def test_match(self, text):
        #######################init#########################
        if self.model_loaded == False:
            #添加不参与训练样本
            if os.path.exists(self.no_train_path):
                csv = pd.read_csv(self.no_train_path, header = 0, sep=",", error_bad_lines=False)
                self.text_list += list(csv['text'])
                self.label_list += list(csv['target'])
            subdirs = [x for x in Path(self.export_dir_path).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
            self.init_embedding()
            self.model_loaded = True
            self.vec_list = self._get_vecs(self.predict_fn, self.text_list, True)
            #self.set_zdy_labels(['睡觉','我回家了','晚安','娃娃了','周杰伦','自然语言处理'],
            #                    ['打开情景模式','打开情景模式','打开情景模式',
            #                     '打开情景模式','打开情景模式','打开情景模式'])
        text_list = self.text_list
        vec_list = self.vec_list
        label_list = self.label_list

        #用于添加自定义问句(自定义优先)
        if self.zdy != {}:
            text_list = self.zdy['text_list'] + text_list
            vec_list = np.concatenate([self.zdy['vec_list'], self.vec_list], axis = 0)
            label_list = self.zdy['label_list'] + label_list
        vec = self._get_vecs(self.predict_fn, [text], need_preprocess = True)
        scores = cosine_similarity(vec, vec_list)[0]
        max_id = np.argmax(scores)
        max_score = scores[max_id]
        max_similar = text_list[max_id]
        logging.info("test result: {}, {}, {}".format(label_list[max_id], max_score, max_similar))
        return label_list[max_id], max_score, max_id

    def set_zdy_labels(self, text_list, label_list):
        if len(text_list) == 0 or len(label_list) == 0: 
            self.zdy = {}
            return
        self.zdy['text_list'] = text_list
        self.zdy['vec_list'] = self._get_vecs(self.predict_fn, 
                                              text_list,
                                              need_preprocess = True)
        self.zdy['label_list'] = label_list

    def _get_vecs(self, predict_fn, text_list, need_preprocess = False):
        #根据batches数据生成向量
        text_list_pred, x_query, x_query_length = self.text2id(text_list)
        label = [0 for _ in range(len(text_list))]

        predictions = predict_fn({'x_query': x_query, 
                                  'x_query_length': x_query_length, 
                                  'label': label})
        return predictions['encode']




