#-*- coding:utf-8 -*-
import sys,os
import yaml
import time
import copy
import logging
import multiprocessing
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.preprocess import Preprocess
from embedding import embedding
from encoder import encoder
from utils.data_utils import *
from utils.ner_util import NERUtil
from utils.tf_utils import load_pb,write_pb
from common.layers import get_train_op
from language_model.bert.modeling import get_assignment_map_from_checkpoint
from task_base import TaskBase
import pdb

class NER(TaskBase):
    def __init__(self, conf):
        super(NER, self).__init__(conf)
        self.task_type = 'ner'
        self.conf = conf
        self.label2tag = {self.tag2label[item]:item for item in self.tag2label}
        pickle.dump(self.label2tag, open(self.label_path, 'wb'))
        self.shuffle = True

        self.pre = Preprocess()
        self.util = NERUtil()
        self.read_data()
        #if self.maxlen == -1:
        #    self.maxlen = max([len(text.split()) for text in self.text_list])
        #model params
        params = conf
        params.update({
            "maxlen":self.maxlen,
            "embedding_size":self.embedding_size,
            "batch_size": self.batch_size,
            "num_output": self.num_class,
            "keep_prob": 1,
            "is_training": False,
        })

        self.encoder = encoder[self.encoder_type](**params)

    def read_data(self):
        self.text_list, self.label_list = self.util.load_ner_data(self.ori_path)
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in self.text_list]
        self.trans_label_list(self.label_list, self.tag2label)

    def prepare(self):
        vocab_dict = embedding[self.embedding_type].build_dict(\
                                                               dict_path = self.dict_path,
                                                               text_list = self.text_list,
                                                               mode = self.mode)
        text2id = embedding[self.embedding_type].text2id
        self.gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
        self.gt.process(self.text_list, self.label_list, text2id,
                        self.encoder.encoder_fun, vocab_dict,
                        self.tfrecords_path, self.label_path, 
                        self.dev_size, mode = self.mode)
        logging.info("tfrecords generated!")

    def create_model_fn(self):
        def init_embedding():
            vocab_dict = embedding[self.embedding_type].build_dict(\
                                                                   dict_path = self.dict_path,
                                                                   text_list = self.text_list,
                                                                   mode = self.mode)
            return  embedding[self.embedding_type](text_list = self.text_list,
                                                   vocab_dict = vocab_dict,
                                                   dict_path = self.dict_path,
                                                   random=self.rand_embedding,
                                                   maxlen = self.maxlen,
                                                   batch_size = self.batch_size,
                                                   embedding_size = self.embedding_size,
                                                   conf = self.conf)

        def model_fn(features, labels, mode, params):
            self.encoder.keep_prob = params['keep_prob']
            self.encoder.is_training = params['is_training']
            seq_len = features['x_query_length']
            global_step = tf.train.get_or_create_global_step()

            ################ encode ##################
            if not self.use_language_model:
                embedding = init_embedding()
                embed = embedding(features = features, name = 'x_query')
                out = self.encoder(embed, 'x_query', features = features, middle_flag = True)
            else:
                out = self.encoder(features = features)
            logits = tf.reshape(out, [-1, int(out.shape[1]), self.num_class])

            transition_params = tf.get_variable('crf', 
                                         [self.num_class,self.num_class], 
                                         dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, seq_len)

            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'logit': logits,
                    'pred_ids': pred_ids,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            else:
                ############### loss ####################
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels,
                                                                      seq_len,
                                                                      transition_params)
                loss = -tf.reduce_mean(log_likelihood)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    return self.train_estimator_spec(mode, loss, global_step, params)
                if mode == tf.estimator.ModeKeys.EVAL:
                    #pdb.set_trace()
                    weights = tf.sequence_mask(seq_len, self.maxlen)
                    metrics = {'acc': tf.metrics.accuracy(labels, pred_ids, weights)}
                    #metrics = {'acc': tf.metrics.accuracy(labels, pred_ids)}
                    return tf.estimator.EstimatorSpec(mode, 
                                                      loss=loss, 
                                                      eval_metric_ops=metrics)
        return model_fn

    def train(self):
        params = {
            'is_training': True,
            'keep_prob': 0.7
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        if self.use_language_model:
            init_vars = tf.train.list_variables(self.init_checkpoint_path)
            init_vars_name = []
            for x in list(init_vars):
                (name, var) = (x[0], x[1])
                init_vars_name.append(name)
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.init_checkpoint_path,
                        vars_to_warm_start=init_vars_name)
            params.update({'base_var':init_vars_name})
        else:
            ws = None
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params,
                                           warm_start_from = ws)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps =
                        self.max_steps)
        self.save()

    def create_input_fn(self, mode):
        n_cpu = multiprocessing.cpu_count()
        def train_input_fn():
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith('train')]
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.repeat()
            gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
            dataset = dataset.map(lambda record: gt.parse_record(record, self.encoder),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(4*self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            return features, label

        def test_input_fn(mode):
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith(mode)]
            dataset = tf.data.TFRecordDataset(filenames)
            gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
            dataset = dataset.map(lambda record: gt.parse_record(record, self.encoder),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            return features, label

        if mode == 'train':
            return train_input_fn
        elif mode == 'test':
            return lambda : test_input_fn("test")
        elif mode == 'dev':
            return lambda : test_input_fn("dev")
        else:
            raise ValueError("unknown input_fn type!")


    def trans_label_list(self, label_list, tag2label):
        for idx,labels in enumerate(label_list):
            for idy,label in enumerate(labels):
                label_list[idx][idy] = tag2label[label_list[idx][idy]]

    def save(self):
        params = {
            'is_training': False,
            'keep_prob': 1
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params)
        def serving_input_receiver_fn():
            features = {'x_query': tf.placeholder(dtype=tf.int64, 
                                                  shape=[None, self.maxlen],
                                                  name='x_query'),
                        'x_query_length': tf.placeholder(dtype=tf.int64,
                                                         shape=[None],
                                                         name='x_query_length'),
                        'label': tf.placeholder(dtype=tf.int64, 
                                                shape=[None],
                                                name='label')}
            features.update(self.encoder.features)
            return tf.estimator.export.ServingInputReceiver(features, features)

        estimator.export_savedmodel(
            self.export_dir_path, # 目录
            serving_input_receiver_fn, # 返回ServingInputReceiver的函数
            assets_extra=None,
            as_text=False,
            checkpoint_path=None)

    def test(self, mode = 'test'):
        params = {
            'is_training': False,
            'keep_prob': 1
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params)
        if mode == 'dev':
            estimator.evaluate(input_fn=self.create_input_fn('dev'))
        elif mode == 'test':
            estimator.evaluate(input_fn=self.create_input_fn('test'))
        else:
            raise ValueError("unknown mode:[%s]"%mode)
