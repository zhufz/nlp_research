#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import predictor
import tensorflow.contrib.legacy_seq2seq as seq2seq
import pdb
import re
import traceback
import pickle
import logging
import multiprocessing
import os,sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

from encoder import encoder
from utils.data_utils import *
from utils.preprocess import Preprocess
from task_base import TaskBase


class Translation(TaskBase):
    def __init__(self, conf):
        super(Translation, self).__init__(conf)
        self.task_type = 'translation'
        self.conf = conf
        self.read_data()
        self.conf.update({
            "maxlen": self.maxlen,
            "maxlen1": self.maxlen,
            "maxlen2": self.maxlen,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "keep_prob": 1,
            "is_training": False,
        })
        self.encoder = encoder[self.encoder_type](**self.conf)

    def read_data(self):
        self.pre = Preprocess()
        encode_list, decode_list, target_list =\
            load_chat_data(self.ori_path)
        self.text_list = encode_list + decode_list
        self.label_list = target_list
        self.data_type = 'translation'

    def create_model_fn(self):
        def cal_loss(out, labels):
            with tf.name_scope("loss"):
                labels = tf.reshape(labels, [-1])
                loss = seq2seq.sequence_loss_by_example([out],
                                                    [labels],
                                                    [tf.ones_like(labels, dtype=tf.float32)])
                loss = tf.reduce_mean(loss)
            return loss

        def model_fn(features, labels, mode, params):
            #model params
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.encoder.keep_prob = 0.7
                self.encoder.is_training = True
            else:
                self.encoder.keep_prob = 1
                self.encoder.is_training = False

            global_step = tf.train.get_or_create_global_step()
            #############  encoder  #################

            if not self.use_language_model:
                self.embedding, vocab_dict  = self.init_embedding()
                self.num_class = len(vocab_dict)
                self.encoder.num_output = self.num_class
                self.embed_encode = self.embedding(features = features,name =
                                                   'seq_encode')
                self.embed_decode = self.embedding(features = features,name =
                                                   'seq_decode')
                out, self.final_state_encode, self.final_state_decode, \
                    pb_nodes  = self.encoder(self.embed_encode,
                                             self.embed_decode,
                                             features = features,
                                             name = 'seq')
            else:
                out = self.encoder(features = features)

            out = tf.reshape(out, [-1, self.num_class])


            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': tf.nn.softmax(out),
                    #'logit': pred,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            ############### loss ##################
            loss = cal_loss(out, labels) 
            with tf.name_scope("output"):
                out = tf.nn.softmax(out)
                self.prob = tf.reshape(out,[-1, self.maxlen, self.num_class], name = 'prob')
                out_max = tf.argmax(self.prob,-1, output_type = tf.int64)
                self.predictions = tf.reshape(out_max, [-1, self.maxlen], name = 'predictions')

            with tf.name_scope("accuracy"):
                labels = tf.cast(labels, tf.int64)
                correct_predictions = tf.equal(self.predictions, labels)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            ############### train ##################
            if mode == tf.estimator.ModeKeys.TRAIN:
                return self.train_estimator_spec(mode, loss, global_step, params)

            ############### eval ##################
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {"accuracy": 
                                   tf.metrics.accuracy(labels=labels, 
                                                       predictions=self.predictions)}
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        return model_fn

    def create_input_fn(self, mode):
        n_cpu = multiprocessing.cpu_count()
        def train_input_fn():
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith('train')]
            if len(filenames) == 0:
                logging.warn("Can't find any tfrecords file for train, prepare now!")
                self.prepare()
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

    def train(self):
        estimator = self.get_train_estimator(self.create_model_fn(), None)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps =
                        self.max_steps)
        #self.save()

    def test(self, mode = 'test'):
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config)
        if mode == 'dev':
            estimator.evaluate(input_fn=self.create_input_fn('dev'))
        elif mode == 'test':
            estimator.evaluate(input_fn=self.create_input_fn('test'))
        else:
            raise ValueError("unknown mode:[%s]"%mode)

    def save(self):
        def get_features():
            features = {'seq_encode': tf.placeholder(dtype=tf.int64, 
                                                  shape=[None, self.maxlen],
                                                  name='encode'),
                        'seq_encode_length': tf.placeholder(dtype=tf.int64,
                                                         shape=[None],
                                                         name='seq_encode_length')}
            features.update(self.encoder.features)
            return features
        self.save_model(self.create_model_fn(), None, get_features)
