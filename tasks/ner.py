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
from utils.tf_utils import get_placeholder_batch_size
from embedding import embedding
from encoder import encoder

from utils.data_utils import *
from utils.ner_util import NERUtil
from task_base import TaskBase
import pdb

class NER(TaskBase):
    def __init__(self, conf):
        super(NER, self).__init__(conf)
        self.task_type = 'ner'
        self.conf = conf
        self.read_data()
        if self.maxlen == -1:
            self.maxlen = max([len(text.split()) for text in self.text_list])
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

        #params['num_output'] = 128
        #self.encoder_base = encoder['transformer'](**params)
        #params['num_output'] = self.num_class
        self.encoder = encoder[self.encoder_type](**params)


    def read_data(self):
        self.pre = Preprocess()
        self.util = NERUtil()
        self.text_list, self.label_list = self.util.load_ner_data(self.ori_path)
        self.text_list = [self.pre.get_dl_input_by_text(text, self.use_generalization) for text in self.text_list]
        self.num_class = self.num_output = len(set(list(chain.from_iterable(self.label_list))))
        self.data_type = 'column_2'

    def create_model_fn(self):
        def model_fn(features, labels, mode, params):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.encoder.keep_prob = 0.5
                self.encoder.is_training = True
            else:
                self.encoder.keep_prob = 1
                self.encoder.is_training = False

            seq_len = features['x_query_length']
            global_step = tf.train.get_or_create_global_step()

            ################ encode ##################
            if not self.use_language_model:
                self.embedding, _ = self.init_embedding()
                embed = self.embedding(features = features, name = 'x_query')
                out = self.encoder(embed, 'x_query', features = features, middle_flag = True)
                #out = self.encoder_base(embed, 'x_query', features = features, middle_flag = True)
                #out = self.encoder(out, 'x_query', features = features, middle_flag = True)
            else:
                out = self.encoder(features = features)

            logits = tf.reshape(out, [-1, int(out.shape[1]), self.num_class])

            batch_size = get_placeholder_batch_size(logits)
            small = -1000
            start_logits = tf.concat([
                small*tf.ones(shape=[batch_size, 1, self.num_class]), 
                tf.zeros(shape=[batch_size, 1, 1])],
                                     axis=-1)
            pad_logits = tf.cast(small * tf.ones(shape=[batch_size, self.maxlen,
                                                        1]), tf.float32)
            logits = tf.concat([logits, pad_logits], axis = -1)
            logits = tf.concat([start_logits, logits], axis = 1)
            seq_len += 1
            transition_params = tf.get_variable('crf', 
                                         [self.num_class + 1,self.num_class + 1], 
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
                labels = tf.concat([
                    tf.cast(self.num_class * tf.ones(shape=[batch_size, 1]), tf.int64), 
                    labels
                ], axis = -1)
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, 
                                                                      labels,
                                                                      seq_len,
                                                                      transition_params)
                loss = -tf.reduce_mean(log_likelihood)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    return self.train_estimator_spec(mode, loss, global_step, params)
                if mode == tf.estimator.ModeKeys.EVAL:
                    weights = tf.sequence_mask(seq_len, self.maxlen+1)
                    metrics = {'acc': tf.metrics.accuracy(labels, pred_ids, weights)}
                    return tf.estimator.EstimatorSpec(mode, 
                                                      loss=loss, 
                                                      eval_metric_ops=metrics)
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
            dataset = dataset.shuffle(buffer_size=100*self.batch_size)
            dataset = dataset.prefetch(4*self.batch_size)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            return features, label

        def test_input_fn(mode):
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith(mode)]
            assert len(filenames) > 0, "Can't find any tfrecords file for %s!"%mode
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

    def save(self):
        def get_features():
            features = {'x_query': tf.placeholder(dtype=tf.int64, 
                                                  shape=[None, self.maxlen],
                                                  name='x_query'),
                        'x_query_length': tf.placeholder(dtype=tf.int64,
                                                         shape=[None],
                                                         name='x_query_length'),
                        }
            features.update(self.encoder.get_features())
            return features
        self.save_model(self.create_model_fn(), None, get_features)

    def train(self):
        estimator = self.get_train_estimator(self.create_model_fn(), None)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps =
                        self.max_steps)
        self.save()

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

    def train_and_evaluate(self):
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path,
                                        save_checkpoints_steps=self.save_interval,
                                        keep_checkpoint_max=5)

        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config)

        early_stop = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name="loss",
            max_steps_without_decrease=estimator.config.save_checkpoints_steps * 2,
            run_every_secs=None,
            run_every_steps=estimator.config.save_checkpoints_steps,
        )

        train_spec=tf.estimator.TrainSpec(
                             input_fn = self.create_input_fn("train"), 
                             max_steps = self.max_steps,
                             hooks=[early_stop])

        eval_spec=tf.estimator.EvalSpec(
                             input_fn = self.create_input_fn("dev"),
                             steps = None,
                             start_delay_secs = 1, # start evaluating after N seconds
                             throttle_secs = 10,  # evaluate every N seconds
                             )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        self.save()

