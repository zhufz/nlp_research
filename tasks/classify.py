#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import predictor
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
from common.loss import get_loss
from task_base import TaskBase


class Classify(TaskBase):
    def __init__(self, conf):
        super(Classify, self).__init__(conf)
        self.task_type = 'classify'
        self.conf = conf
        self.read_data()
        self.num_class = len(set(self.label_list))
        self.num_output = self.num_class
        logging.info(">>>>>>>>>>>> class num:%s <<<<<<<<<<<<<<<"%self.num_class)
        self.conf.update({
            "maxlen": self.maxlen,
            "maxlen1": self.maxlen,
            "maxlen2": self.maxlen,
            "num_class": self.num_class,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "num_output": self.num_output,
            "keep_prob": 1,
            "is_training": False,
        })
        self.encoder = encoder[self.encoder_type](**self.conf)

    def read_data(self):
        self.pre = Preprocess()
        csv = pd.read_csv(self.ori_path, header = 0, sep=",", error_bad_lines=False)
        self.text_list = list(csv['text'])
        self.label_list = list(csv['target'])
        for idx,text in enumerate(self.text_list):
            self.text_list[idx] = self.pre.get_dl_input_by_text(text)
            if len(self.text_list[idx]) == 0:
                logging.error("find blank lines in %s"%idx)
        self.data_type = 'column_2'

    def create_model_fn(self):
        def cal_loss(pred, labels, batch_size, conf):
            loss = get_loss(type = self.loss_type, logits = pred, labels =
                                    labels, labels_sparse = True, **conf)
            return loss

        def model_fn(features, labels, mode, params):
            #model params
            self.encoder.keep_prob = params['keep_prob']
            self.encoder.is_training = params['is_training']
            global_step = tf.train.get_or_create_global_step()

            #############  encoder  #################
            if not self.use_language_model:
                self.embedding,_ = self.init_embedding()
                self.embed_query = self.embedding(features = features, name = 'x_query')
                out = self.encoder(self.embed_query, 
                                    name = 'x_query',
                                    features = features)
            else:
                out = self.encoder(features = features)
            #pred = tf.nn.softmax(tf.layers.dense(out, self.num_class))
            pred = tf.nn.softmax(out)
            pred_labels = tf.argmax(pred, axis=-1)

            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'encode': out,
                    'logit': pred,
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            ############### loss ##################
            loss = cal_loss(pred, labels, self.batch_size, self.conf) 

            ############### train ##################
            if mode == tf.estimator.ModeKeys.TRAIN:
                return self.train_estimator_spec(mode, loss, global_step, params)

            ############### eval ##################
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {"accuracy": 
                                   tf.metrics.accuracy(labels=labels, 
                                                       predictions=pred_labels)}
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        return model_fn

    def create_input_fn(self, mode):
        n_cpu = multiprocessing.cpu_count()
        def train_input_fn():
            size = self.num_class
            num_classes_per_batch = self.num_class_per_batch
            assert num_classes_per_batch <= self.num_class, \
                "num_classes_per_batch is %s > %s"%(num_classes_per_batch, self.num_class)
            num_sentences_per_class = self.batch_size // num_classes_per_batch

            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith('train')]
            assert len(filenames) > 0, "Can't find any tfrecords file for train!"
            logging.info("tfrecords train class num: {}".format(len(filenames)))
            datasets = [tf.data.TFRecordDataset(filename) for filename in filenames]
            datasets = [dataset.repeat() for dataset in datasets]
            #assert self.batch_size == num_sentences_per_class* num_classes_per_batch
            def generator():
                while True:
                    labels = np.random.choice(range(size),
                                              num_classes_per_batch,
                                              replace=False)
                    for label in labels:
                        for _ in range(num_sentences_per_class):
                            yield label

            choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
            dataset = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)
            gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
            dataset = dataset.map(lambda record: gt.parse_record(record, self.encoder),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(4*self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            #test
            #sess = tf.Session()
            #features,label = sess.run([features,label])
            #features['x_query_pred'] = [item.decode('utf-8') for item in
            #                           features['x_query_pred'][1]]
            return features, label

        def test_input_fn(mode):
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith(mode)]
            assert self.num_class == len(filenames), "the num of tfrecords file error!"
            logging.info("tfrecords test class num: {}".format(len(filenames)))
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
        params = {
            'is_training': True,
            'keep_prob': 0.7
        }
        estimator = self.get_train_estimator(self.create_model_fn(), params)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps =
                        self.max_steps)
        self.save()

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

    def save(self):
        params = {
            'is_training': False,
            'keep_prob': 1
        }
        def get_features():
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
            return features
        self.save_model(self.create_model_fn(), params, get_features)
