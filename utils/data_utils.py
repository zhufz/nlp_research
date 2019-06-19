#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import numpy as np
import os
import pickle
import random
import logging
from tqdm import tqdm
import pdb
from functools import partial
from itertools import chain
from collections import defaultdict
from gensim import corpora,models,similarities
import tensorflow as tf
import sys,os

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from common.similarity import Similarity
from utils.recall import OriginRecall, InvertRecall, Annoy
from utils.match_util import MatchUtil
from utils.ner_util import NERUtil

def load_class_mp(class_path):
    #load class mapping from class_path
    lines = [line.strip() for line in open(class_path).readlines()]
    mp = {}
    mp_rev = {}
    for idx,line in enumerate(lines):
        if line.strip() == '':continue
        mp[line.strip()] = idx
        mp_rev[idx] = line.strip()
    return mp, mp_rev

def generate_class_mp(label_list, class_path):
    #generate class mapping by label_list, and saved in "class_path"
    classes = set(label_list)
    class_mp = {}
    class_mp_rev = {}
    for idx,item in enumerate(classes):
        class_mp[item] = idx
        class_mp_rev[idx] = item

    with open(class_path,'w') as f_w:
        for idx in range(len(class_mp)):
            f_w.write("{}\n".format(class_mp_rev[idx]))
    return class_mp, class_mp_rev

def label2id(class_mp, label_list):
    #transfor text label to int label
    return [class_mp[item] for item in label_list]

def get_len(text_list):
    #get length for each text in text_list
    len_list = []
    for text in text_list:
        len_list.append(len(text))
    return len_list

def batch_iter(inputs, batch_size, num_epochs, shuffle = True):
    #generate iterator for inputs
    #inputs = np.array(list(inputs))
    inputs = list(inputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        if shuffle:
            random.shuffle(inputs)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index]

def load_classify_data(path):
    #load data for classify task
    df = pd.read_csv(path, header = 0)
    return df['text'],df['intent']

def load_seq2seq_data(path):
    #古诗
    x_texts = [line.strip() for line in open(path)]
    y_texts = []
    # 将标签整体往前移动一位， 代表当前对下一个的预测值
    for idx, text in enumerate(x_texts):
        if len(text) <1: 
            y_texts.append(x_texts[idx])
            continue
        y_texts.append(x_texts[idx][1:]+x_texts[idx][0])

    return x_texts, y_texts

def load_chat_data(path):
    texts = [line.strip().split('\t') for line in open(path)]
    encode_texts = []
    decode_texts = []
    target_texts = []
    for item in texts:
        if len(item) < 2: continue
        encode_texts.append(item[0])
        decode_texts.append("<s> "+item[1][1:])
        target_texts.append(item[1]+" </s>")
    return encode_texts, decode_texts, target_texts

class GenerateTfrecords(object):

    def __init__(self, mode, maxlen):
        self.tfrecords_mode = mode
        self.maxlen = maxlen
        self.match_util = MatchUtil()

    def _output_tfrecords(self, dataset, idx, path, mode):
        file_name = os.path.join(path, "{}_class_{:04d}".format(mode, idx))
        with tf.python_io.TFRecordWriter(file_name) as writer:
          for item in dataset:
            writer.write(item)

    def _serialized_example(self, **kwargs):
        def get_feature(kwargs):
            ret = {}
            for key in kwargs:
                if type(kwargs[key]) != str:
                    ret[key] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value = kwargs[key]))
                else:
                    ret[key] = tf.train.Feature(
                        bytes_list = tf.train.BytesList(
                            value = [tf.compat.as_bytes(kwargs[key])]))
            return ret
        """Helper for creating a serialized Example proto."""
        example = tf.train.Example(features=
                                   tf.train.Features(
                                       feature=get_feature(kwargs)))
        return example.SerializeToString()

    def parse_record(self, record, encoder):
        if self.tfrecords_mode == 'class':
            keys_to_features = {
                "x_query": tf.FixedLenFeature([self.maxlen], tf.int64),
                "x_query_length": tf.FixedLenFeature([1], tf.int64),
                "label": tf.FixedLenFeature([1], tf.int64),
            }
            keys_to_features.update(encoder.keys_to_features())
            parsed = tf.parse_single_example(record, keys_to_features)
            label = tf.reshape(parsed['label'], [1])
            ret =  {'x_query': tf.reshape(parsed['x_query'], [self.maxlen]),
                    'x_query_length': tf.reshape(parsed['x_query_length'], [1])[0],
                    'label': label[0]}
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label[0]
        elif self.tfrecords_mode in ['pair','point']:
            keys_to_features = {
                "x_query": tf.FixedLenFeature([self.maxlen], tf.int64),
                "x_sample": tf.FixedLenFeature([self.maxlen], tf.int64),
                "x_query_length": tf.FixedLenFeature([1], tf.int64),
                "x_sample_length": tf.FixedLenFeature([1], tf.int64),
                "label": tf.FixedLenFeature([1], tf.int64),
            }
            keys_to_features.update(encoder.keys_to_features())
            parsed = tf.parse_single_example(record, keys_to_features)
            # Perform additional preprocessing on the parsed data.
            label = tf.reshape(parsed['label'], [1])
            ret =  {'x_query': tf.reshape(parsed['x_query'], [self.maxlen]),
                    'x_sample': tf.reshape(parsed['x_sample'], [self.maxlen]),
                    'x_query_length': tf.reshape(parsed['x_query_length'], [1])[0],
                    'x_sample_length': tf.reshape(parsed['x_sample_length'], [1])[0],
                    'label': label[0]
                    } 
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label[0]
        elif self.tfrecords_mode == 'ner':
            keys_to_features = {
                "x_query": tf.FixedLenFeature([self.maxlen], tf.int64),
                "x_query_length": tf.FixedLenFeature([1], tf.int64),
                "label": tf.FixedLenFeature([self.maxlen], tf.int64),
            }
            keys_to_features.update(encoder.keys_to_features())
            parsed = tf.parse_single_example(record, keys_to_features)
            label = tf.reshape(parsed['label'], [self.maxlen])
            ret =  {'x_query': tf.reshape(parsed['x_query'], [self.maxlen]),
                    'x_query_length': tf.reshape(parsed['x_query_length'], [1])[0],
                    'label': label}
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label
        elif self.tfrecords_mode == 'translation':
            keys_to_features = {
                "encode": tf.FixedLenFeature([self.maxlen], tf.int64),
                "encode_length": tf.FixedLenFeature([1], tf.int64),
                "decode": tf.FixedLenFeature([self.maxlen], tf.int64),
                "decode_length": tf.FixedLenFeature([1], tf.int64),
                "target": tf.FixedLenFeature([self.maxlen], tf.int64),
                "target_length": tf.FixedLenFeature([1], tf.int64),
            }
            keys_to_features.update(encoder.keys_to_features())
            parsed = tf.parse_single_example(record, keys_to_features)
            label = tf.reshape(parsed['target'], [self.maxlen])
            ret =  {'seq_encode': tf.reshape(parsed['encode'], [self.maxlen]),
                    'seq_encode_length': tf.reshape(parsed['encode_length'], [1])[0],
                    'seq_decode': tf.reshape(parsed['decode'], [self.maxlen]),
                    'seq_decode_length': tf.reshape(parsed['decode_length'], [1])[0]}
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label
        else:
            raise ValueError('unknown tfrecords mode')


    def process_class_data(self, text_id_list, text_pred_list, len_list,
                           text_list, label_list, label_path,
                           encoder_fun, output_path, dev_size, mode, **kwargs):
        mp_label = {item:idx for idx,item in enumerate(list(set(label_list)))}
        pickle.dump(mp_label, open(label_path, 'wb'))

        mp_dataset = defaultdict(list)
        for idx,text_id in enumerate(tqdm(text_id_list, ncols = 70)):
            label = label_list[idx]
            input_dict = {'x_query': text_id, 
                          'x_query_pred': text_pred_list[idx], 
                          'x_query_raw': text_list[idx], 
                          'x_query_length': [len_list[idx]], 
                          'label': [mp_label[label]]
                          }
            input_dict.update(encoder_fun(**input_dict))
            del input_dict['x_query_pred']
            del input_dict['x_query_raw']
            serialized = self._serialized_example(**input_dict)
            mp_dataset[label].append(serialized)
        logging.info('generating tfrecords ...')
        train_num = 0
        dev_num = 0
        test_num = 0
        for label in mp_dataset:
            if mode == 'train':
                if dev_size >0 and dev_size <1: 
                    _dev_size = int(len(mp_dataset[label])*dev_size)
                else:
                    _dev_size = dev_size
                dataset_train = mp_dataset[label][:-_dev_size]
                dataset_test = mp_dataset[label][-_dev_size:]
                self._output_tfrecords(dataset_train, mp_label[label], output_path,
                                       "train")
                self._output_tfrecords(dataset_test, mp_label[label], output_path, 
                                       "dev")
                train_num += len(dataset_train)
                dev_num += len(dataset_test)
                #logging.info('training num [{}] for label [{}]'.\
                #             format(len(dataset_train), label))
                #logging.info('dev num [{}] for label [{}]'.format(_dev_size,
                #                                                   label))
            else:
                dataset = mp_dataset[label]
                self._output_tfrecords(dataset, mp_label[label], output_path,
                                       "test")
                test_num += len(dataset)
                #logging.info('testing num [{}] for label [{}]'.format(len(dataset),
                #                                                   label))
        if mode == 'train':
            logging.info('training num [{}]'.format(train_num))
            logging.info('dev num [{}]'.format(dev_num))
        else:
            logging.info('testing num [{}]'.format(test_num))

    def process_pair_data(self, text_id_list, text_pred_list, 
                          len_list, text_list, label_list, 
                          encoder_fun, output_path, dev_size, mode, **kwargs):
        def _generate_input_dict(text_id_list, text_pred_list, len_list,
                                encoder_fun, query_id, sample_id, label):
            input_dict = {'x_query': text_id_list[query_id], 
                          'x_sample': text_id_list[sample_id], 
                          'x_query_pred': text_pred_list[query_id],
                          'x_sample_pred': text_pred_list[sample_id],
                          'x_query_raw': text_list[query_id],
                          'x_sample_raw': text_list[sample_id],
                          'x_query_length': [len_list[query_id]],
                          'x_sample_length': [len_list[sample_id]],
                          'label': [label]}
            input_dict.update(encoder_fun(**input_dict))
            del input_dict['x_query_pred']
            del input_dict['x_query_raw']
            del input_dict['x_sample_pred']
            del input_dict['x_sample_raw']
            return input_dict
        input_dict_fun = partial(_generate_input_dict, 
                                 text_id_list, 
                                 text_pred_list, 
                                 len_list, 
                                 encoder_fun)
        if mode == 'train':
            train_list, test_list = self.match_util.get_pair_id(text_id_list, 
                                                     label_list,
                                                     dev_size)
            logging.info('generating tfrecords for training ...')
            for qid,query_id in enumerate(tqdm(train_list, ncols=70)):
                serial_list = []
                pos_list = train_list[query_id][1]
                neg_list = train_list[query_id][0]
                for idx in pos_list:
                    for idy in neg_list:
                        #generate <query,pos> and <query,neg> pairs
                        input_dict = input_dict_fun(query_id, idx, 1)
                        serialized_pos = self._serialized_example(**input_dict)
                        input_dict = input_dict_fun(query_id, idy, 1)
                        serialized_neg = self._serialized_example(**input_dict)
                        #append continuous pos and neg serialized sample
                        serial_list.append(serialized_pos)
                        serial_list.append(serialized_neg)
                self._output_tfrecords(serial_list, qid, output_path, "train")
            logging.info('generating tfrecords for testing ...')
            mp_dataset = defaultdict(list)
            for idx, (query_id, item_list) in enumerate(tqdm(test_list, ncols = 70)):
                serial_list = []
                for label, sample_id in item_list:
                    input_dict = {'x_query': text_id_list[query_id], 
                                  'x_sample': text_id_list[sample_id], 
                                  'x_query_pred': text_pred_list[query_id],
                                  'x_sample_pred': text_pred_list[sample_id],
                                  'x_query_raw': text_list[query_id],
                                  'x_sample_raw': text_list[sample_id],
                                  'x_query_length': [len_list[query_id]],
                                  'x_sample_length': [len_list[sample_id]],
                                  'label': [label]}
                    input_dict.update(encoder_fun(**input_dict))
                    del input_dict['x_query_pred']
                    del input_dict['x_query_raw']
                    del input_dict['x_sample_pred']
                    del input_dict['x_sample_raw']
                    serialized = self._serialized_example(**input_dict)
                    serial_list.append(serialized)
                self._output_tfrecords(serial_list, idx, output_path, "dev")
        else:
            raise NotImplementedError("this module not finished")

    def process_point_data(self, text_a_list, text_a_id_list, 
                           text_a_pred_list, len_a_list, 
                           text_b_list, text_b_id_list, 
                           text_b_pred_list, len_b_list, label_list,
                           encoder_fun, output_path, dev_size, mode, **kwargs):
        def _generate_input_dict(text_a, text_a_id, text_a_pred, len_a,
                                 text_b, text_b_id, text_b_pred, len_b,
                                 encoder_fun, label):
            input_dict = {'x_query': text_a_id, 
                          'x_query_pred': text_a_pred,
                          'x_query_raw': text_a,
                          'x_query_length': [len_a],
                          'x_sample': text_b_id, 
                          'x_sample_pred': text_b_pred,
                          'x_sample_raw': text_b,
                          'x_sample_length': [len_b],
                          'label': [label]}
            input_dict.update(encoder_fun(**input_dict))
            del input_dict['x_query_pred']
            del input_dict['x_query_raw']
            del input_dict['x_sample_pred']
            del input_dict['x_sample_raw']
            return input_dict
        logging.info('generating tfrecords ...')
        serial_mp = defaultdict(list)
        if dev_size >0 and dev_size <1: dev_size = int(dev_size * 100)
        for idx in tqdm(range(len(text_a_list)), ncols = 70):
            label = label_list[idx]
            input_dict = _generate_input_dict(text_a_list[idx],
                                        text_a_id_list[idx],
                                        text_a_pred_list[idx],
                                        len_a_list[idx],
                                        text_b_list[idx],
                                        text_b_id_list[idx],
                                        text_b_pred_list[idx],
                                        len_b_list[idx],
                                        encoder_fun, 
                                        label)
            serialized = self._serialized_example(**input_dict)
            serial_mp[label].append(serialized)
        for label in serial_mp:
            if mode == 'train':
                train_serial_list = serial_mp[label][:-dev_size]
                self._output_tfrecords(train_serial_list, label, output_path, "train")
                test_serial_list = serial_mp[label][-dev_size:]
                self._output_tfrecords(test_serial_list, label, output_path, "dev")
            else:
                serial_list = serial_mp[label]
                self._output_tfrecords(serial_list, label, output_path, "test")

    def process_ner_data(self, text_id_list, text_pred_list, 
                         len_list, text_list, label_list, label_path,
                         encoder_fun, output_path, dev_size, mode, **kwargs):
        _label_list = list(chain.from_iterable(label_list))
        mp_label = {item:idx for idx,item in enumerate(list(set(_label_list)))}
        pickle.dump(mp_label, open(label_path, 'wb'))

        dataset = []
        for idx,text_id in enumerate(tqdm(text_id_list, ncols = 70)):
            labels = label_list[idx]
            labels = [mp_label[item] for item in labels]
            O_label = mp_label['O']
            input_dict = {'x_query': text_id, 
                          'x_query_pred': text_pred_list[idx], 
                          'x_query_raw': text_list[idx], 
                          'x_query_length': [len_list[idx]], 
                          'label': labels[:self.maxlen]+[O_label]*(max(self.maxlen-len(labels),0))
                          }
            input_dict.update(encoder_fun(**input_dict))
            del input_dict['x_query_pred']
            del input_dict['x_query_raw']
            serialized = self._serialized_example(**input_dict)
            dataset.append(serialized)

        logging.info('generating [%s] tfrecords ...'%mode)
        if dev_size >0 and dev_size <1: dev_size = int(len(dataset)*dev_size)
        if mode == 'train':
            dataset_train = dataset[:-dev_size]
            dataset_test = dataset[-dev_size:]
            self._output_tfrecords(dataset_train, 0, output_path, "train")
            self._output_tfrecords(dataset_test, 0, output_path, "dev")
            logging.info('output [%s] train tfrecords ...'%len(dataset_train))
            logging.info('output [%s] dev tfrecords ...'%len(dataset_test))
        else:
            self._output_tfrecords(dataset, 0, output_path, "test")
            logging.info('output [%s] test tfrecords ...'%len(dataset))

    def process_translation_data(self, encode_id_list, encode_len_list, 
                            decode_id_list, decode_len_list, 
                            target_id_list, target_len_list, 
                            encoder_fun, output_path, dev_size, mode, **kwargs):
        dataset = []
        for idx,encode_id in enumerate(tqdm(encode_id_list, ncols = 70)):
            label = target_id_list[idx]
            input_dict = {'encode': encode_id_list[idx], 
                          'encode_length': [encode_len_list[idx]], 
                          'decode': decode_id_list[idx], 
                          'decode_length': [decode_len_list[idx]], 
                          'target': target_id_list[idx], 
                          'target_length': [target_len_list[idx]], 
                          }
            serialized = self._serialized_example(**input_dict)
            dataset.append(serialized)

        logging.info('generating [%s] tfrecords ...'%mode)
        if dev_size >0 and dev_size <1: dev_size = int(len(dataset)*dev_size)
        if mode == 'train':
            dataset_train = dataset[:-dev_size]
            dataset_test = dataset[-dev_size:]
            self._output_tfrecords(dataset_train, 0, output_path, "train")
            self._output_tfrecords(dataset_test, 0, output_path, "dev")
            logging.info('output [%s] train tfrecords ...'%len(dataset_train))
            logging.info('output [%s] dev tfrecords ...'%len(dataset_test))
        else:
            self._output_tfrecords(dataset, 0, output_path, "test")
            logging.info('output [%s] test tfrecords ...'%len(dataset))

    def process(self, text_list, label_list, sen2id_fun, encoder_fun, vocab_dict, path,
                label_path, dev_size = 1, data_type = 'column_2', mode = 'train'):
        """
        sen2id_fun: for sen2id in embedding
        encoder_fun: add features for encoder
        tfrecords_mode for task:
            classify task: class
            match task: class, pair, point
            ner: ner
        """
        params_dict = {'output_path': path,
                       'label_path': label_path,
                       'text_list': text_list,
                       'label_list': label_list,
                       'encoder_fun': encoder_fun,
                       'dev_size': dev_size,
                       'mode': mode}
        logging.info('sentence to id ...')
        if data_type == 'column_2':
            #2 columns: text label
            assert self.tfrecords_mode in ['class','pair','ner'], "error data format for tfrecords_mode"
            text_pred_list, text_id_list, len_list = sen2id_fun(text_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            params_dict.update({'text_pred_list': text_pred_list, 
                                'text_id_list': text_id_list, 
                                'len_list': len_list})

        elif data_type == 'column_3':
            #3 columns: text_a, text_b, label
            assert self.tfrecords_mode in ['point'], "error data format for tfrecords_mode"
            size = len(label_list)
            text_a_list = text_list[:size]
            text_b_list = text_list[-size:]
            text_a_pred_list, text_a_id_list, len_a_list = sen2id_fun(text_a_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            text_b_pred_list, text_b_id_list, len_b_list = sen2id_fun(text_b_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            params_dict.update({'text_a_list': text_a_list,
                                'text_a_pred_list': text_a_pred_list,
                                'text_a_id_list': text_a_id_list,
                                'len_a_list': len_a_list,
                                'text_b_list': text_b_list,
                                'text_b_pred_list': text_b_pred_list,
                                'text_b_id_list': text_b_id_list,
                                'len_b_list': len_b_list})
        elif data_type == 'translation':
            size = len(label_list)
            encode_list = text_list[:size]
            decode_list = text_list[-size:]
            encode_pred_list, encode_id_list, encode_len_list = sen2id_fun(encode_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            decode_pred_list, decode_id_list, decode_len_list = sen2id_fun(decode_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            target_pred_list, target_id_list, target_len_list = sen2id_fun(label_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            params_dict.update({'encode_id_list': encode_id_list,
                                'encode_len_list': encode_len_list, 
                                'decode_id_list': decode_id_list,
                                'decode_len_list': decode_len_list,
                                'target_id_list': target_id_list,
                                'target_len_list': target_len_list})
        else:
            raise ValueError('unknown data type error')

        if self.tfrecords_mode == 'class':
            self.process_class_data(**params_dict)
        elif self.tfrecords_mode == 'pair':
            self.process_pair_data(**params_dict)
        elif self.tfrecords_mode == 'point':
            self.process_point_data(**params_dict)
        elif self.tfrecords_mode == 'ner':
            self.process_ner_data(**params_dict)
        elif self.tfrecords_mode == 'translation':
            self.process_translation_data(**params_dict)
        else:
            raise ValueError('unknown tfrecords mode')

