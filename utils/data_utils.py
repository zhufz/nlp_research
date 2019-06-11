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
from collections import defaultdict
from gensim import corpora,models,similarities
import tensorflow as tf
import sys,os

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from common.similarity import Similarity
from utils.recall import OriginRecall, InvertRecall, Annoy

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

def load_ner_data(path):
    data = []
    data_label = []
    with open(path) as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append(' '.join(sent_))
            #data_label.append(' '.join(tag_))
            data_label.append(tag_)
            sent_, tag_ = [], []

    return data, data_label

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

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

class GenerateTfrecords():
    """utils for tfrecords
    """
    def __init__(self, mode, maxlen):
        #class mode: label: class id 
        #pair mode: label: 1/0, whether come from same class 
        self.tfrecords_mode = mode
        self.maxlen = maxlen

    def _get_pair_id(self, text_id_list, label_list, test_size = 1):
        #return:
        #train_list:{样本id:{1:[正样本列表],0:[负样本列表]},...}
        #test_list: (样本id, [(1/0, 正/负样本id),(1/0, 正/负样本id),...]
        mp_label = defaultdict(list)
        for idx in range(len(text_id_list)):
            mp_label[label_list[idx]].append(idx)

        label_set = set(mp_label) # all labels set
        #label d1 d2
        train_list = defaultdict(dict)
        test_list = []

        #1667+91=1758
        for label in mp_label:
            #choose positive sample
            pos_list = mp_label[label]
            for idx in range(len(pos_list)-test_size):
                #if len(pos_list)-1 == 1:pdb.set_trace()
                tmp_pos_list = self._get_pos(pos_list, idx, len(pos_list)-test_size)
                for item in tmp_pos_list:
                    #train_list.append((1, pos_list[idx], item))
                    if 1 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][1] = []
                    train_list[pos_list[idx]][1].append(item)
                tmp_neg_list = self._get_neg(mp_label, label, label_set)
                for item in tmp_neg_list:
                    #train_list.append((0, pos_list[idx], item))
                    if 0 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][0] = []
                    train_list[pos_list[idx]][0].append(item)
            #test: the last sample fot each label 
            for item in pos_list[-test_size:]:
                test_list.append((item, \
                                   self._get_pos_neg(mp_label, label,
                                                     label_set, test_size)))
        return train_list, test_list

    def _get_pos(self, pos_data, idx, length):
        #select an id not equals to the idx from range(0,length) 
        assert 1 != length, "can't select diff pos sample with max=1"
        res_idx = idx
        #pdb.set_trace()
        res_list = []
        for tmp_idx in range(length):
            if idx == tmp_idx:continue
            res_list.append(pos_data[tmp_idx])
        return res_list

    def _get_neg(self, data, label, label_set):
        #select an neg label sample from data
        res_list = []
        for tmp_label in list(label_set):
            if tmp_label == label: continue
            res_list.append(random.choice(data[tmp_label][:-1]))
        return res_list

    def _get_pos_neg(self, data, label, label_set, test_size):
        data_list = []
        for tmp_label in list(label_set):
            if label == tmp_label:
                for item in data[tmp_label][:-test_size]:
                    data_list.append((1, item))
                #data_list.append((1, random.choice(data[tmp_label][:-1])))
            else:
                for item in data[tmp_label][:-test_size]:
                    data_list.append((0, item))
                #data_list.append((0, random.choice(data[tmp_label][:-1])))
        return data_list

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
        #pdb.set_trace()
        if self.tfrecords_mode == 'class':
            keys_to_features = {
                "x_query": tf.FixedLenFeature([self.maxlen], tf.int64),
                #"x_query_pred": tf.VarLenFeature(tf.string),
                #"x_query_raw": tf.VarLenFeature(tf.string),
                "x_query_length": tf.FixedLenFeature([1], tf.int64),
                "label": tf.FixedLenFeature([1], tf.int64),
            }
            keys_to_features.update(encoder.keys_to_features())
            parsed = tf.parse_single_example(record, keys_to_features)
            # Perform additional preprocessing on the parsed data.
            label = tf.reshape(parsed['label'], [1])
            ret =  {'x_query': tf.reshape(parsed['x_query'], [self.maxlen]),
                    #'x_query_pred': parsed['x_query_pred'],
                    #'x_query_raw': parsed['x_query_raw'],
                    'x_query_length': tf.reshape(parsed['x_query_length'], [1])[0],
                    'label': label[0]}
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label[0]
        elif self.tfrecords_mode in ['pair','point']:
            keys_to_features = {
                "x_query": tf.FixedLenFeature([self.maxlen], tf.int64),
                "x_sample": tf.FixedLenFeature([self.maxlen], tf.int64),
                #"x_query_pred": tf.VarLenFeature(tf.string),
                #"x_sample_pred": tf.VarLenFeature(tf.string),
                #"x_query_raw": tf.VarLenFeature(tf.string),
                #"x_sample_raw": tf.VarLenFeature(tf.string),
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
                    #'x_query_pred': parsed['x_query_pred'],
                    #'x_sample_pred': parsed['x_sample_pred'],
                    #'x_query_raw': parsed['x_query_raw'],
                    #'x_sample_raw': parsed['x_sample_raw'],
                    'x_query_length': tf.reshape(parsed['x_query_length'], [1])[0],
                    'x_sample_length': tf.reshape(parsed['x_sample_length'], [1])[0],
                    'label': label[0]
                    } 
            ret.update(encoder.parsed_to_features(parsed = parsed))
            return ret, label[0]
        else:
            raise ValueError('unknown tfrecords mode')

    def process(self, text_list, label_list, sen2id_fun, encoder_fun, vocab_dict, path,
                label_path, test_size = 1, data_type = 'column_2'):
        """
        sen2id_fun: for sen2id in embedding
        encoder_fun: add features for encoder
        """
        dataset = []
        tmp_label = None
        output_path = path
        label_id = 0

        ################## save label info ######################
        mp_label = {item:idx for idx,item in enumerate(list(set(label_list)))}
        pickle.dump(mp_label, open(label_path, 'wb'))

        ##################### text 2 id##########################
        logging.info('sentence to id ...')
        if data_type == 'column_2':
            text_pred_list, text_id_list, len_list = sen2id_fun(text_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
            assert self.tfrecords_mode in ['class','pair'], "error data format for tfrecords_mode"
        elif data_type == 'column_3':
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
            assert self.tfrecords_mode in ['point'], "error data format for tfrecords_mode"
        else:
            raise ValueError('unknown data type error')
        #########################################################

        if self.tfrecords_mode == 'class':
            mp_dataset = defaultdict(list)
            for idx,text_id in enumerate(text_id_list):
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
            ##################################################
            logging.info('generating tfrecords ...')
            for label in mp_dataset:
                dataset_train = mp_dataset[label][:-test_size]
                dataset_test = [mp_dataset[label][-test_size]]
                self._output_tfrecords(dataset_train, mp_label[label], output_path,
                                       "train")
                self._output_tfrecords(dataset_test, mp_label[label], output_path, 
                                       "test")
            logging.info('training class num: {}'.format(len(mp_dataset)))
            logging.info('testing num in each class: {}'.format(test_size))

        elif self.tfrecords_mode == 'pair':
            #build pairwise data from data like 'class' type
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
            train_list, test_list = self._get_pair_id(text_id_list, 
                                                     label_list,
                                                     test_size)
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
            ##################################################
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
                self._output_tfrecords(serial_list, idx, output_path, "test")
        elif self.tfrecords_mode == 'point':
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
            if test_size >0 and test_size <1: test_size = int(test_size * 100)
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
                train_serial_list = serial_mp[label][:-test_size]
                self._output_tfrecords(train_serial_list, label, output_path, "train")
                test_serial_list = serial_mp[label][-test_size:]
                self._output_tfrecords(test_serial_list, label, output_path, "test")
        else:
            raise ValueError('unknown tfrecords mode')
