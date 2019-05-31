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



#class PairGenerator():
#    def __init__(self, rel_file, index_file, test_file):
#        self.rel = self.read_relation(filename=rel_file)
#        self.index_data, self.label_data = self.read_index(filename = index_file)
#        self.test_data = self.read_test(filename = test_file)
#        self.pair_list = self.make_pair(self.rel)
#
#    def read_relation(self, filename):
#        data = []
#        for line in open(filename):
#            line = line.strip().split()
#            data.append( (int(line[0]), int(line[1]), int(line[2])) )
#        print('[%s]\n\trelation size: %s' % (filename, len(data)))
#        return data
#
#    def read_index(self, filename):
#        data_list = [line.strip() for line in open(filename).readlines()]
#        data = [0 for _ in range(len(data_list))]
#        label = [0 for _ in range(len(data_list))]
#        for line in data_list:
#            line = line.split('\t')
#            data[int(line[0])] =  line[1]
#            label[int(line[0])] =  line[2]
#        print('[%s]\n\tindex size: %s' % (filename, len(data)))
#        #data 为句子列表
#        #label 为对应的标签，如（播放、关闭等）
#        return data, label
#
#    def read_test(self, filename):
#        data_list = [line.strip() for line in open(filename).readlines()]
#        data = []
#        for line in data_list:
#            arr = line.split()
#            if len(arr) == 3:
#                data.append((arr[0], arr[1], arr[2]))
#        print('[%s]\n\ttest size: %s' % (filename, len(data)))
#        return data
#
#    def get_rel_set(self, rel):
#        rel_set = {}
#        for label, d1, d2 in rel:
#            if d1 not in rel_set:
#                rel_set[d1] = {}
#            if label not in rel_set[d1]:
#                rel_set[d1][label] = []
#            rel_set[d1][label].append(d2)
#        return rel_set
#
#    def make_pair(self, rel):
#        rel_set = self.get_rel_set(rel)
#        pair_list = []
#        for d1 in rel_set:
#            label_list = sorted(rel_set[d1].keys(), reverse = True)
#            for hidx, high_label in enumerate(label_list[:-1]):
#                for low_label in label_list[hidx+1:]:
#                    for high_d2 in rel_set[d1][high_label]:
#                        for low_d2 in rel_set[d1][low_label]:
#                            pair_list.append( (d1, high_d2, low_d2) )
#        print('Pair Instance Count:', len(pair_list))
#        return pair_list
#
#    def get_batch(self,  data, batch_size, num_epochs, maxlen1, maxlen2, task,
#                  mode = 'random', random_select_query = False, shuffle = True,
#                  margin = None, semi_hard = False):
#        #定义召回类对象并初始化
#        #recall = InvertRecall(data)
#        rel_set = self.get_rel_set(self.rel)
#        result_list = []
#        cnt_batch_size = 0
#        epoch = 0
#        rel_keys_list = list(rel_set.keys())
#        rel_keys_id = 0
#        rel_keys_len = len(rel_keys_list)
#        while epoch < num_epochs:
#            if random_select_query:
#                #random select a query
#                d1 = random.choice(rel_keys_list)
#            else:
#                if rel_keys_id == rel_keys_len - 1:
#                    rel_keys_id = 0
#                    epoch += 1
#                    if shuffle:
#                        random.shuffle(rel_keys_list)
#                d1 = rel_keys_list[rel_keys_id]
#                rel_keys_id += 1
#            if cnt_batch_size == 0:
#                X1,X2,X1_len,X2_len = [],[],[],[]
#            pos_list = rel_set[d1][1]
#            neg_list = rel_set[d1][0]
#
#            if mode == 'supervised':
#                min_idx, max_idx = self._get_hard_d2(task, 
#                                                     data, 
#                                                     d1, 
#                                                     pos_list, 
#                                                     neg_list,
#                                                     margin,
#                                                     semi_hard)
#                d2p = pos_list[min_idx]
#                d2n = neg_list[max_idx]
#            else:
#                d2p = random.choice(pos_list)
#                d2n = random.choice(neg_list)
#            X1.append(data[d1])
#            X1.append(data[d1])
#            X2.append(data[d2p])
#            X2.append(data[d2n])
#            cnt_batch_size += 2
#            if cnt_batch_size == batch_size:
#                cnt_batch_size = 0
#                yield X1,X2
#
#    def _get_hard_d2(self, task, data, d1, pos_list, neg_list, margin, semi_hard):
#        #get hard positvie and hard negative sample
#        tmp_list = []
#        for d2 in pos_list:
#            tmp_list.append((data[d1], data[d2]))
#        pos_pred = task.predict_prob(tmp_list)
#        min_idx = np.argmin(pos_pred)
#        min_score = np.min(pos_pred)
#
#        tmp_list = []
#        neg_list = random.sample(neg_list, min(128,len(neg_list)))
#        #logging.info('{} neg sample selected!'.format(len(neg_list)))
#        for d2 in neg_list:
#            tmp_list.append((data[d1], data[d2]))
#        neg_pred = task.predict_prob(tmp_list)
#        #pdb.set_trace()
#        if semi_hard:
#            neg_pred_tmp = [item if item > min_score and \
#                            item < min_score+margin else None \
#                            for item in neg_pred]
#            neg_pred_tmp = list(filter(lambda x : x != None, neg_pred_tmp))
#            if len(neg_pred_tmp) != 0:
#                neg_pred = neg_pred_tmp
#                #logging.warn('{} simi-hard sample selected!'.format(len(neg_pred)))
#            else:
#                pass
#                #logging.warn('no simi-hard sample selected!')
#        max_idx = np.argmax(neg_pred)
#        return min_idx, max_idx
#
#    def get_test_batch(self, data, maxlen1, maxlen2, query = None):
#        if query == None:
#            mp = defaultdict(list)
#            for d1, label, d2 in self.test_data:
#                mp[int(d1)].append((int(label), int(d2)))
#
#            for d1 in mp:
#                X1,X2,labels = [],[],[]
#                for idx in range(len(mp[d1])):
#                    labels.append(mp[d1][idx][0])
#                    d2 = mp[d1][idx][1]
#                    X1.append(data[d1])
#                    X2.append(data[d2])
#                yield X1,X2,labels
#        else:
#            X1,X2,labels = [],[],[]
#            for item in data:
#                X1.append(query)
#                X2.append(item)
#            yield X1,X2,labels

class GenerateTfrecords():
    """utils for tfrecords
    """
    def __init__(self, mode, maxlen):
        #class mode: label: class id 
        #pair mode: label: 1/0, whether come from same class 
        self.tfrecords_mode = mode
        self.maxlen = maxlen
        assert self.tfrecords_mode in ['pair','class']

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
        elif self.tfrecords_mode == 'pair':
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

    def _output_tfrecords(self, dataset, idx, path, mode):
        file_name = os.path.join(path, "{}_class_{:04d}".format(mode, idx))
        with tf.python_io.TFRecordWriter(file_name) as writer:
          for item in dataset:
            writer.write(item)

    def process(self, text_list, label_list, sen2id_fun, encoder_fun, vocab_dict, path,
                label_path, test_size = 1):
        """
        sen2id_fun: for sen2id in embedding
        encoder_fun: add features for encoder
        """
        dataset = []
        tmp_label = None
        output_path = path
        label_id = 0
        mp_label = {item:idx for idx,item in enumerate(list(set(label_list)))}
        pickle.dump(mp_label, open(label_path, 'wb'))


        text_pred_list, text_id_list, len_id_list = sen2id_fun(text_list, 
                                                               vocab_dict,
                                                               self.maxlen, 
                                                               need_preprocess=False)
        if self.tfrecords_mode == 'class':
            mp_dataset = defaultdict(list)
            for idx,text_id in enumerate(text_id_list):
                label = label_list[idx]
                input_dict = {'x_query': text_id, 
                              'x_query_pred': text_pred_list[idx], 
                              'x_query_raw': text_list[idx], 
                              'x_query_length': [len_id_list[idx]], 
                              'label': [mp_label[label]]
                              }
                input_dict.update(encoder_fun(**input_dict))
                del input_dict['x_query_pred']
                del input_dict['x_query_raw']
                serialized = self._serialized_example(**input_dict)
                mp_dataset[label].append(serialized)
            ##################################################
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
            def _generate_input_dict(text_id_list, text_pred_list, len_id_list,
                                    encoder_fun, query_id, sample_id, label):
                input_dict = {'x_query': text_id_list[query_id], 
                              'x_sample': text_id_list[sample_id], 
                              'x_query_pred': text_pred_list[query_id],
                              'x_sample_pred': text_pred_list[sample_id],
                              'x_query_raw': text_list[query_id],
                              'x_sample_raw': text_list[sample_id],
                              'x_query_length': [len_id_list[query_id]],
                              'x_sample_length': [len_id_list[sample_id]],
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
                                     len_id_list, 
                                     encoder_fun)
            train_list, test_list = self.get_pair_id(text_id_list, 
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
                                  'x_query_length': [len_id_list[query_id]],
                                  'x_sample_length': [len_id_list[sample_id]],
                                  'label': [label]}
                    input_dict.update(encoder_fun(**input_dict))
                    del input_dict['x_query_pred']
                    del input_dict['x_query_raw']
                    del input_dict['x_sample_pred']
                    del input_dict['x_sample_raw']
                    serialized = self._serialized_example(**input_dict)
                    serial_list.append(serialized)
                self._output_tfrecords(serial_list, idx, output_path, "test")
        else:
            raise ValueError('unknown tfrecords mode')

    def get_pair_id(self, text_id_list, label_list, test_size = 1):
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
                tmp_pos_list = self.get_pos(pos_list, idx, len(pos_list)-test_size)
                for item in tmp_pos_list:
                    #train_list.append((1, pos_list[idx], item))
                    if 1 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][1] = []
                    train_list[pos_list[idx]][1].append(item)
                tmp_neg_list = self.get_neg(mp_label, label, label_set)
                for item in tmp_neg_list:
                    #train_list.append((0, pos_list[idx], item))
                    if 0 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][0] = []
                    train_list[pos_list[idx]][0].append(item)
            #test: the last sample fot each label 
            for item in pos_list[-test_size:]:
                test_list.append((item, \
                                   self.get_pos_neg(mp_label, label,
                                                     label_set, test_size)))
        return train_list, test_list

    def get_pos(self, pos_data, idx, length):
        #select an id not equals to the idx from range(0,length) 
        assert 1 != length, "can't select diff pos sample with max=1"
        res_idx = idx
        #pdb.set_trace()
        res_list = []
        for tmp_idx in range(length):
            if idx == tmp_idx:continue
            res_list.append(pos_data[tmp_idx])
        return res_list

    def get_neg(self, data, label, label_set):
        #select an neg label sample from data
        res_list = []
        for tmp_label in list(label_set):
            if tmp_label == label: continue
            res_list.append(random.choice(data[tmp_label][:-1]))
        return res_list

    def get_pos_neg(self, data, label, label_set, test_size):
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
