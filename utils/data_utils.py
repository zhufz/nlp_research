import numpy as np
import pandas as pd
import numpy as np
import os
import pickle
import random
import logging
import pdb
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



class PairGenerator():
    def __init__(self, rel_file, index_file, test_file):
        self.rel = self.read_relation(filename=rel_file)
        self.index_data, self.label_data = self.read_index(filename = index_file)
        self.test_data = self.read_test(filename = test_file)
        self.pair_list = self.make_pair(self.rel)

    def read_relation(self, filename):
        data = []
        for line in open(filename):
            line = line.strip().split()
            data.append( (int(line[0]), int(line[1]), int(line[2])) )
        print('[%s]\n\trelation size: %s' % (filename, len(data)))
        return data

    def read_index(self, filename):
        data_list = [line.strip() for line in open(filename).readlines()]
        data = [0 for _ in range(len(data_list))]
        label = [0 for _ in range(len(data_list))]
        for line in data_list:
            line = line.split('\t')
            data[int(line[0])] =  line[1]
            label[int(line[0])] =  line[2]
        print('[%s]\n\tindex size: %s' % (filename, len(data)))
        #data 为句子列表
        #label 为对应的标签，如（播放、关闭等）
        return data, label

    def read_test(self, filename):
        data_list = [line.strip() for line in open(filename).readlines()]
        data = []
        for line in data_list:
            arr = line.split()
            if len(arr) == 3:
                data.append((arr[0], arr[1], arr[2]))
        print('[%s]\n\ttest size: %s' % (filename, len(data)))
        return data

    def get_rel_set(self, rel):
        rel_set = {}
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        return rel_set

    def make_pair(self, rel):
        rel_set = self.get_rel_set(rel)
        pair_list = []
        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse = True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx+1:]:
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append( (d1, high_d2, low_d2) )
        print('Pair Instance Count:', len(pair_list))
        return pair_list

    def get_batch(self,  data, batch_size, num_epochs, maxlen1, maxlen2, task,
                  mode = 'random', random_select_query = False, shuffle = True,
                  margin = None, semi_hard = False):
        #定义召回类对象并初始化
        #recall = InvertRecall(data)
        rel_set = self.get_rel_set(self.rel)
        result_list = []
        cnt_batch_size = 0
        epoch = 0
        rel_keys_list = list(rel_set.keys())
        rel_keys_id = 0
        rel_keys_len = len(rel_keys_list)
        while epoch < num_epochs:
            if random_select_query:
                #random select a query
                d1 = random.choice(rel_keys_list)
            else:
                if rel_keys_id == rel_keys_len - 1:
                    rel_keys_id = 0
                    epoch += 1
                    if shuffle:
                        random.shuffle(rel_keys_list)
                d1 = rel_keys_list[rel_keys_id]
                rel_keys_id += 1
            if cnt_batch_size == 0:
                X1,X2,X1_len,X2_len = [],[],[],[]
            pos_list = rel_set[d1][1]
            neg_list = rel_set[d1][0]

            if mode == 'supervised':
                min_idx, max_idx = self._get_hard_d2(task, 
                                                     data, 
                                                     d1, 
                                                     pos_list, 
                                                     neg_list,
                                                     margin,
                                                     semi_hard)
                d2p = pos_list[min_idx]
                d2n = neg_list[max_idx]
            else:
                d2p = random.choice(pos_list)
                d2n = random.choice(neg_list)
            X1.append(data[d1])
            X1.append(data[d1])
            X2.append(data[d2p])
            X2.append(data[d2n])
            cnt_batch_size += 2
            if cnt_batch_size == batch_size:
                cnt_batch_size = 0
                yield X1,X2

    def _get_hard_d2(self, task, data, d1, pos_list, neg_list, margin, semi_hard):
        #get hard positvie and hard negative sample
        tmp_list = []
        for d2 in pos_list:
            tmp_list.append((data[d1], data[d2]))
        pos_pred = task.predict_prob(tmp_list)
        min_idx = np.argmin(pos_pred)
        min_score = np.min(pos_pred)

        tmp_list = []
        neg_list = random.sample(neg_list, min(128,len(neg_list)))
        #logging.info('{} neg sample selected!'.format(len(neg_list)))
        for d2 in neg_list:
            tmp_list.append((data[d1], data[d2]))
        neg_pred = task.predict_prob(tmp_list)
        #pdb.set_trace()
        if semi_hard:
            neg_pred_tmp = [item if item > min_score and \
                            item < min_score+margin else None \
                            for item in neg_pred]
            neg_pred_tmp = list(filter(lambda x : x != None, neg_pred_tmp))
            if len(neg_pred_tmp) != 0:
                neg_pred = neg_pred_tmp
                #logging.warn('{} simi-hard sample selected!'.format(len(neg_pred)))
            else:
                pass
                #logging.warn('no simi-hard sample selected!')
        max_idx = np.argmax(neg_pred)
        return min_idx, max_idx

    def get_test_batch(self, data, maxlen1, maxlen2, query = None):
        if query == None:
            mp = defaultdict(list)
            for d1, label, d2 in self.test_data:
                mp[int(d1)].append((int(label), int(d2)))

            for d1 in mp:
                X1,X2,labels = [],[],[]
                for idx in range(len(mp[d1])):
                    labels.append(mp[d1][idx][0])
                    d2 = mp[d1][idx][1]
                    X1.append(data[d1])
                    X2.append(data[d2])
                yield X1,X2,labels
        else:
            X1,X2,labels = [],[],[]
            for item in data:
                X1.append(query)
                X2.append(item)
            yield X1,X2,labels


def _create_serialized_example(current, vocab):
    """Helper for creating a serialized Example proto."""
    #example = tf.train.Example(features=tf.train.Features(feature={
    #    "decode_pre": _int64_feature(_sentence_to_ids(predecessor, vocab)),
    #    "encode": _int64_feature(_sentence_to_ids(current, vocab)),
    #    "decode_post": _int64_feature(_sentence_to_ids(successor, vocab)),
    #}))
    example = tf.train.Example(features=tf.train.Features(feature={
        "features": _int64_feature(_sentence_to_ids(current, vocab)),
    }))
    #example = tf.train.Example(features=tf.train.Features(feature=
    #    _int64_feature(_sentence_to_ids(current, vocab)),
    #))
    return example.SerializeToString()


def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
      for j in indices:
        writer.write(dataset[j])

def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.
    Args:
      name: Name of the dataset (e.g. "train").
      dataset: List of serialized Example protos.
      indices: List of indices of 'dataset' to be written.
      num_shards: The number of output shards.
    """
    tf.logging.info("Writing dataset %s", name)
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                       num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    tf.logging.info("Finished writing %d sentences in dataset %s.", len(indices), name)

def process_input_file(file_name):
    dataset = []
    for sentence_str in tf.gfile.FastGFile(filename):
        sentence_tokens = sentence_str.split()
        sentence_tokens = sentence_tokens[:FLAGS.max_sentence_length]
        serialized = _create_serialized_example(sentence_tokens, vocab)
        dataset.append(serialized)
        stats.update(["sentence_count"])
    indices = range(len(dataset))
    val_indices = indices[:FLAGS.num_validation_sentences]
    train_indices = indices[FLAGS.num_validation_sentences:]

    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices, FLAGS.validation_output_shards)
