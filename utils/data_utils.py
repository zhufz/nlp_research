import numpy as np
import pandas as pd
import numpy as np
import os
import pickle
import random
import pdb
from collections import defaultdict
from gensim import corpora,models,similarities
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
                  mode = 'random', **kwargs):
        if mode == 'random':
            return self.get_batch_random(data, batch_size, num_epochs, maxlen1,
                                  maxlen2)
        elif mode == 'supervised':
            return self.get_batch_supervised(data, batch_size, num_epochs, maxlen1,
                                  maxlen2, task)
        else:
            raise ValueError('unknown mode to get batch data!')

    def get_batch_random(self, data, batch_size, num_epochs, maxlen1, maxlen2):
        #定义召回类对象并初始化
        #recall = InvertRecall(data)
        rel_set = self.get_rel_set(self.rel)
        result_list = []
        cnt_batch_size = 0
        epoch = 0
        while epoch < num_epochs:
            #random select a query
            d1 = random.choice(list(rel_set.keys()))
            if cnt_batch_size == 0:
                X1,X2,X1_len,X2_len = [],[],[],[]
            pos_list = rel_set[d1][1]
            neg_list = rel_set[d1][0]
            #random select pos and neg sample
            #pos_list = recall(data, pos_list, d1, num = 64, reverse = False)
            d2p = random.choice(pos_list)
            #neg_list = recall(data, neg_list, d1, num = 64, reverse = True)
            d2n = random.choice(neg_list)
            X1.append(data[d1])
            X1.append(data[d1])
            X2.append(data[d2p])
            X2.append(data[d2n])
            cnt_batch_size += 2
            if cnt_batch_size == batch_size:
                epoch += 1
                cnt_batch_size = 0
                yield X1,X2

    def get_batch_supervised(self, data, batch_size, num_epochs, maxlen1,
                             maxlen2, task):
        #定义召回类对象并初始化
        #recall = InvertRecall(data)
        #先动态挑选pair_list,再生成batch数据
        rel_set = self.get_rel_set(self.rel)
        result_list = []
        cnt_batch_size = 0
        epoch = -1
        while epoch < num_epochs:
            epoch += 1
            for d1 in rel_set:
                if cnt_batch_size == 0:
                    X1,X2 = [],[]
                #find best pos sample
                label = 1
                pos_list = rel_set[d1][label]
                tmp_list = []
                #pos_list = recall(data, pos_list, d1, num = 64, reverse = False)
                for d2 in pos_list:
                    tmp_list.append((data[d1], data[d2]))
                pos_pred = task.predict_prob(tmp_list)
                min_idx = np.argmin(pos_pred)
                min_d2 = pos_list[min_idx]
                X1.append(data[d1])
                X2.append(data[min_d2])

                #find best neg sample
                label = 0
                neg_list = rel_set[d1][label]
                tmp_list = []
                #neg_list = recall(data, neg_list, d1, num = 64, reverse = True)
                for item in neg_list:
                    tmp_list.append((data[d1], data[d2]))
                neg_pred = task.predict_prob(tmp_list)
                max_idx = np.argmax(neg_pred)
                max_d2 = neg_list[max_idx]
                X1.append(data[d1])
                X2.append(data[max_d2])
                cnt_batch_size += 2
                if cnt_batch_size == batch_size:
                    cnt_batch_size = 0
                    yield X1,X2

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

