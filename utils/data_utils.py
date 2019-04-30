import numpy as np
import pandas as pd
import numpy as np
import os
import random
import pdb
from collections import defaultdict
from gensim import corpora,models,similarities
import sys,os

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from common.similarity import Similarity

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
    default = tags["0"]
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


class Recall():
    def __init__(self, data_list):
        data_list = self._check(data_list)
        self.dictionary = corpora.Dictionary(data_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in data_list]
        self.tfidf = models.TfidfModel(corpus) #文档建tfidf模型

    def _check(self, data_list):
        assert type(data_list) == list and len(data_list)>0,'type error or empty for data_list'
        if type(data_list[0]) != list:
            return [data.split() for data in data_list]
        return data_list

    def _init_query(self, query_list):
        query_list = self._check(query_list)
        #相似度模型
        corpus = [self.dictionary.doc2bow(doc) for doc in query_list]
        self.index = similarities.SparseMatrixSimilarity(self.tfidf[corpus],
                                                         num_features=len(self.dictionary.keys()))

    def _cal_similarity(self, text):
        if type(text) == str:
            text = text.split()
        doc_test_vec = self.dictionary.doc2bow(text)
        text_tfidf = self.tfidf[doc_test_vec]
        sim = self.index[text_tfidf]
        return sim

    def __call__(self, data, query_id_list, text_id, num, reverse = False):
        query_list = [data[idx] for idx in query_id_list]
        text = data[text_id]
        self._init_query(query_list)
        sim = self._cal_similarity(text)
        if reverse == True:
            sorted_idx = np.argsort(-np.array(sim))
        else:
            sorted_idx = np.argsort(np.array(sim))
        query_id_list = np.array(query_id_list)[sorted_idx][:num]
        return query_id_list

class Recall2():
    #基于倒排索引
    def __init__(self, data_list):
        data_list = self._check(data_list)
        self.dictionary = corpora.Dictionary(data_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in data_list]
        self.tfidf = models.TfidfModel(corpus) #文档建tfidf模型

    def _check(self, data_list):
        assert type(data_list) == list and len(data_list)>0,'type error or empty for data_list'
        if type(data_list[0]) != list:
            return [data.split() for data in data_list]
        return data_list

    def create_inverted_index(self, data_list):
        inverted_index = defaultdict(set)
        for idx, word_list in enumerate(data_list):
            for word in word_list:
                inverted_index[word].add(idx)
        return inverted_index

    def __call__(self, data, query_id_list, text_id, num, **kwargs):
        query_list = [data[idx].split() for idx in query_id_list]
        text = data[text_id].split()
        inverted_index = self.create_inverted_index(query_list)
        doc_test_vec = self.dictionary.doc2bow(text)
        text_tfidf = [item[1] for item in self.tfidf[doc_test_vec]]
        sorted_id = np.argsort(-np.array(text_tfidf))

        ret = set()
        for idx in sorted_id:
            word = text[idx]
            if word in inverted_index:
                ret.update([query_id_list[idx] for idx in inverted_index[word]])
            if len(ret) > num: break
        ret = (list(ret))[:num]
        if len(ret) < num:
            add_num = num - len(ret)
            if add_num <= len(query_id_list):
                ret += random.sample(query_id_list, add_num)
            else:
                ret += query_id_list
        return ret

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
        print('[%s]\n\tInstance size: %s' % (filename, len(data)))
        return data

    def read_index(self, filename):
        data_list = [line.strip() for line in open(filename).readlines()]
        data = [0 for _ in range(len(data_list))]
        label = [0 for _ in range(len(data_list))]
        for line in data_list:
            line = line.split('\t')
            data[int(line[0])] =  line[1]
            label[int(line[0])] =  line[2]
        print('[%s]\n\tInstance size: %s' % (filename, len(data)))
        return data, label

    def read_test(self, filename):
        data_list = [line.strip() for line in open(filename).readlines()]
        data = []
        for line in data_list:
            arr = line.split()
            if len(arr) == 3:
                data.append((arr[0], arr[1], arr[2]))
        print('[%s]\n\tInstance size: %s' % (filename, len(data)))
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
        recall = Recall2(data)
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
            d1_len = min(maxlen1, len(data[d1]))
            d2p_len = min(maxlen2, len(data[d2p]))
            d2n_len = min(maxlen2, len(data[d2n]))
            X1.append(data[d1])
            X1.append(data[d1])
            X2.append(data[d2p])
            X2.append(data[d2n])
            X1_len.append(d1_len)
            X1_len.append(d1_len)
            X2_len.append(d2p_len)
            X2_len.append(d2n_len)
            cnt_batch_size += 2
            if cnt_batch_size == batch_size:
                epoch += 1
                cnt_batch_size = 0
                #yield X1,X2,X1_len,X2_len
                yield X1,X2

    def get_batch_supervised(self, data, batch_size, num_epochs, maxlen1,
                             maxlen2, task):
        #定义召回类对象并初始化
        recall = Recall2(data)
        #先动态挑选pair_list,再生成batch数据
        rel_set = self.get_rel_set(self.rel)
        result_list = []
        cnt_batch_size = 0
        epoch = -1
        while epoch < num_epochs:
            epoch += 1
            for d1 in rel_set:
                if cnt_batch_size == 0:
                    X1,X2,X1_len,X2_len = [],[],[],[]
                #find best pos sample
                label = 1
                pos_list = rel_set[d1][label]
                tmp_list = []
                #pos_list = recall(data, pos_list, d1, num = 64, reverse = False)
                for d2 in pos_list:
                    d1_len = min(maxlen1, len(data[d1]))
                    d2_len = min(maxlen2, len(data[d2]))
                    tmp_list.append((data[d1], data[d2], d1_len, d2_len))
                pos_pred = task.predict_prob(tmp_list)
                min_idx = np.argmin(pos_pred)
                min_d2 = pos_list[min_idx]
                d1_len = min(maxlen1, len(data[d1]))
                min_d2_len = min(maxlen2, len(data[min_d2]))
                X1.append(data[d1])
                X2.append(data[min_d2])
                X1_len.append(d1_len)
                X2_len.append(min_d2_len)

                #find best neg sample
                label = 0
                neg_list = rel_set[d1][label]
                tmp_list = []
                neg_list = recall(data, neg_list, d1, num = 64, reverse = True)
                for item in neg_list:
                    d1_len = min(maxlen1, len(data[d1]))
                    d2_len = min(maxlen2, len(data[d2]))
                    tmp_list.append((data[d1], data[d2], d1_len, d2_len))
                neg_pred = task.predict_prob(tmp_list)
                max_idx = np.argmax(neg_pred)
                max_d2 = neg_list[max_idx]
                d1_len = min(maxlen1, len(data[d1]))
                max_d2_len = min(maxlen2, len(data[max_d2]))
                X1.append(data[d1])
                X2.append(data[max_d2])
                X1_len.append(d1_len)
                X2_len.append(max_d2_len)
                cnt_batch_size += 2
                if cnt_batch_size == batch_size:
                    cnt_batch_size = 0
                    #yield X1,X2,X1_len,X2_len
                    yield X1,X2

    def get_test_batch(self, data, maxlen1, maxlen2, query = None):

        if query == None:
            #data = self.test_data
            mp = defaultdict(list)
            for d1, label, d2 in self.test_data:
                mp[int(d1)].append((int(label), int(d2)))

            for d1 in mp:
                X1,X2,labels,X1_len,X2_len = [],[],[],[],[]
                for idx in range(len(mp[d1])):
                    labels.append(mp[d1][idx][0])
                    d2 = mp[d1][idx][1]
                    X1.append(data[d1])
                    X2.append(data[d2])
                    X1_len.append(min(maxlen1, len(data[d1])))
                    X2_len.append(min(maxlen2, len(data[d2])))
                #yield X1,X2,X1_len,X2_len,labels
                yield X1,X2,labels
        else:
            X1,X2,labels,X1_len,X2_len = [],[],[],[],[]
            for item in data:
                X1.append(query)
                X2.append(item)
                X1_len.append(min(maxlen1, len(query)))
                X2_len.append(min(maxlen2, len(item)))
            #yield X1,X2,X1_len,X2_len,labels
            yield X1,X2,labels

