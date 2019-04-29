import pandas as pd
import pdb
import sys
import random
from collections import defaultdict

class GenerateData():
    def __init__(self, path):
        self.path = path
        #csv = pd.read_csv(self.path, sep="\t", header = 0, error_bad_lines=False)
        csv = pd.read_csv(self.path, header = 0, error_bad_lines=False)
        self.text = csv['text']
        self.label = csv['target']
        self.data = defaultdict(list)


    def process(self, train_rate = 0.9):
        """train:test = 8:2"""
        #train_path = '.'.join(self.path.split('.')[:-1]) + '_train.csv'
        #test_path = '.'.join(self.path.split('.')[:-1])+ '_test.csv'

        train_path = 'data/classify_train.csv'
        test_path = 'data/classify_test.csv'

        for idx in range(len(self.text)):
            self.data[self.label[idx]].append(self.text[idx])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for key in self.data:
            all_len = len(self.data[key])
            train_len = int(all_len*train_rate)
            test_len = all_len - train_len
            for idx,item in enumerate(self.data[key]):
                #if idx<train_len:
                if idx<all_len-2:
                    train_x.append(self.data[key][idx])
                    train_y.append(key)
                else:
                    test_x.append(self.data[key][idx])
                    test_y.append(key)

        dt_train = pd.DataFrame({'text':train_x,'intent':train_y})
        dt_test = pd.DataFrame({'text':test_x,'intent':test_y})
        dt_train.to_csv(train_path,index=False,sep=',')
        dt_test.to_csv(test_path,index=False,sep=',')
        print("split finished!")

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

    def get_pos_neg(self, data, label, label_set):
        data_list = []
        for tmp_label in list(label_set):
            if label == tmp_label:
                data_list.append((1, random.choice(data[tmp_label][:-1])))
            else:
                data_list.append((0, random.choice(data[tmp_label][:-1])))
        return data_list

    def get_pos_neg1(self, data, label, label_set, test_size):
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


    def process_match(self):
        index_path = 'data/match_index.csv'
        relation_path = 'data/match_relation.csv'
        test_path = 'data/match_test.csv'
        #label_path = 'data/match_label.csv'
        index_datas = []
        for idx in range(len(self.text)):
            index_datas.append((self.text[idx],self.label[idx]))
            self.data[self.label[idx]].append(idx)
        label_set = set(self.label) # all labels set
        #label d1 d2
        result_list = []
        test_list = []
        test_size = 1
        for label in self.data:
            #choose positive sample
            pos_list = self.data[label]
            for idx in range(len(pos_list)-test_size):
                #if len(pos_list)-1 == 1:pdb.set_trace()
                tmp_pos_list = self.get_pos(pos_list, idx,
                                            len(pos_list)-test_size)
                for item in tmp_pos_list:
                    result_list.append((1, pos_list[idx],item))
                tmp_neg_list = self.get_neg(self.data, label, label_set)
                for item in tmp_neg_list:
                    result_list.append((0, pos_list[idx], item))
            #test: the last sample fot each label 
            test_list.append((pos_list[-1], \
                                   self.get_pos_neg1(self.data, label,
                                                     label_set, test_size)))
        with open(index_path,'w') as f_index, \
                open(relation_path,'w') as f_rel, \
                open(test_path,'w') as f_test:
            for idx,item in enumerate(index_datas):
                f_index.write("{}\t{}\t{}\n".format(idx,item[0],item[1]))
            for item in result_list:
                f_rel.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
            for item in test_list:
                #pdb.set_trace()
                for data in item[1]:
                    #d1, label, d2
                    f_test.write("{}\t{}\t{}\n".format(item[0], data[0], data[1]))

if __name__ == '__main__':
    split = GenerateData('./data/intent.csv')
    if len(sys.argv) > 1:
        if sys.argv[1] == 'match':
            split.process_match()
        else:
            split.process()
    else:
        print('match or classfiy?')






