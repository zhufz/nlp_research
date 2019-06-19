#-*- coding:utf-8 -*-
from collections import defaultdict
import random
import pdb
import sys,os

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.ac import AC

class NERUtil():
    def __init__(self):
        pass

    def load_ner_data(self, path):
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

    def replace_by_position(self, string, pos, subs):
        """ 原始string
            pos: list类型，位置列表[(l,r),...]
            subs:与list对应的代替换后的字符串，[str0,str1,...]
        """
        idx = 0
        mp = {}
        for idx, (pos_l, pos_r) in enumerate(pos):
            mp[pos_l] = (pos_r, subs[idx])

        new_list = []
        idx = 0
        while idx < len(string) :
            if idx in mp:
                new_list.append(mp[idx][1])
                idx = mp[idx][0]
            else:
                new_list.append(string[idx])
                idx += 1
        return ''.join(new_list)

    def process_class_data(self, mp_type2word, cls_data_path, tmp_path):
        """ 由于class 数据内，可能包含泛化词(<>括号里内容)，
            因此需要实例化
        """
        ac = AC()
        for key in mp_type2word.keys():
            ac.add(key)
        f_out = open(tmp_path, 'w')
        for idx,line in enumerate(open(cls_data_path, 'r')):
            if idx == 0: continue #filter the first head text
            line = line.split('\t')[0].strip()
            res = ac.search(line)
            if res == {}:
               f_out.write(line+'\n')
            else:
               pos_list = []
               str_list = []
               for key in res:
                   for pos in res[key]:
                       pos_list.append(pos)
                       str_list.append(random.choice(mp_type2word[key]))
               new_line = self.replace_by_position(line, pos_list, str_list)
               f_out.write(new_line+'\n')
        f_out.close()

    def generate_from_gen(self, line, type_ac, mp_type2word, num):
        """将泛化句子模板进行随机实例化
            num 为实例化的数量
        """
        out_list = []
        res = type_ac.search(line)
        for idx in range(num):
            if res == {}:
                out_list.append(line)
                break
            pos_list = []
            str_list = []
            for key in res:
                for pos in res[key]:
                    pos_list.append(pos)
                    str_list.append(random.choice(mp_type2word[key]))
            new_line = self.replace_by_position(line, pos_list, str_list)
            out_list.append(new_line)
        return out_list

    def generate_ner_data(self, mp_word2type, mp_type2word, tmp_path, out_path):
        """根据词典标注实体
        """
        #word ac
        ac = AC()
        for key in mp_word2type.keys():
            ac.add(key)
        #type ac
        type_ac = AC()
        for key in mp_type2word.keys():
            type_ac.add(key)

        result = []
        with open(tmp_path) as f_in:
            for line in f_in:
                line = line.strip()
                res = ac.search(line)
                gen_pos_list = []
                gen_str_list = []
                for word in res:
                    pos_list = res[word]
                    for pos in pos_list:
                        gen_pos_list.append(pos)
                        gen_str_list.append("<"+mp_word2type[word]+">")
                #根据字典将原始句子泛化
                gen_line = self.replace_by_position(line, gen_pos_list, gen_str_list)
                #将泛化句子随机实例化
                out_lines =  self.generate_from_gen(gen_line, 
                                                    type_ac,
                                                    mp_type2word,
                                                    50)
                for line in out_lines:
                    res = ac.search(line)
                    mp = {}
                    for word in res:
                        pos_list = res[word]
                        for pos in pos_list:
                            mp[pos[0]] = (pos[1], mp_word2type[word])
                    char_list = []
                    tag_list = []
                    idx = 0
                    while idx < len(line):
                        if idx not in mp:
                            char_list.append(line[idx])
                            tag_list.append('O')
                            idx += 1
                        else:
                            for idy in range(idx, mp[idx][0]):
                                char_list.append(line[idy])
                                if idy == idx:
                                    tag_list.append("B-%s"%mp[idx][1])
                                else:
                                    tag_list.append("I-%s"%mp[idx][1])
                            idx = mp[idx][0]
                    result.append((char_list,tag_list))
        with open(out_path,'w') as f_out:
            for (char_list,tag_list) in result:
                for idx, item in enumerate(char_list):
                    f_out.write("%s\t%s\n"%(char_list[idx],tag_list[idx]))
                f_out.write("\n")

    def convert_class_to_ner(self):
        """ 分类模型的语料转成ner模型的语料
            用到泛化字典（实体->类型）:./conf/gen
        """
        #BIO
        gen_path = './conf/gen'
        cls_data_path = './data/match/intent.csv'
        tmp_path = './data/ner/intent_ner_tmp.csv'
        out_path = './data/ner/intent_ner.csv'
        mp_type2word = defaultdict(list)
        mp_word2type = {}
        for line in open(gen_path):
            arr = line.strip().split('\t')
            if len(arr) < 2: continue
            mp_type2word["<"+arr[1].strip()+">"].append(arr[0].strip())
            mp_word2type[arr[0].strip()] = arr[1].strip()

        self.process_class_data(mp_type2word, cls_data_path, tmp_path)
        self.generate_ner_data(mp_word2type, mp_type2word, tmp_path, out_path)

if __name__ == '__main__':
    util =  NERUtil()
    util.convert_class_to_ner()


