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
        for idx,line in enumerate(lines):
            line = line.strip()
            if line != '':
                if '\t' in line:
                    [char, label] = line.split('\t')
                else:
                    [char, label] = line.split()
                sent_.append(char)
                tag_.append(label)
            else:
                data.append(' '.join(sent_))
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

class DGNERUtil():

    def load_data(self, path):
        #将dg(daguan)数据转为BMES格式
        res_list = []
        maxlen = 0
        lensum = 0
        for idx,line in enumerate(open(path)):
            arr = line.strip().split()
            texts = []
            labels = []
            for item in arr:
                words = item.split('/')[0]
                if words != '':
                    char_list = words.split('_')
                else:
                    char_list = []
                word_type = item.split('/')[1]
                tmp_labels = []
                if word_type != 'o':
                    if len(char_list) == 0:
                        continue
                    if len(char_list) == 1:
                        tmp_labels.append('S-'+word_type)
                    else:
                        for char in char_list:
                            tmp_labels.append('M-'+word_type)
                        tmp_labels[0] = 'B-'+word_type
                        tmp_labels[-1] = 'E-'+word_type
                else:
                    for char in char_list:
                        tmp_labels.append('O')
                texts += char_list
                labels += tmp_labels
            if len(texts) > maxlen: 
                maxlen = len(texts)
            lensum+=len(texts)
            res_list.append('\n'.join([texts[i]+'\t'+labels[i] for i in
                                       range(len(texts))]))
        with open(path+'.bmes.txt','w') as f_out:
            for item in res_list:
                f_out.write(item+"\n\n")

    def load_test_data(self,path):
        f_out = open(path+'.bmes.txt','w')
        for line in open(path):
            new_line = ' '.join(line.strip().split('_'))
            f_out.write(new_line+"\n")
        f_out.close()

    def convert_bmes_to_dg(self, file, maxlen=128):
        #将BIO格式重新转回dg需要的格式
        with open(file) as f_in, open(file+'.out.txt','w') as f_out:
            lines = f_in.readlines()
            out_lines = []
            res = []
            words = []
            tags = []
            for idx,line in enumerate(lines):
                if line.strip()=="" and idx != 0:
                    out_lines.append(' '.join(words))
                    res.append(tags)
                    words = []
                    tags = []
                else:
                    words.append(line.strip().split('\t')[0])
                    tags.append(line.strip().split('\t')[1])

            for idx,line in enumerate(out_lines):
                char_list = line.split()
                tag_list = res[idx][:len(char_list)]
                result = []
                tmp = []
                ctype = ""
                for idy,char in enumerate(char_list):
                    if idy == len(char_list) - 1:
                        last_ctype = tag_list[idy][2] if tag_list[idy] != 'O' else 'o'
                        if last_ctype == ctype:
                            tmp.append(char)
                            result.append('_'.join(tmp)+'/'+ctype)
                        else:
                            if tmp != []:
                                result.append('_'.join(tmp)+'/'+ctype)
                            result.append(char+'/'+last_ctype)

                    if tag_list[idy].startswith("E-") \
                            or tag_list[idy].startswith("S-"):
                        tmp.append(char)
                        if ctype == "":ctype = tag_list[idy][2]
                        result.append('_'.join(tmp)+'/'+ctype)
                        tmp = []
                        ctype = ""
                    elif tag_list[idy].startswith("M-"):
                        tmp.append(char)
                        ctype = tag_list[idy][2]
                    elif tag_list[idy].startswith("B-"):
                        if len(tmp) >= 1:
                            if ctype == "":ctype = tag_list[idy-1][2]
                            result.append('_'.join(tmp)+'/'+ctype)
                            tmp = []
                            ctype = ""
                        tmp.append(char)
                        ctype = tag_list[idy][2]
                    elif tag_list[idy].startswith("O"):
                        if len(tmp) >= 1 and ctype != "o":
                            if ctype == "":ctype = tag_list[idy-1][2]
                            result.append('_'.join(tmp)+'/'+ctype)
                            tmp = []
                            ctype = ""
                        ctype = 'o'
                        tmp.append(char)
                    else:
                        print("error occured in line %s"%idx)
                        pdb.set_trace()
                f_out.write('  '.join(result)+'\n')

    def __call__(self, text):
        text_list  = [text]
        return self.test(text_list)[0]


    def process(self):
        #self.load_data(path='../data/ner/daguan/ori/train.txt')
        #self.load_test_data(path='../data/ner/daguan/ori/test.txt')
        self.convert_bmes_to_dg("../data/ner/daguan/test.txt.out.txt")


if __name__ == '__main__':
    #将分类模型格式的数据转换为ner模型所需要的BIO格式
    #util =  NERUtil()
    #util.convert_class_to_ner()


    #daguan比赛数据转换为BMES
    util =  DGNERUtil()
    util.process()




