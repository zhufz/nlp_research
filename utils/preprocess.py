# -*-coding:utf-8 -*-
import jieba
import pdb
import pandas as pd
import sys,os
import logging
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.ac import AC
from collections import defaultdict

def word_tokenize(text):
    return text.split()

def char_tokenize(text):
    words = text.split()
    chars = []
    for word in words:
        if word[0]=='<' and word[-1]=='>':
            chars.append(word)
        else:
            for char in word:
                chars.append(char)
    return chars

def clean_str(text):
    #text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    #text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text


class Preprocess():
    def __init__(self):
        gen_path = os.path.join(ROOT_PATH, './conf/gen')
        #gen_path = './conf/gen'
        self.mp = self.load_gen(path = gen_path)
        self.ac = AC()
        for key in self.mp:
            self.ac.add(key)

    def load_gen(self, path):
        mp = {}
        if os.path.exists(path):
            lines = [line.strip() for line in open(path).readlines()]
            for line in lines:
                arr = line.split('\t')
                if len(arr) <2:continue
                mp[arr[0]] = "<"+arr[1]+">"
        else:
            logging.warn("gen file not found!")
        return mp


    def segment(self, text):
        if type(text) != str or text.strip() == '': return []
        seg_list = jieba.cut(text, cut_all=False)
        return list(seg_list)

    def get_map(self, buffer):
        if buffer in self.mp:
            return self.mp[buffer]
        return buffer

    def is_digit(self, word):
        for char in word:
            if char in set(['1','2','3','4','5','6','7','8','9','0',\
                 '一','二','三','四','五','六','七','八','九','零']):
                continue
            else:
                return False
        return True


    def generalization(self, seg_list):
        for idx in range(len(seg_list)):
            if self.is_digit(seg_list[idx]):
                seg_list[idx] = "<NUM>"
        #segment based generalization
        #for idx in range(len(seg_list)):
        #    item = seg_list[idx]
        #    if item in self.mp:
        #        seg_list[idx] = self.mp[item]

        #match based generalization
        seg_bound = set()
        pos = 0
        for item in seg_list:
            seg_bound.add(pos)
            seg_bound.add(pos+len(item)-1)
            pos += len(item)
        #匹配结果
        #pdb.set_trace()
        res = self.ac.search(''.join(seg_list))
        #print('search result:',res)
        #过滤非分词边界
        new_res = defaultdict(list)
        for word in res:
            positions = res[word]
            for position in positions:
                if position[0] in seg_bound and position[1]-1 in seg_bound:
                    new_res[word].append(position)

        flag = []
        start = 0
        inc = 0
        #打基于分词的flag标记
        for seg in seg_list:
            for idx in range(start, start+len(seg)):
                flag.append(inc)
            start += len(seg)
            inc +=1
        inc +=1
        #刷新基于匹配结果的flag标记
        for key in new_res:
            for position in new_res[key]:
                for idx in range(position[0], position[1]):
                    flag[idx] = inc
                inc +=1
        ret = []
        buffer = ""
        sentence = ''.join(seg_list)
        #根据匹配结果重新拆分句子
        for idx in range(len(sentence)):
            if idx == 0:
                buffer = sentence[idx]
            elif flag[idx] != flag[idx-1]:
                ret.append(self.get_map(buffer))
                buffer = sentence[idx]
            else:
                buffer += sentence[idx]
            if idx == len(sentence)-1:
                ret.append(self.get_map(buffer))
        #print(flag)
        #print(sentence)
        #print(ret)
        #pdb.set_trace()
        return ret

    def merge_gene(self, seg_list):
        if len(seg_list) == 0: return seg_list
        new_list = []
        flag = False
        tmp = ""
        for item in seg_list:
            if item != "<" and flag == False:
                new_list.append(item)
            elif item == '>':
                tmp+=item
                new_list.append(tmp)
                tmp = ""
                flag = False
            elif item == '<' or flag == True:
                tmp+=item
                flag = True

        return new_list

    def bigram(self, seg_list):
        if len(seg_list) == 0: return seg_list
        bi_list = []
        for idx in range(len(seg_list)-1):
            a = seg_list[idx]
            b = seg_list[idx+1]
            if a > b:
                b, a = a, b
            bi_list.append(a+"_"+b)
        return bi_list

    def skipgram(self, seg_list):
        if len(seg_list) == 0: return seg_list
        skip_list = []
        for idx in range(len(seg_list)-2):
            for idy in range(idx+2,len(seg_list)):
                a = seg_list[idx]
                b = seg_list[idy]
                if a > b:
                    b, a = a, b
                skip_list.append(a+"|"+b)
        return skip_list


    def process(self, text_list):
        ret = []
        for text in text_list:
            seg_list = self.segment(text)
            new_list = self.merge_gene(seg_list)
            new_list = self.generalization(new_list)
            bi_list = self.bigram(new_list)
            skip_list = self.skipgram(new_list)
            ret.append(new_list+bi_list+skip_list)
        return ret

    def get_dl_input_by_file(self, file):
        dt = pd.read_csv(file, header = 0)
        ret = []
        for text in dt['text']:
            ret.append(self.get_dl_input_by_text(text))
        dataframe = pd.DataFrame({'text':ret,'intent':dt['intent']})
        dataframe.to_csv(file+'.feature.csv',sep=',',columns=['text','intent'])

    def get_dl_input_by_text(self, text):
        seg_list = self.segment(text)
        new_list = self.merge_gene(seg_list)
        new_list = self.generalization(new_list)
        return ' '.join(new_list)


if __name__ == '__main__':
    pre = Preprocess()
    #seg_list = pre.segment('给我把<设备名>打开')
    #print(seg_list)
    #new_list = pre.merge_gene(seg_list)
    #print(new_list)
    #bi_list = pre.bigram(new_list)
    #print(bi_list)
    #print(pre.process(["你大爷的"]))
    #print(pre.process(["智能电茶壶煮茶"]))
    #print(pre.process(["停"]))
    pre.get_dl_input_by_file('./data/intent_train.csv')
    pre.get_dl_input_by_file('./data/intent_test.csv')
