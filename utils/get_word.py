#-*- coding:utf-8 -*-
import jieba
import gensim
import sys

def segment(text):
    if text.strip() == '': return []
    seg_list = jieba.cut(text, cut_all=False)
    return list(seg_list)


def get_word(file):
    lines = [line.strip() for line in open(file).readlines()]
    words = set()
    for line in lines:
        words = words.union(set(segment(line)))
    return words

    #with open(file+'.words','w') as f_w:
    #    for item in words:
    #        f_w.write("{}\n".format(item))

def subsample_embedding(path, words):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    with open(path+'.sub','w') as f_w:
        cnt = 0

        for item in words:
            if item in model:
                if item.strip() == '':continue
                cnt += 1
                vec = [str(x) for x in model[item]]
                length = len(vec)
                f_w.write("{} {}\n".format(item, " ".join(vec)))
        f_w.seek(0)
        f_w.write("{} {}\n".format(cnt, length))

def get_char_embedding(path):
    with open(path) as f, open(path+'.sub','w') as f_w:
        for idx, line in enumerate(f):
            if idx ==0:continue
            if len(line.split()[0]) ==1:
                f_w.write(line)

if __name__ == "__main__":
    #words = get_word(sys.argv[1])
    #subsample_embedding(sys.argv[2], words)
    get_char_embedding(sys.argv[1])



