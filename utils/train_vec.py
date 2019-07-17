#-*- coding:utf-8 -*-
import gensim
import sys
import pdb

def train(sentences, save_path = "word2vec/vocabulary", embedding_size = 128):
    word_set = set()
    for word_list in sentences:
        for word in word_list:
            word_set.add(word)
    vocabulary_size = len(word_set)

    model = gensim.models.Word2Vec(sentences, size=embedding_size, 
        window=5, min_count=0, iter=20, sg=1,  max_vocab_size=vocabulary_size)
    model.wv.save_word2vec_format(save_path ,binary=False)

def read(file_name, split_char = None):
    ret = []
    for line in open(file_name):
        line = line.strip()
        if split_char in [None," "]:
            word_list = line.split()
        else:
            word_list = line.split(split_char)
        ret.append(word_list)
    return ret

if __name__ == '__main__':
    #arg0:file_name
    #arg1:split_char
    # python3 common/train_vec.py data/ner/*.txt " " 
    file_name = sys.argv[1] 
    sentences = read(file_name,sys.argv[2])
    save_path = file_name+".vec"
    train(sentences, save_path, 128)



