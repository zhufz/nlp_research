#-*- coding:utf-8 -*-
import tensorflow as tf
import pdb
import copy
import numpy as np
import tensorflow as tf
import h5py
import json
import re
from language_model.bilm_tf.bilm import TokenBatcher, BidirectionalLanguageModel
from encoder import EncoderBase

class Elmo(EncoderBase):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        super(Elmo, self).__init__(**kwargs)
        self.embedding_dim = kwargs['embedding_size']
        self.is_training = kwargs['is_training']
        self.maxlen = kwargs['maxlen']
        self.vocab_file = kwargs['elmo_vocab_path']
        self.options_file = kwargs['elmo_options_path']
        self.placeholder = {}
        self.batcher = TokenBatcher(self.vocab_file)
        self.model = None

    def __call__(self, name = 'encoder', features = None, reuse = tf.AUTO_REUSE, **kwargs):


        if features != None:
            self.placeholder[name+'_input_ids'] = features[name+'_input_ids']
        else:
            self.placeholder[name+'_input_ids']= tf.placeholder('int32',
                                                shape=(None, self.maxlen),
                                                name = name+"_input_ids")
        with open(self.options_file, 'r') as fin:
            options = json.load(fin)

        if self.model == None:
            self.model = BidirectionalLanguageModel(options,
                                           use_character_inputs=False)
        ops = self.model(self.placeholder[name+'_input_ids'])

        out = ops['lm_embeddings']
        out = tf.concat([out,out[:,:,-2:,:]],2)
        #output = tf.add_n([out[:,0],out[:,1]])  # sum embedding of the three layers
        print(out.shape)
        output = tf.squeeze(tf.add_n(tf.split(out,num_or_size_splits=out.shape[1],axis=1)),1)

        dense = tf.layers.dense(output, 
                                self.num_output,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                activation=None,
                                reuse = reuse)
        return dense

    def build_ids(self, text, **kwargs):
        sentence = text.strip().split()
        char_ids = self.batcher.batch_sentences([sentence], self.maxlen)
        return char_ids[0]

    def encoder_fun(self, x_query_raw, name = 'encoder', **kwargs):
        flag = True
        if type(x_query_raw) != list:
            flag = False
            x_query_raw = [x_query_raw]

        input_ids = []
        for idx, item in enumerate(x_query_raw):
            tmp_input_ids = self.build_ids(x_query_raw[idx])
            input_ids.append(tmp_input_ids)
        if flag == False:
            input_ids = input_ids[0]
        return {name+"_input_ids": input_ids}

    def keys_to_features(self, name = 'encoder'):
        keys_to_features = {
            name+"_input_ids": tf.FixedLenFeature([self.maxlen], tf.int64), 
        }
        return keys_to_features

    def parsed_to_features(self, parsed, name = 'encoder'):
        ret = {
            name + "_input_ids": tf.reshape(parsed[name+ "_input_ids"], [self.maxlen]), 
        }
        return ret

    def get_features(self, name = 'encoder'):
        features = {}
        features[name+'_input_ids'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_input_ids")
        return features
