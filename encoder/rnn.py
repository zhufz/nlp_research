#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb
from common.layers import RNNLayer
from encoder import EncoderBase
import copy

class RNN(EncoderBase):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.num_hidden = kwargs['num_hidden'] if 'num_hidden' in kwargs else 256
        self.num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        self.is_training = kwargs['is_training']
        self.keep_prob = kwargs['keep_prob']
        self.rnn_type = kwargs['rnn_type']
        self.cell_type = kwargs['cell_type']
        self.maxlen = kwargs['maxlen']
        self.rnn_layer = RNNLayer(self.rnn_type, 
                                  self.num_hidden,
                                  self.num_layers,
                                  self.cell_type,
                                  use_attention = True)
        self.placeholder = {}

    def __call__(self, embed, name = 'encoder', middle_flag = False, hidden_flag
                 = False,  features = None, reuse = tf.AUTO_REUSE, **kwargs):
        #middle_flag: if True return middle output for each time step
        #hidden_flag: if True return hidden state
        length_name = name + "_length" 
        self.placeholder[length_name] = tf.placeholder(dtype=tf.int32, 
                                                shape=[None], 
                                                name = length_name)
        if features != None:
            self.placeholder[length_name] = features[length_name]

        self.initial_state = None
        with tf.variable_scope("rnn", reuse = reuse):
            #for gru,lstm outputs:[batch_size, max_time, num_hidden]
            #for bi_gru,bi_lstm outputs:[batch_size, max_time, num_hidden*2]

            outputs, state = self.rnn_layer(inputs = embed,
                                            seq_len = self.placeholder[length_name],
                                            maxlen = self.maxlen)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            #flatten:
            outputs_shape = outputs.shape.as_list()
            if middle_flag:
                outputs = tf.reshape(outputs, [-1, outputs_shape[2]])
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
                dense = tf.nn.dropout(dense, self.keep_prob)
                #[batch_size, max_time, num_output]
                dense = tf.reshape(dense, [-1, outputs_shape[1], self.num_output])
            else:
                outputs = tf.reshape(outputs, [-1, outputs_shape[1]*outputs_shape[2]])
                #[batch_size, num_output]
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
            #使用最后一个time的输出
            #outputs = outputs[:, -1, :]
            if hidden_flag:
                return dense, state
            else:
                return dense

    def get_features(self, name = 'encoder'):
        features = {}
        length_name = name + "_length" 
        features[length_name] = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name = length_name)
        return features
