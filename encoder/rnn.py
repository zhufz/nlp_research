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
        self.maxlen = kwargs['maxlen']
        self.num_hidden = kwargs['num_hidden'] if 'num_hidden' in kwargs else 256
        self.num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        self.keep_prob = kwargs['keep_prob']
        self.batch_size = kwargs['batch_size']
        self.rnn_type = kwargs['rnn_type']
        self.num_output = kwargs['num_output']
        self.rnn_layer = RNNLayer(self.rnn_type, 
                                  self.num_hidden,
                                  self.num_layers)
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

            outputs, _, state = self.rnn_layer(inputs = embed,
                              seq_len = self.placeholder[length_name])
            #flatten:
            outputs_shape = outputs.shape.as_list()
            if middle_flag:
                outputs = tf.reshape(outputs, [-1, outputs_shape[2]])
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
                #[batch_size, max_time, num_output]
                dense = tf.reshape(dense, [-1, outputs_shape[1], self.num_output])
            else:
                outputs = tf.reshape(outputs, [-1, outputs_shape[1]*outputs_shape[2]])
                #[batch_size, num_output]
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
            #使用最后一个time的输出
            #outputs = outputs[:, -1, :]
            if hidden_flag:
                return dense, state, self.rnn_layer.pb_nodes
            else:
                return dense

    def feed_dict(self, name = 'encoder', initial_state = None,  **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = name + "_length" 
            feed_dict[self.placeholder[length_name]] = kwargs[key]
        #初始状态值传入
        if initial_state != None:
            feed_dict.update(self.rnn_layer.feed_dict(initial_state))
        return feed_dict

    def pb_feed_dict(self, graph, name = 'encoder', initial_state = None, **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = name + "_length" 
            key_node = graph.get_operation_by_name(length_name).outputs[0]
            feed_dict[key_node] = kwargs[key]
        #初始状态值传入
        if initial_state != None:
            feed_dict.update(self.rnn_layer.feed_dict(initial_state, graph))
        return feed_dict

    def get_features(self, name = 'encoder'):
        features = {}
        length_name = name + "_length" 
        features[length_name] = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name = length_name)
        return features
