import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb
from common.layers import rnn_layer

class RNN(object):
    def __init__(self, **args):
        self.maxlen = args['maxlen']
        self.num_hidden = args['num_hidden'] if 'num_hidden' in args else 256
        self.num_layers = args['num_layers'] if 'num_layers' in args else 2
        self.keep_prob = args['keep_prob']
        self.batch_size = args['batch_size']
        self.rnn_type = args['rnn_type']
        self.num_output = args['num_output']
        self.placeholder = {}

    def __call__(self, embed, scope_name = 'encoder', ner_flag = False, reuse = tf.AUTO_REUSE):
        length_name = scope_name + "_length" 
        self.placeholder[length_name] = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name = length_name)
        with tf.variable_scope("rnn", reuse = reuse):
            #for gru,lstm outputs:[batch_size, max_time, num_hidden]
            #for bi_gru,bi_lstm outputs:[batch_size, max_time, num_hidden*2]
            outputs, state = rnn_layer(inputs = embed,
                              seq_len = self.placeholder[length_name], 
                              num_hidden = self.num_hidden,
                              num_layers = self.num_layers,
                              rnn_type = self.rnn_type,
                              keep_prob = self.keep_prob)
            #flatten:
            outputs_shape = outputs.shape.as_list()
            if ner_flag:
                outputs = tf.reshape(outputs, [-1, outputs_shape[2]])
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
                dense = tf.reshape(dense, [-1, outputs_shape[1], self.num_output])
            else:
                outputs = tf.reshape(outputs, [-1, outputs_shape[1]*outputs_shape[2]])
                dense = tf.layers.dense(outputs, self.num_output, name='fc')
            #使用最后一个time的输出
            #outputs = outputs[:, -1, :]

            return dense

    def feed_dict(self, scope_name = 'encoder', **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = key + "_length" 
            feed_dict[self.placeholder[length_name]] = kwargs[key]

        return feed_dict

    def pb_feed_dict(self, graph, scope_name = 'encoder', **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = key + "_length" 
            key_node = graph.get_operation_by_name(length_name).outputs[0]
            feed_dict[key_node] = kwargs[key]
        return feed_dict

