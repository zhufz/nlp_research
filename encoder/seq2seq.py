#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb
from common.layers import RNNLayer
from encoder import EncoderBase
import copy

class Seq2seq(EncoderBase):
    def __init__(self, **kwargs):
        super(Seq2seq, self).__init__(**kwargs)
        self.maxlen = kwargs['maxlen']
        self.num_hidden = kwargs['num_hidden'] if 'num_hidden' in kwargs else 256
        self.num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        self.keep_prob = kwargs['keep_prob']
        self.batch_size = kwargs['batch_size']
        self.num_output = kwargs['num_output']
        self.rnn_type = kwargs['rnn_type']
        self.rnn_encode_layer = RNNLayer(self.rnn_type, self.num_hidden, self.num_layers)
        self.rnn_decode_layer = RNNLayer(self.rnn_type, self.num_hidden, self.num_layers)
        self.placeholder = {}

    def __call__(self, net_encode, net_decode, name = 'seq2seq', 
                 features = None,  reuse = tf.AUTO_REUSE, **kwargs):
        #def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
        length_encode_name = name + "_encoder_length" 
        length_decode_name = name + "_decoder_length" 
        self.placeholder[length_encode_name] = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name = length_encode_name)
        self.placeholder[length_decode_name] = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name = length_decode_name)
        if features != None:
            self.features = copy.copy(self.placeholder)
            self.placeholder[length_encode_name] = features[length_encode_name]
            self.placeholder[length_decode_name] = features[length_decode_name]

        outputs, final_state_encode, final_state_encode_for_feed = self.rnn_encode_layer(inputs = net_encode, 
                                                     seq_len = self.placeholder[length_encode_name],
                                                     name = 'encoder')
        # TODO: 修复decoder依赖encoder无法单个预测问题
        outputs, final_state_decode, final_state_decode_for_feed  = self.rnn_decode_layer(inputs = net_decode, 
                                                     seq_len = self.placeholder[length_decode_name], 
                                                     initial_state = final_state_encode,
                                                     name = 'decoder')
        outputs_shape = outputs.shape.as_list()
        outputs = tf.reshape(outputs, [-1, outputs_shape[2]])
        dense = tf.layers.dense(outputs, self.num_output, name='fc')
        #[batch_size, max_time, num_output]
        dense = tf.reshape(dense, [-1, outputs_shape[1], self.num_output])

        return dense, final_state_encode_for_feed, \
            final_state_decode_for_feed, self.rnn_decode_layer.pb_nodes

    def feed_dict(self, name = 'seq2seq', initial_state = None, **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_encode_name = name + "_encoder_length" 
            length_decode_name = name + "_decoder_length" 
            if kwargs[key][0] != None:
                feed_dict[self.placeholder[length_encode_name]] = kwargs[key][0]
            if kwargs[key][1] != None:
                feed_dict[self.placeholder[length_decode_name]] = kwargs[key][1]
        if initial_state != None:
            feed_dict.update(self.rnn_decode_layer.feed_dict(initial_state))
        return feed_dict

    def pb_feed_dict(self, graph, name = 'seq2seq', initial_state = None, **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_encode_name = name + "_encoder_length" 
            length_decode_name = name + "_decoder_length" 
            key_node0 = graph.get_operation_by_name(length_encode_name).outputs[0]
            key_node1 = graph.get_operation_by_name(length_decode_name).outputs[0]
            if kwargs[key][0] != None:
                feed_dict[key_node0] = kwargs[key][0]
            if kwargs[key][1] != None:
                feed_dict[key_node1] = kwargs[key][1]
        if initial_state != None:
            feed_dict.update(self.rnn_decode_layer.feed_dict(initial_state, graph))
        return feed_dict

