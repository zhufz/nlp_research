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
        self.num_hidden = kwargs['num_hidden'] if 'num_hidden' in kwargs else 256
        self.num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        self.rnn_type = kwargs['rnn_type']
        self.rnn_encode_layer = RNNLayer(self.rnn_type, self.num_hidden, self.num_layers)
        self.rnn_decode_layer = RNNLayer(self.rnn_type, self.num_hidden, self.num_layers)
        self.placeholder = {}

    def __call__(self, net_encode, net_decode, name = 'seq2seq', 
                 features = None,  reuse = tf.AUTO_REUSE, **kwargs):
        #def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
        length_encode_name = name + "_encode_length" 
        length_decode_name = name + "_decode_length" 
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

        outputs, final_state_encode  = self.rnn_encode_layer(inputs = net_encode, 
                                                     seq_len = self.placeholder[length_encode_name],
                                                     name = 'encode')
        # TODO: 修复decoder依赖encoder无法单个预测问题
        outputs, final_state_decode  = self.rnn_decode_layer(inputs = net_decode, 
                                                     seq_len = self.placeholder[length_decode_name], 
                                                     initial_state = final_state_encode,
                                                     name = 'decode')
        outputs_shape = outputs.shape.as_list()
        outputs = tf.reshape(outputs, [-1, outputs_shape[2]])
        dense = tf.layers.dense(outputs, self.num_output, name='fc')
        #[batch_size, max_time, num_output]
        dense = tf.reshape(dense, [-1, outputs_shape[1], self.num_output])

        return dense

