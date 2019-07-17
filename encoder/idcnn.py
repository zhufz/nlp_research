#-*- coding:utf-8 -*-
import tensorflow as tf
from encoder import EncoderBase
import pdb
import copy

class IDCNN(EncoderBase):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        super(IDCNN, self).__init__(**kwargs)
        self.embedding_dim = kwargs['embedding_size']
        self.maxlen = kwargs['maxlen']
        self.keep_prob = kwargs['keep_prob']
        self.filter_width = 3
        self.num_filter = 100
        self.repeat_times = 4
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 3
            },
            {
                'dilation': 5
            },
        ]
        self.placeholder = {}

    def __call__(self, embed, name = 'idcnn', features = None, 
                 middle_flag = False,reuse = tf.AUTO_REUSE, **kwargs):

        """
        :param idcnn_inputs: [batch_size, maxlen, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        with tf.variable_scope("idcnn", reuse = reuse):
            model_inputs = tf.expand_dims(embed, 1)
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            #print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=tf.contrib.layers.xavier_initializer())
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            final_out_from_layers = []
            total_width_for_last_dim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if i == (len(self.layers) - 1):
                            final_out_from_layers.append(conv)
                            total_width_for_last_dim += self.num_filter
                        layerInput = conv
            final_out = tf.concat(axis=3, values=final_out_from_layers)
            final_out = tf.nn.dropout(final_out, self.keep_prob)
            final_out = tf.squeeze(final_out, [1])
            if middle_flag:
                final_out = tf.reshape(final_out, [-1, self.maxlen, total_width_for_last_dim])
                dense = tf.layers.dense(final_out, self.num_output, name='fc')
                return dense
            final_out = tf.reshape(final_out, [-1, self.maxlen * total_width_for_last_dim])
            logits = tf.layers.dense(final_out,
                                    self.num_output,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                    name='fc',
                                    reuse = reuse)
            return logits
