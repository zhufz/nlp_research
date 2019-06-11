#-*- coding:utf-8 -*-
import tensorflow as tf
from encoder import EncoderBase
#simple CNN demo
class CNN(EncoderBase):
    def __init__(self, **kwargs):
        super(DCNN, self).__init__(**kwargs)
        self.document_max_len = kwargs['maxlen']
        self.embedding_size = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE, **kwargs):
        #input: [batch_size, sentence_len, embedding_size,1]
        #output:[batch_size, num_filters*filter_sizes]
        embed = tf.expand_dims(embed, -1)
        pooled_outputs = []
        with tf.variable_scope("cnn", reuse = reuse):
            for filter_size in self.filter_sizes:
                conv = tf.layers.conv2d(
                    embed,
                    filters=self.num_filters,
                    kernel_size=[filter_size, self.embedding_size],
                    strides=(1, 1),
                    padding="VALID",
                    activation=tf.nn.relu)
                pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=[self.document_max_len - filter_size + 1, 1],
                    strides=(1, 1),
                    padding="VALID")
                pooled_outputs.append(pool)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)
            dense = tf.layers.dense(h_drop, self.num_output, activation=None)
            return dense

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
