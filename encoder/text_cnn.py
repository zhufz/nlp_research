import tensorflow as tf
from common.layers import get_initializer
from encoder import Base
#from kim cnn 2014:https://arxiv.org/abs/1408.5882


class TextCNN(Base):
    def __init__(self, **args):
        self.maxlen = args['maxlen']
        self.embedding_size = args['embedding_size']
        self.keep_prob = args['keep_prob']
        self.num_output = args['num_output']
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE, **kwargs):
        #input: [batch_size, sentence_len, embedding_size,1]
        #output:[batch_size, num_output]
        with tf.variable_scope("text_cnn", reuse = reuse):
            embed = tf.expand_dims(embed, -1)
            conv_outputs = []
            for i, size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv%d" % i, reuse = reuse):
                    # Convolution Layer begins
                    conv_filter = tf.get_variable(
                        "conv_filter%d" % i, [size, self.embedding_size, 1, self.num_filters],
                        initializer=get_initializer(type = 'random_uniform',
                                                    minval = -0.01, 
                                                    maxval = 0.01)
                    )
                    bias = tf.get_variable(
                        "conv_bias%d" % i, [self.num_filters],
                        initializer=get_initializer(type = 'zeros')
                    )
                    output = tf.nn.conv2d(embed, conv_filter, [1, 1, 1, 1], "VALID") + bias
                    # Applying non-linearity
                    output = tf.nn.relu(output)
                    # Pooling layer, max over time for each channel
                    output = tf.reduce_max(output, axis=[1, 2])
                    conv_outputs.append(output)

            # Concatenate all different filter outputs before fully connected layers
            conv_outputs = tf.concat(conv_outputs, axis=1)
            #total_channels = conv_outputs.get_shape()[-1]
            h_pool_flat = tf.reshape(conv_outputs, [-1, self.num_filters * len(self.filter_sizes)])
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)
            #dense = tf.layers.dense(h_drop, self.num_output, activation=None)
            dense = tf.layers.dense(h_pool_flat, 
                                    self.num_output, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                    activation=None,
                                    reuse = reuse)
                #logits = tf.layers.dense(mean_sentence,
                #                        self.num_output,
                #                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                #                        name='fc',
                #                        reuse = reuse)
        return dense

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
