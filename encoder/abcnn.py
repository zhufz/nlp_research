import tensorflow as tf
import numpy as np
import pdb
from common.layers import get_initializer
from encoder import Base
import copy
#refer:https://github.com/galsang/ABCNN/blob/master/ABCNN.py


class ABCNN(Base):
    def __init__(self, **kwargs):
        """
        Implmenentaion of ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)
        :param maxlen: sentence length
        :param num_output: The dimension of output.
        :param embedding_size: The size of embedding.
        """
        super(ABCNN, self).__init__(**kwargs)
        self.keep_prob = kwargs['keep_prob']
        self.batch_size = kwargs['batch_size']
        self.num_output = kwargs['num_output']
        self.s = kwargs['maxlen']
        self.d0 = kwargs['embedding_size']
        self.w = 4
        self.l2_reg = 0.0004
        self.model_type = "ABCNN3"  #(ABCNN1, ABCNN2, ABCNN3)

        self.di = 50
        self.num_layers = 2
        self.placeholder = {}



        # zero padding to inputs for wide convolution
    def pad_for_wide_conv(self, x):
        return tf.pad(x, np.array([[0, 0], [0, 0], [self.w - 1, self.w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1)+1e-8)
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1)+1e-8)
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)

    def euclidean_score(self, v1, v2):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
        return 1 / (1 + euclidean)

    def make_attention_mat(self, x1, x2):
        # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
        # x2 => [batch, height, 1, width]
        # [batch, width, wdith] = [batch, s, s]
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 -
                                                    tf.matrix_transpose(x2)),
                                          axis=1)+1e-8)
        return 1 / (1 + euclidean)

    def convolution(self, name_scope, x, d, reuse):
        with tf.variable_scope("conv", reuse) as scope:
            #conv = tf.contrib.layers.conv2d(
            #    inputs=x,
            #    num_outputs=self.di,
            #    kernel_size=(d, self.w),
            #    stride=1,
            #    padding="VALID",
            #    activation_fn=tf.nn.tanh,
            #    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
            #    biases_initializer=tf.constant_initializer(1e-04),
            #    reuse=reuse,
            #    trainable=True,
            #    scope=scope
            #)
            conv = tf.layers.conv2d(x, self.di, [d, self.w])

            # Weight: [filter_height, filter_width, in_channels, out_channels]
            # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

            # [batch, di, s+w-1, 1]
            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
            return conv_trans

    def w_pool(self, variable_scope, x, attention, reuse):
        # x: [batch, di, s+w-1, 1]
        # attention: [batch, s+w-1]
        with tf.variable_scope(variable_scope + "-w_pool", reuse = reuse):
            if self.model_type == "ABCNN2" or self.model_type == "ABCNN3":
                pools = []
                # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                for i in range(self.s):
                    # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                    pools.append(tf.reduce_sum(x[:, :, i:i + self.w, :] *
                                               attention[:, :, i:i + self.w, :],
                                               axis=2,
                                               keep_dims=True))

                # [batch, di, s, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap")
            else:
                w_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, self.w),
                    strides=1,
                    padding="VALID",
                    name="w_ap"
                )
                # [batch, di, s, 1]

            return w_ap

    def all_pool(self, variable_scope, x, reuse):
        with tf.variable_scope(variable_scope + "-all_pool", reuse = reuse):
            if variable_scope.startswith("input"):
                pool_width = self.s
                d = self.d0
            else:
                pool_width = self.s + self.w - 1
                d = self.di

            all_ap = tf.layers.average_pooling2d(
                inputs=x,
                # (pool_height, pool_width)
                pool_size=(1, pool_width),
                strides=1,
                padding="VALID",
                name="all_ap"
            )
            # [batch, di, 1, 1]

            # [batch, di]
            all_ap_reshaped = tf.reshape(all_ap, [-1, d])
            #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

            return all_ap_reshaped

    def CNN_layer(self, variable_scope, x1, x2, d, reuse):
        # x1, x2 = [batch, d, s, 1]
        with tf.variable_scope(variable_scope, reuse = reuse):
            if self.model_type == "ABCNN1" or self.model_type == "ABCNN3":
                with tf.name_scope("att_mat"):
                    #[s, d]
                    aW = tf.get_variable(name="aW",
                                         shape=(self.s, d),
                                         initializer=get_initializer(type='xavier'),
                                         regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))

                    #[batch, s, s]
                    att_mat = self.make_attention_mat(x1, x2)

                    x1_a = \
                        tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl",
                                                                 att_mat, aW)), -1)
                    x2_a = tf.expand_dims(tf.matrix_transpose(
                        tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat),
                                  aW)), -1)
                    # x1_a: [batch, d, s, 1]
                    # x2_a: [batch, d, s, 1]

                    # [batch, d, s, 2]
                    x1 = tf.concat([x1, x1_a], axis=3)
                    x2 = tf.concat([x2, x2_a], axis=3)

            left_conv = self.convolution(name_scope="left", x=self.pad_for_wide_conv(x1), d=d, reuse=False)
            right_conv = self.convolution(name_scope="right", x=self.pad_for_wide_conv(x2), d=d, reuse=True)

            left_attention, right_attention = None, None

            if self.model_type == "ABCNN2" or self.model_type == "ABCNN3":
                # [batch, s+w-1, s+w-1]
                att_mat = self.make_attention_mat(left_conv, right_conv)
                # [batch, s+w-1], [batch, s+w-1]
                left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

            left_wp = self.w_pool(variable_scope="left", x=left_conv, attention=left_attention, reuse = reuse)
            left_ap = self.all_pool(variable_scope="left", x=left_conv, reuse = reuse)
            right_wp = self.w_pool(variable_scope="right", x=right_conv, attention=right_attention, reuse = reuse)
            right_ap = self.all_pool(variable_scope="right", x=right_conv, reuse = reuse)

            return left_wp, left_ap, right_wp, right_ap

    def feed_dict(self, x_query, x_sample):
        feed_dict = {}
        feed_dict[self.placeholder['len1']] = x_query
        feed_dict[self.placeholder['len2']] = x_sample
        return feed_dict

    def pb_feed_dict(self, graph, x_query, x_sample):
        feed_dict = {}
        feed_dict[graph.get_operation_by_name("x_query_length").outputs[0]] = x_query
        feed_dict[graph.get_operation_by_name("x_sample_length").outputs[0]] = x_sample
        return feed_dict

    def get_features(self):
        features = {}
        features['len1'] = tf.placeholder(tf.float32, shape=[None],
                                   name="x_query_length")
        features['len2'] = tf.placeholder(tf.float32, shape=[None],
                                   name="x_sample_length")
        return features

    def __call__(self, x_query, x_sample, name = 'encoder', 
                 features = None, reuse = tf.AUTO_REUSE, **kwargs):
        """
        @param x_query:[batch_size, sentence_size, embedding_size]
        @param x_sample:[batch_size, sentence_size, embedding_size]
        return:
            [batch_size, num_output]
        """
        self.placeholder['len1'] = tf.placeholder(tf.float32, shape=[None],
                                   name="x_query_length")
        self.placeholder['len2'] = tf.placeholder(tf.float32, shape=[None],
                                   name="x_sample_length")
        if features != None:
            self.placeholder['len1'] = tf.cast(features["x_query_length"],tf.float32)
            self.placeholder['len2'] = tf.cast(features["x_sample_length"],tf.float32)

        with tf.variable_scope('abcnn', reuse = reuse):
            x_query = tf.transpose(x_query, [0, 2, 1])
            x_sample = tf.transpose(x_sample, [0, 2, 1])
            x1_expanded = tf.expand_dims(x_query, -1)
            x2_expanded = tf.expand_dims(x_sample, -1)

            len1=tf.expand_dims(self.placeholder['len1'] ,axis=-1)
            len2=tf.expand_dims(self.placeholder['len2'],axis=-1)

            features = tf.concat([len1, len2],axis=1)

            LO_0 = self.all_pool(variable_scope="input-left", x=x1_expanded,
                                 reuse = reuse)
            RO_0 = self.all_pool(variable_scope="input-right", x=x2_expanded,
                                 reuse = reuse)

            LI_1, LO_1, RI_1, RO_1 = self.CNN_layer(variable_scope="CNN-1",
                                                    x1=x1_expanded,
                                                    x2=x2_expanded, 
                                                    d=self.d0,
                                                    reuse = reuse)
            sims = [self.cos_sim(LO_0, RO_0), self.cos_sim(LO_1, RO_1)]

            if self.num_layers > 1:
                _, LO_2, _, RO_2 = self.CNN_layer(variable_scope="CNN-2",
                                                  x1=LI_1, x2=RI_1, d=self.di,
                                                  reuse = reuse)
                sims.append(self.cos_sim(LO_2, RO_2))


            with tf.variable_scope("output-layer", reuse = reuse):
                output_features = tf.concat([features, tf.stack(sims, axis=1)], axis=1, name="output_features")
                out = tf.layers.dense(output_features,
                                     self.num_output,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                     name='fc')
                out = tf.squeeze(out, -1)
                #out = tf.sigmoid(out)
                #out = tf.contrib.layers.fully_connected(
                #    inputs=output_features,
                #    num_outputs=self.num_output,
                #    activation_fn=None,
                #    weights_initializer=tf.contrib.layers.xavier_initializer(),
                #    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                #    biases_initializer=tf.constant_initializer(1e-04),
                #    scope="FC"
                #)
                return out


