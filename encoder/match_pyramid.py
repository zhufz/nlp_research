import sys
import tensorflow as tf
import numpy as np

class MatchPyramid():
    def __init__(self, **kwargs):
        self.maxlen1 = kwargs['maxlen1']
        self.maxlen2 = kwargs['maxlen2']
        self.psize1 = 3
        self.psize2 = 3
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',\
                                          shape=(None, self.maxlen1, self.maxlen2, 3))
    def __call__(self, x_query, x_sample, scope_name = 'encoder'):
        # batch_size * X1_maxlen * X2_maxlen
        self.cross = tf.einsum('abd,acd->abc', x_query, x_sample)
        self.cross_img = tf.expand_dims(self.cross, 3)
        # convolution
        self.w1 = tf.get_variable('w1',
            initializer=tf.truncated_normal_initializer(mean=0.0,
                stddev=0.2, dtype=tf.float32),
            dtype=tf.float32, shape=[2, 10, 1, 8])
        self.b1 = tf.get_variable('b1',
            initializer=tf.constant_initializer(),
            dtype=tf.float32, shape=[8])
        # batch_size * X1_maxlen * X2_maxlen * feat_out
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img,
            self.w1, [1, 1, 1, 1], "SAME") + self.b1)

        # dynamic pooling
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
        stride1 = self.maxlen1 / self.psize1
        stride2 = self.maxlen2 / self.psize2
        suggestion1 = self.maxlen1 / stride1
        suggestion2 = self.maxlen2 / stride2

        if suggestion1 != self.psize1 or suggestion2 != self.psize2:
            print("DynamicMaxPooling Layer can not "
                  "generate ({} x {}) output feature map,"
                  "please use ({} x {} instead.)"
                  .format(self.psize1, self.psize2, 
                          suggestion1, suggestion2))
            exit()

        self.pool1 = tf.nn.max_pool(self.conv1_expand, 
                        [1, stride1, stride2, 1], 
                        [1, stride1, stride2, 1], "VALID")

        with tf.variable_scope('fc1'):
            #self.fc1 = tf.nn.relu(tf.contrib.layers.linear(tf.reshape(self.pool1, 
            #                                               [-1, self.psize1 * self.psize2 * 8]
            #                                               ), 20))
            self.fc1 = tf.nn.relu(tf.layers.dense(tf.reshape(self.pool1, 
                                                           [-1, self.psize1 * self.psize2 * 8]
                                                           ), 20))
        self.pred = tf.layers.dense(self.fc1, 1, name = 'scores')
        out = tf.squeeze(self.pred, -1)
        #self.pred = tf.contrib.layers.linear(self.fc1, 1)
        return out

    def feed_dict(self, **kwargs):
        feed_dict = {}
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(kwargs['x_query'], kwargs['x_sample'])
        return feed_dict

    def pb_feed_dict(self,graph,  name = 'dpool_index', **kwargs):
        feed_dict = {}
        pool_index = graph.get_operation_by_name(name).outputs[0]
        feed_dict[pool_index] = self.dynamic_pooling_index(kwargs['x_query'], kwargs['x_sample'])
        return feed_dict

    def dynamic_pooling_index(self, len1, len2, compress_ratio1 = 1, compress_ratio2 = 1):
        def dpool_index_(batch_idx, len1_one, len2_one, cur_maxlen1, cur_maxlen2):
            '''
            TODO: Here is the check of sentences length to be positive.
            To make sure that the lenght of the input sentences are positive.
            if len1_one == 0:
                print("[Error:DynamicPooling] len1 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            if len2_one == 0:
                print("[Error:DynamicPooling] len2 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            '''
            if len1_one == 0:
                stride1 = cur_maxlen1
            else:
                stride1 = 1.0 * cur_maxlen1 / len1_one

            if len2_one == 0:
                stride2 = cur_maxlen2
            else:
                stride2 = 1.0 * cur_maxlen2 / len2_one

            idx1_one = [int(i / stride1) for i in range(cur_maxlen1)]
            idx2_one = [int(i / stride2) for i in range(cur_maxlen2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx,
                                      mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        dpool_bias1 = dpool_bias2 = 0
        if self.maxlen1 % compress_ratio1 != 0:
            dpool_bias1 = 1
        if self.maxlen2 % compress_ratio2 != 0:
            dpool_bias2 = 1
        cur_maxlen1 = self.maxlen1 // compress_ratio1 + dpool_bias1
        cur_maxlen2 = self.maxlen2 // compress_ratio2 + dpool_bias2
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i] // compress_ratio1, 
                         len2[i] // compress_ratio2, cur_maxlen1, cur_maxlen2))
        return np.array(index)


