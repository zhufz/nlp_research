import tensorflow as tf
import pdb


class VDCNN(object):
    def __init__(self, **args):
        #TODO: fix problem
        self.embedding_size = args['embedding_size']
        self.is_training = args['is_training']
        self.keep_prob = args['keep_prob']
        self.num_output = args['num_output']
        self.filter_sizes = [3, 3, 3, 3, 3]
        self.num_filters = [64, 64, 128, 256, 512]
        self.num_blocks = [2, 2, 2, 2]
        self.learning_rate = 1e-3
        self.cnn_initializer = tf.keras.initializers.he_normal()
        self.fc_initializer = tf.truncated_normal_initializer(stddev=0.05)


    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE):
        # ============= Embedding Layer =============
        self.x_expanded = tf.expand_dims(embed, -1)
        # ============= First Convolution Layer =============
        with tf.variable_scope("conv-0", reuse = reuse):
            conv0 = tf.layers.conv2d(
                self.x_expanded,
                filters=self.num_filters[0],
                kernel_size=[self.filter_sizes[0], self.embedding_size],
                kernel_initializer=self.cnn_initializer,
                activation=tf.nn.relu)
            conv0 = tf.transpose(conv0, [0, 1, 3, 2])

        # ============= Convolution Blocks =============
        with tf.variable_scope("conv-block-1", reuse = reuse):
            conv1 = self.conv_block(conv0, 1, reuse = reuse)

        with tf.variable_scope("conv-block-2", reuse = reuse):
            conv2 = self.conv_block(conv1, 2, reuse = reuse)

        with tf.variable_scope("conv-block-3", reuse = reuse):
            conv3 = self.conv_block(conv2, 3, reuse = reuse)

        with tf.variable_scope("conv-block-4", reuse = reuse):
            conv4 = self.conv_block(conv3, 4, max_pool=False, reuse = reuse)

        # ============= k-max Pooling =============
        with tf.variable_scope("k-max-pooling", reuse = reuse):
            h = tf.transpose(tf.squeeze(conv3, -1), [0, 2, 1])
            top_k = tf.nn.top_k(h, k=8, sorted=False).values
            h_flat = tf.reshape(top_k, [-1, 512 * 8])

        # ============= Fully Connected Layers =============
        with tf.variable_scope("fc-1", reuse = reuse):
            fc1_out = tf.layers.dense(h_flat, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        with tf.variable_scope("fc-2", reuse = reuse):
            fc2_out = tf.layers.dense(fc1_out, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        h_drop = tf.nn.dropout(fc2_out, self.keep_prob)
        dense = tf.layers.dense(h_drop, self.num_output, activation=None)
        return dense


    def conv_block(self, input, i, reuse, max_pool=True):

        with tf.variable_scope("conv-block-%s" % i, reuse = reuse):
            # Two "conv-batch_norm-relu" layers.
            for j in range(2):
                with tf.variable_scope("conv-%s" % j, reuse = reuse):
                    # convolution
                    conv = tf.layers.conv2d(
                        input,
                        filters=self.num_filters[i],
                        kernel_size=[self.filter_sizes[i], self.num_filters[i-1]],
                        kernel_initializer=self.cnn_initializer,
                        activation=None,
                        padding='SAME')
                    # batch normalization
                    conv = tf.layers.batch_normalization(conv, training=self.is_training)
                    # relu
                    conv = tf.nn.relu(conv)
                    conv = tf.transpose(conv, [0, 1, 3, 2])

            if max_pool:
                # Max pooling
                pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=(3, 1),
                    strides=(2, 1),
                    padding="SAME")
                return pool
            else:
                return conv

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
