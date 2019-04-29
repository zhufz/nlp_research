import tensorflow as tf

class DPCNN():
    def __init__(self, **args):
        self.seq_length = args['maxlen']
        self.keep_prob = args['keep_prob']
        self.embedding_dim = args['embedding_size']
        self.num_output = args['num_output']
        self.num_filters = 250
        self.kernel_size = 3

    def __call__(self, embed, scope_name = 'encoder', reuse = tf.AUTO_REUSE):
        embedding_inputs = tf.expand_dims(embed, axis=-1)  # [None,seq,embedding,1]
        # region_embedding  # [batch,seq-3+1,1,250]
        region_embedding = tf.layers.conv2d(embedding_inputs, self.num_filters,
                                            [self.kernel_size, self.embedding_dim])

        pre_activation = tf.nn.relu(region_embedding, name='preactivation')

        with tf.variable_scope("conv3_0", reuse = reuse):
            conv3 = tf.layers.conv2d(pre_activation, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.variable_scope("conv3_1", reuse = reuse):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + region_embedding
        with tf.variable_scope("pool_1", reuse = reuse):
            pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
            pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        with tf.variable_scope("conv3_2", reuse = reuse):
            conv3 = tf.layers.conv2d(pool, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.variable_scope("conv3_3", reuse = reuse):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + pool
        pool_size = int((self.seq_length - 3 + 1)/2)
        conv3 = tf.layers.max_pooling1d(tf.squeeze(conv3, [2]), pool_size, 1)
        conv3 = tf.squeeze(conv3, [1]) # [batch,250]
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        dense = tf.layers.dense(conv3, self.num_output, activation=None)
        scores = tf.nn.softmax(dense, axis=1, name="scores")
        return scores

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
