import tensorflow as tf
from common.layers import get_initializer

class DCNN():
    def __init__(self, **args):
        self.batch_size = args['batch_size']
        self.sentence_length = args['maxlen']
        self.embed_dim = args['embedding_size']
        self.keep_prob = args['keep_prob']
        self.num_output = args['num_output']
        self.num_filters = [6,14]
        self.top_k = 4
        self.k1 = 19
        self.num_hidden = 100
        self.ws = [7, 5]
        self.W1 = tf.get_variable(
                    "W1",
                    [self.ws[0], int(self.embed_dim), 1, self.num_filters[0]],
                    initializer=get_initializer(type = 'truncated_normal',
                                                stddev = 0.01))

        self.b1 = tf.get_variable(
                    "b1",
                    [self.num_filters[0], self.embed_dim],
                    initializer=get_initializer(type = 'constant', value=0.1))

        self.W2 = tf.get_variable(
                    "W2",
                    [self.ws[1], int(self.embed_dim/2), self.num_filters[0], self.num_filters[1]],
                    initializer=get_initializer(type = 'truncated_normal',
                                                stddev = 0.01))

        self.b2 = tf.get_variable(
                    "b2",
                    [self.num_filters[1], self.embed_dim],
                    initializer=get_initializer(type = 'constant', value=0.1))

    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        self.k1 = k
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.variable_scope("per_dim_conv_k_max_pooling"):
            for i in range(self.embed_dim):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                #conv:[batch_size, sent_length+ws-1, num_filters]
                conv = tf.reshape(conv, [self.batch_size, self.num_filters[0], self.sentence_length])#[batch_size, sentence_length, num_filters]
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.reshape(values, [self.batch_size, k, self.num_filters[0]])
                #k_max pooling in axis=1
                convs.append(values)
            conv = tf.stack(convs, axis=2)
        #[batch_size, k1, embed_dim, num_filters[0]]
        #print conv.get_shape()
        return conv

    def per_dim_conv_layer(self, x, w, b, reuse):
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.variable_scope("per_dim_conv", reuse = reuse):
            for i in range(len(input_unstack)):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_dim, num_filters[1]]
        return conv

    def fold_k_max_pooling(self, x, k, reuse):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.variable_scope("fold_k_max_pooling", reuse):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_dim/2, num_filters[1]]
        return fold

    def __call__(self, embed, scope_name = 'encoder', reuse = tf.AUTO_REUSE):
        with tf.variable_scope("dcnn"):
            sent = tf.expand_dims(embed, -1)
            conv1 = self.per_dim_conv_layer(sent, self.W1, self.b1, reuse)
            conv1 = self.fold_k_max_pooling(conv1, self.k1, reuse)
            conv2 = self.per_dim_conv_layer(conv1, self.W2, self.b2, reuse)
            fold = self.fold_k_max_pooling(conv2, self.top_k, reuse)
            fold_flatten = tf.reshape(fold, [-1, int(self.top_k*self.embed_dim*14/4)])
            dense = tf.layers.dense(fold_flatten, self.num_output, activation=None)
            return dense

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
