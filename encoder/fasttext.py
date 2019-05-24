import tensorflow as tf
from encoder import Base
import pdb

class FastText(Base):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        self.maxlen = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']
        self.placeholder = {}

    def __call__(self, embed, name = 'encoder', features = None, reuse = tf.AUTO_REUSE, **kwargs):
        with tf.variable_scope("fast_text", reuse = reuse):
            #embed: [batch_size, self.maxlen, embedding_size]
            if features == None:
                length = tf.placeholder(tf.int32, name='x_query_length',shape=[])
            else:
                length = features['x_query_length']
            #pdb.set_trace()
            #mask:[batch_size, self.maxlen]
            mask = tf.sequence_mask(length, self.maxlen, tf.float32)
            mask = tf.expand_dims(mask, -1)
            embed = embed*mask
            mean_sentence = tf.reduce_mean(embed, axis=1)
            logits = tf.layers.dense(mean_sentence,
                                    self.num_output,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                    name='fc',
                                    reuse = reuse)
            return logits

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
