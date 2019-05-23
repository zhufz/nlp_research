import tensorflow as tf
from encoder import Base

class FastText(Base):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        self.seq_length = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']

    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE, **kwargs):
        with tf.variable_scope("fast_text", reuse = reuse):
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
