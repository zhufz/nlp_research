import tensorflow as tf
from common.layers import get_initializer
from encoder import Base

class FastAttentionText(Base):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        super(FastAttentionText, self).__init__(**kwargs)
        self.seq_length = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.hidden_num = 512
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']

    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE):
        with tf.variable_scope("fast_text", reuse = reuse):
            att = tf.get_variable(
                "att", [self.seq_length, self.embedding_dim],
                initializer=get_initializer(minval = -0.01, maxval = 0.01)
            )
            mul =tf.nn.softmax(tf.multiply(embed, att, name= 'attention_score'))
            embed = tf.multiply(embed, mul)
            mean_sentence = tf.reduce_mean(embed, axis=1)
            logits = tf.layers.dense(mean_sentence,
                                    self.hidden_num,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                    name='fc')
            h_drop = tf.nn.dropout(logits, self.keep_prob)
            dense = tf.layers.dense(h_drop, self.num_output, activation=None)
            return dense

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict
