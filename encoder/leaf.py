import tensorflow as tf
from encoder import Base
from common.layers import get_initializer
import pdb
import copy

#matching based fasttext
class LEAF(Base):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        super(LEAF, self).__init__(**kwargs)
        self.maxlen = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']
        self.placeholder = {}
        self.num_hidden = 300

    def __call__(self, embed, name = 'encoder', features = None, reuse = tf.AUTO_REUSE, **kwargs):
        with tf.variable_scope("fast_text", reuse = reuse):
            #embed: [batch_size, self.maxlen, embedding_size]
            length = tf.placeholder(tf.int32, name=name + '_length',shape=[])
            if features != None:
                length = features[name + '_length']
            #pdb.set_trace()
            #label_embedding: [num_output, num_hidden]
            label_embedding = tf.get_variable(
                "label_embedding", 
                [self.num_output, self.num_hidden],
                initializer=get_initializer(type = 'random_uniform',
                                            minval = -0.01, 
                                            maxval = 0.01)
            )

            #mask:[batch_size, self.maxlen]
            mask = tf.sequence_mask(length, self.maxlen, tf.float32)
            mask = tf.expand_dims(mask, -1)
            embed = embed*mask
            mean_sentence = tf.reduce_mean(embed, axis=1)
            #vec: [batch_size, num_hidden]
            vec = tf.layers.dense(mean_sentence,
                                    self.num_hidden,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                    name='fc',
                                    reuse = reuse)
            #att: [batch_size, num_output]
            att = tf.matmul(vec, tf.transpose(label_embedding))
            return att

    def feed_dict(self, name = 'encoder', **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = name + "_length" 
            feed_dict[self.placeholder[length_name]] = kwargs[key]
        return feed_dict

    def pb_feed_dict(self, graph, name = 'encoder', **kwargs):
        feed_dict = {}
        for key in kwargs:
            length_name = name + "_length" 
            key_node = graph.get_operation_by_name(length_name).outputs[0]
            feed_dict[key_node] = kwargs[key]
        return feed_dict

    def get_features(self, name = 'encoder'):
        features = {}
        length_name = name+'_length'
        features[length_name] = tf.placeholder(tf.int32, name=name + '_length',shape=[])
        return features
