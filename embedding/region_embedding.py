
import gensim
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
import numpy as np
from itertools import chain
import tensorflow as tf
from utils.preprocess import *
from embedding.embedding_base import Base
from common.layers import get_initializer
import pandas as pd
import collections
import pickle
import pdb

#refer:http://t.cn/ES8SBNO
class Layer(object):
    """Layer"""
    def __init__(self, name, activation=None, dropout=None, decay_mult=None):
        self._name = name
        self._activation = activation
        self._dropout = dropout
        self._decay_mult = decay_mult

    def get_variable(self, name, **kwargs):
        """get variable with regularization'
        """
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult
        return tf.get_variable(name, **kwargs)

    def __call__(self, *inputs):
        """forward
        Args:
                inputs(type): input op
        Returns:
                type: output op
        """
        
        outputs = []
        for x in inputs:
            if type(x) == tuple or type(x) == list:
                y = self._forward(*x)
            else:
                y = self._forward(x)
            if self._activation:
                y = self._activation(y)
            if self._dropout:
                if hasattr(tf.flags.FLAGS, 'training'):
                    y = tf.cond(tf.flags.FLAGS.training, 
                            lambda: tf.nn.dropout(y, keep_prob = 1.0 - self._dropout), 
                            lambda: y)
                else:
                    y = tf.nn.dropout(y, keep_prob = 1.0 - self._dropout)
            outputs.append(y)
        
        def get_shape_desc(x):
            """get shape description
            """
            if type(x) == list or type(x) == tuple:
                return '[%s]' % ', '.join([str(xi.shape) for xi in x])
            return str(x.shape)
#        print >> sys.stderr, '''layer {
#    type: %s
#    name: %s
#    shape[in]: %s
#    shape[out]: %s
#}''' % (self.__class__.__name__, self._name, get_shape_desc(x), get_shape_desc(y))
#        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def _forward(self, x):
        return x

class RegionAlignmentLayer(Layer):
    """RegionAlignmentLayer"""
    def __init__(self, region_size, name="RegionAlig", **args):
        Layer.__init__(self, name, **args) 
        self._region_size = region_size

    def _forward(self, x):
        """forward
            region_size: region size
        """
        region_radius = int(self._region_size / 2)
        aligned_seq = []
        for i in range(region_radius, x.shape[1] - region_radius):
            aligned_seq.append(tf.slice(x, [0, i - region_radius], 
                                        [-1, self._region_size]))
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2])
        return aligned_seq

class EmbeddingLayer():
    """EmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, name="embedding", 
            initializer=None, **kwargs):
        self._decay_mult = None
        self._emb_size = emb_size
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
                initializer=initializer)
    def get_variable(self, name, **kwargs):
        """get variable with regularization'
        """
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult
        return tf.get_variable(name, **kwargs)

    def _forward(self, seq):
        return tf.nn.embedding_lookup(self._W, seq)


class WindowPoolEmbeddingLayer(EmbeddingLayer):
    """WindowPoolEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="win_pool_embedding", \
            initializer=None, \
            **kwargs):
        Layer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, emb_size],
                initializer=initializer)
        super(WindowPoolEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WindowPoolEmbeddingLayer, self)._forward(region_aligned_seq)

        return self._region_merge_fn(region_aligned_emb, axis=2)


class ScalarRegionEmbeddingLayer(EmbeddingLayer):
    """WordContextRegionEmbeddingLayer(Scalar)"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="scalar_region_embedding", \
            initializer=None, \
            **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, 1],
                initializer=initializer)
        super(ScalarRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(ScalarRegionEmbeddingLayer, self)._forward(region_aligned_seq)

        region_radius = int(self._region_size / 2)
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)


class MultiRegionEmbeddingLayer(EmbeddingLayer):
    """"WordContextRegionEmbeddingLayer(Multi-region)"""
    def __init__(self, vocab_size, emb_size, region_sizes, \
            region_merge_fn=None, \
            name="multi_region_embedding", \
            initializer=None, \
            **kwargs):
        
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_sizes = region_sizes[:]
        self._region_sizes.sort()
        self._region_merge_fn = region_merge_fn
        region_num = len(region_sizes)

        self._K = [None] * region_num
        self._K[-1] = tf.get_variable(name + '_K_%d' % (region_num - 1), \
                    shape=[vocab_size, self._region_sizes[-1], emb_size], \
                    initializer=initializer)

        for i in range(region_num - 1):
            st = self._region_sizes[-1]/2 - self._region_sizes[i]/2
            ed = st + self._region_sizes[i]
            self._K[i] = self._K[-1][:, st:ed, :]

        super(MultiRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        """_forward
        """

        multi_region_emb = [] 

        for i, region_kernel in enumerate(self._K):
            region_radius = int(self._region_sizes[i] / 2)
            region_aligned_seq = RegionAlignmentLayer(self._region_sizes[i], name="RegionAlig_%d" % (i))(seq)
            region_aligned_emb = super(MultiRegionEmbeddingLayer, self)._forward(region_aligned_seq)
             
            trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
            context_unit = tf.nn.embedding_lookup(region_kernel, trimed_seq)

            projected_emb = region_aligned_emb * context_unit
            region_emb =  self._region_merge_fn(projected_emb, axis=2)
            multi_region_emb.append(region_emb)
        
        return multi_region_emb


class WordContextRegionEmbeddingLayer(EmbeddingLayer):
    """WordContextRegionEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="word_context_region_embedding", \
            initializer=None, \
            **kwargs):
        Layer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, emb_size],
                initializer=initializer)
        super(WordContextRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WordContextRegionEmbeddingLayer, self)._forward(region_aligned_seq)

        region_radius = int(self._region_size / 2)
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)


class ContextWordRegionEmbeddingLayer(EmbeddingLayer):
    """ContextWordRegionEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, 
            region_merge_fn=None,
            name="embedding",
            initializer=None, **kwargs):
        super(ContextWordRegionEmbeddingLayer, self).__init__(vocab_size * region_size, emb_size, name,
                initializer, **kwargs)
        self._region_merge_fn = region_merge_fn
        self._word_emb = tf.get_variable(name + '_wordmeb', shape=[vocab_size, emb_size], 
                initializer=initializer)
        self._unit_id_bias = np.array([i * vocab_size for i in range(region_size)])
        self._region_size = region_size

    def _region_aligned_units(self, seq):
        """
        _region_aligned_unit
        """
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_seq = region_aligned_seq + self._unit_id_bias
        region_aligned_unit = super(ContextWordRegionEmbeddingLayer, self)._forward(region_aligned_seq)
        return region_aligned_unit
    
    def _forward(self, seq):
        """forward
        """
        region_radius = int(self._region_size / 2)
        word_emb = tf.nn.embedding_lookup(self._word_emb, \
                tf.slice(seq, \
                [0, region_radius], \
                [-1, tf.cast(seq.get_shape()[1] - 2 * region_radius, tf.int32)]))
        word_emb = tf.expand_dims(word_emb, 2)
        region_aligned_unit = self._region_aligned_units(seq)
        embedding = region_aligned_unit * word_emb
        embedding = self._region_merge_fn(embedding, axis=2)
        return embedding

class RegionEmbedding(Base):
    def __init__(self, text_list, dict_path, vocab_dict = None, random = False,
                 maxlen = 40, embedding_size = 128, region_size = 3, **kwargs):
        super(RegionEmbedding, self).__init__(**kwargs)
        self.embedding_path = None
        self.dict_path = dict_path
        self.maxlen = maxlen
        self.size = embedding_size
        self.vocab_dict = vocab_dict
        self.vocab_size = len(self.vocab_dict)
        self.embedding_size = embedding_size
        self.region_size = region_size
        params = {"vocab_size": self.vocab_size, 
                  "emb_size" : self.embedding_size,
                  "region_size" : self.region_size}

        fn_dict = {
                    'reduce_max': tf.reduce_max, \
                    'reduce_sum': tf.reduce_sum, \
                    'concat': tf.reshape}

        fn_name = kwargs.get("fn_dict", "reduce_max")
        params['region_merge_fn'] = fn_dict.get(fn_name, tf.reduce_max)

        if "region_type" in kwargs:
            region_type = kwargs['region_type']
        else:
            region_type = "context_word_region"  #default value

        if region_type == "context_word_region":
            self.embedding = ContextWordRegionEmbeddingLayer(**params)
        elif region_type == "word_context_region":
            self.embedding = WordContextRegionEmbeddingLayer(**params)
        elif region_type == "multi_region":
            self.embedding = MultiRegionEmbeddingLayer(**params)
        elif region_type == "scale_region":
            self.embedding = ScalarRegionEmbeddingLayer(**params)
        elif region_type == "window_pool":
            self.embedding = WindowPoolEmbeddingLayer(**params)
        else:
            raise ValueError("unknown region type!")
        self.input_ids = {}

    @staticmethod
    def build_dict(dict_path, text_list = None, mode = "train"):
        if not os.path.exists(dict_path) or mode == "train":
            assert text_list != None, "text_list can't be None in train mode"
            chars = list()
            for content in text_list:
                for char in char_tokenize(clean_str(content)):
                    chars.append(char)

            char_counter = collections.Counter(chars).most_common()
            vocab_dict = dict()
            vocab_dict["<pad>"] = 0
            vocab_dict["<unk>"] = 1
            for char, _ in char_counter:
                vocab_dict[char] = len(vocab_dict)

            with open(dict_path, "wb") as f:
                pickle.dump(vocab_dict, f)
        else:
            with open(dict_path, "rb") as f:
                vocab_dict = pickle.load(f)

        return vocab_dict

    @staticmethod
    def text2id(text_list, vocab_dict, maxlen, need_preprocess = True):
        """
        文本id化
        """
        if need_preprocess:
            pre = Preprocess()
            text_list = [pre.get_dl_input_by_text(text) for text in text_list]
        x = list(map(lambda d: char_tokenize(clean_str(d)), text_list))
        x_len = [min(len(text), maxlen) for text in x]
        x = list(map(lambda d: list(map(lambda w: vocab_dict.get(w,vocab_dict["<unk>"]), d)), x))
        x = list(map(lambda d: d[:maxlen], x))
        x = list(map(lambda d: d + (maxlen - len(d)) * [vocab_dict["<pad>"]], x))
        return text_list, x, x_len

    def __call__(self, features = None, name = "region_embedding"):
        """define placeholder"""
        if features == None:
            self.input_ids[name] = tf.placeholder(dtype=tf.int32, shape=[None,
                                                                     self.maxlen], name = name)
        else:
            self.input_ids[name] = features[name]
        return self.embedding._forward(self.input_ids[name])

    def feed_dict(self, input_x, name = 'region_embedding'):
        feed_dict = {}
        feed_dict[self.input_ids[name]] = input_x
        return feed_dict

    def pb_feed_dict(self, graph, input_x, name = 'region_embedding'):
        feed_dict = {}
        input_x_node = graph.get_operation_by_name(name).outputs[0]
        feed_dict[input_x_node] = input_x
        return feed_dict

