#-*- coding:utf-8 -*-
#所有encoder的基类
import copy

class EncoderBase(object):
    def __init__(self, **kwargs):
        if 'num_output' in kwargs:
            self.num_output = kwargs['num_output']
        if 'keep_prob' in kwargs:
            self.keep_prob = kwargs['keep_prob']
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'maxlen' in kwargs:
            self.maxlen = kwargs['maxlen']
        self.features = {} #placeholder dict for export model

    def encoder_fun(self, **kwargs):
        return {}

    def keys_to_features(self, **kwargs):
        return {}

    def parsed_to_features(self, **kwargs):
        return {}

    def get_features(self, **kwargs):
        features = {}
        return features
