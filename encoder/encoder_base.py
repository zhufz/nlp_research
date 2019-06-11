#-*- coding:utf-8 -*-
#所有encoder的基类
import copy

class EncoderBase(object):
    def __init__(self, **kwargs):
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
