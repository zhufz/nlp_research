#所有encoder的基类
import copy

class Base():
    def __init__(self, **kwargs):
        self.features = {} #placeholder dict for export model

    def encoder_fun(self, **kwargs):
        return {}

    def keys_to_features(self, **kwargs):
        return {}

    def parsed_to_features(self, **kwargs):
        return {}
