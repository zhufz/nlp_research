#所有encoder的基类
import copy

class Base():
    def __init__(self, **kwargs):
        pass

    def embed_fun(self, text_id, name = 'base_embedding', **kwargs):
        input_dict = {}
        input_dict[name] = text_id
        return input_dict 
