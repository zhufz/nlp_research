#所有encoder的基类
class Base():
    def __init__():
        pass

    def encoder_fun(self, **kwargs):
        #用于计算encoder内部所需要的id值，如长度特征
        return {}

    def keys_to_features(self, **kwargs):
        return {}

    def parsed_to_features(self, **kwargs):
        return {}
