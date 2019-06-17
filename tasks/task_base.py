#-*- coding: utf-8 -*-

class TaskBase(object):
    def __init__(self, conf):
        for attr in conf:
            setattr(self, attr, conf[attr])

    def read_data(self):
        raise NotImplementedError('subclasses must override read_data()!')
