#-*- coding: utf-8 -*-
import tensorflow as tf
from functools import partial
from common.lr import cyclic_learning_rate
from common.layers import get_train_op

class TaskBase(object):
    def __init__(self, conf):
        for attr in conf:
            setattr(self, attr, conf[attr])

    def read_data(self):
        raise NotImplementedError('subclasses must override read_data()!')

    def train_estimator_spec(self, mode, loss, global_step, params):
        if self.use_clr:
            self.learning_rate = cyclic_learning_rate(global_step=global_step,
                                                  learning_rate = self.learning_rate, 
                                                  mode = self.clr_mode)
        optim_func = partial(get_train_op,
                             global_step, 
                             self.optimizer_type, 
                             loss,
                             clip_grad =5)

        if 'base_var' in params:
            #if contains base model variable list
            tvars = tf.trainable_variables()
            new_var_list = []
            base_var_list = []
            for var in tvars:
                name = var.name
                m = re.match("^(.*):\\d+$", name)
                if m is not None: 
                    name = m.group(1)
                if name in params['base_var']: 
                    base_var_list.append(var)
                    continue
                new_var_list.append(var)
            optimizer_base = optim_func(learning_rate = self.base_learning_rate,
                                        var_list = base_var_list)
            optimizer_now = optim_func(learning_rate = self.learning_rate,
                                       var_list = new_var_list)
            if self.learning_rate == 0:
                raise ValueError('learning_rate can not be zero')
            if self.base_learning_rate == 0:
                # if base_learning_rate is set to be zero, than only
                # the downstream net parameters will be trained
                optimizer = optimizer_now
            else:
                optimizer = tf.group(optimizer_base, optimizer_now)
        else:
            optimizer = optim_func(learning_rate = self.learning_rate)
        return tf.estimator.EstimatorSpec(mode, loss = loss,
                                              train_op=optimizer)
