#-*- coding:utf-8 -*-
import tensorflow as tf

from utils.tf_utils import  get_placeholder_batch_size
import numpy as np
import collections
import pdb
class RNNLayer:
    def __init__(self, rnn_type, num_hidden, num_layers, cell_type='lstm',
                 use_attention = False):
        self.rnn_type = rnn_type
        self.cell_type = cell_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.key_node = None
        self.use_attention = use_attention

    def __call__(self, inputs, seq_len, initial_state = None, 
                 name = 'rnn', rnn_keep_prob = 0.8, maxlen = None, reuse = tf.AUTO_REUSE):
        with tf.variable_scope("rnn_"+name, reuse = reuse):
            assert self.num_layers >0, "num_layers need larger than 0"
            assert self.num_hidden >0, "num_hidden need larger than 0"

            def cell(embedded, hidden_size, cell_type = 'lstm', reuse = False):
                if cell_type == 'gru':
                    _cell = tf.contrib.rnn.GRUCell(hidden_size,
                                                    reuse = reuse)
                    return tf.contrib.rnn.DropoutWrapper(_cell, 
                                                     output_keep_prob = rnn_keep_prob)
                elif cell_type == 'lstm':
                    _cell = tf.nn.rnn_cell.LSTMCell(hidden_size, 
                                                    initializer = tf.orthogonal_initializer(), 
                                                    reuse = reuse)
                    return tf.contrib.rnn.DropoutWrapper(_cell, 
                                                     output_keep_prob = rnn_keep_prob)
                else:
                    raise ValueError('unknown cell_type %s'%cell_type)

            batch_size = get_placeholder_batch_size(inputs)

            if self.rnn_type == 'si': #single direction
                cells = [cell(inputs, self.num_hidden, self.cell_type) for n in range(self.num_layers)]
                stack = tf.contrib.rnn.MultiRNNCell(cells)
                if initial_state == None:
                    self.initial_state = stack.zero_state(batch_size, dtype=tf.float32)
                else:
                    self.initial_state = initial_state
                outputs, state = tf.nn.dynamic_rnn(stack, 
                                                   inputs, 
                                                   seq_len, 
                                                   initial_state = self.initial_state,
                                                   dtype=tf.float32)
                self.initial_state = tf.identity(self.initial_state, name='initial_state')
                #state_for_feed = tf.identity(state, name="state")
            elif self.rnn_type == 'bi': #bidirection
                #multi layer lstm
                fw_cells = [cell(inputs, self.num_hidden, self.cell_type) for n in range(self.num_layers)]
                bw_cells = [cell(inputs, self.num_hidden, self.cell_type) for n in range(self.num_layers)]
                stack_fw = tf.contrib.rnn.MultiRNNCell(fw_cells)
                stack_bw = tf.contrib.rnn.MultiRNNCell(bw_cells)
                if initial_state == None:
                    self.initial_state_fw = stack_fw.zero_state(batch_size, dtype=tf.float32)
                    self.initial_state_bw = stack_bw.zero_state(batch_size, dtype=tf.float32)
                else:
                    assert len(initial_state) == 2, "initial_state length shoud be 2"
                    self.initial_state_fw = initial_state[0]
                    self.initial_state_bw = initial_state[1] 
                (fw_outputs,bw_outputs), state = \
                    tf.nn.bidirectional_dynamic_rnn(stack_fw,
                                                    stack_bw,
                                                    inputs, 
                                                    seq_len, 
                                                    initial_state_fw = self.initial_state_fw, 
                                                    initial_state_bw = self.initial_state_bw, 
                                                    dtype=tf.float32)
                #fw_state: #[batch_size, num_layers, num_hidden]
                #bw_state: #[batch_size, num_layers, num_hidden]
                self.initial_state_fw = tf.identity(self.initial_state_fw, name='initial_state_fw')
                self.initial_state_bw = tf.identity(self.initial_state_bw, name='initial_state_bw')
                outputs = tf.concat((fw_outputs, bw_outputs), 2)
                #state = tf.concat((fw_state, bw_state), 1)
                #state_for_feed = tf.identity(state, name="state")
            else:
                raise ValueError("unknown rnn type")
            #if self.use_attention:
            #    assert maxlen != None, "attention need maxlen parameter"
            #    outputs = self.attention_output(outputs, seq_len, maxlen)
            return outputs,state

def get_initializer(type = 'random_uniform', **kwargs):
    '''
    params:
    constant: value
    zeros:
    ones:
    random_normal: mean stddev
    truncated_normal: mean stddev
    random_uniform: minval, maxval
    xavier:
    variance_scaling:
    不同初始化区别参考：https://mp.weixin.qq.com/s/9fQdp4G3dFOvBbDnCPlbYw
    '''
    #default value
    value = kwargs['value'] if 'value' in kwargs else 0.0
    minval = kwargs['minval'] if 'minval' in kwargs else -1
    maxval = kwargs['maxval'] if 'maxval' in kwargs else 1
    mean = kwargs['mean'] if 'mean' in kwargs else 0.0
    stddev = kwargs['stddev'] if 'stddev' in kwargs else 1.0


    if type == 'constant':
        return tf.constant_initializer(value = value, dtype = tf.float32)
    elif type == 'zeros':
        return tf.zeros_initializer()
    elif type == 'ones':
        return tf.ones_initializer()
    elif type == 'random_normal':
        return tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
    elif type == 'truncated_normal':
        return tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
    elif type == 'random_uniform':
        return tf.random_uniform_initializer(minval = minval,
                                             maxval = maxval, 
                                             seed=None, 
                                             dtype=tf.float32)
    elif type == 'xavier' or type == 'xavior_uniform':
        return tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    elif type == 'xavier_normal':
        return tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
    elif type =='variance_scaling':
        return tf.variance_scaling_initializer(scale=1.0,mode="fan_in",
                                                        distribution="uniform",seed=None,dtype=tf.float32)
    else:
        raise ValueError('unknown type of initializer!')

def get_train_op(global_step, optimizer_type, loss, learning_rate, var_list = None, clip_grad = 5):
    with tf.variable_scope("train_step"):
        if optimizer_type == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'Adadelta':
            optim = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'Adagrad':
            optim = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'RMSProp':
            optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'Momentum':
            optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        if var_list != None:
            grads_and_vars = optim.compute_gradients(loss, var_list)
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] if g!= None else [g,v] for g, v in grads_and_vars]
        else:
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
        return train_op

def conv(inputs, output_units, bias=True, activation=None, dropout=None,
                                 scope='conv-layer', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=get_initializer(type = 'variance_scaling'),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=get_initializer(type = 'zeros'),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
    return z


