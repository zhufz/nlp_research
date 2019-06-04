import tensorflow as tf

from utils.tf_utils import  get_placeholder_batch_size
import numpy as np
import collections
import pdb
class RNNLayer:
    def __init__(self, rnn_type, num_hidden, num_layers):
        self.rnn_type = rnn_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.key_node = None

    def feed_dict(self, initial_state, graph = None):
        #used to feed initial state
        feed_dict = {}

        if self.rnn_type == 'lstm':
            if graph != None:
                self.initial_state = graph.get_tensor_by_name(self.pb_nodes[0]+":0")
            feed_dict[self.initial_state] = initial_state
        elif self.rnn_type == 'gru':
            if graph != None:
                self.initial_state = graph.get_tensor_by_name(self.pb_nodes[0]+":0")
            feed_dict[self.initial_state] = initial_state
        elif self.rnn_type == 'bi_lstm' and self.num_layers == 1:
            if graph != None:
                self.initial_state_fw = graph.get_tensor_by_name(self.pb_nodes[0]+":0")
                self.initial_state_bw = graph.get_tensor_by_name(self.pb_nodes[1]+":0")
            feed_dict[self.initial_state_fw] = initial_state[0]
            feed_dict[self.initial_state_bw] = initial_state[1]
        elif self.rnn_type == 'bi_lstm':
            if graph != None:
                self.initial_state_fw = graph.get_tensor_by_name(self.pb_nodes[0]+":0")
                self.initial_state_bw = graph.get_tensor_by_name(self.pb_nodes[1]+":0")
            feed_dict[self.initial_state_fw] = initial_state[0]
            feed_dict[self.initial_state_bw] = initial_state[1]
        elif self.rnn_type == 'bi_gru':
            if graph != None:
                self.initial_state_fw = graph.get_tensor_by_name(self.pb_nodes[0]+":0")
                self.initial_state_bw = graph.get_tensor_by_name(self.pb_nodes[1]+":0")
            feed_dict[self.initial_state_fw] = initial_state[0]
            feed_dict[self.initial_state_bw] = initial_state[1]
        return feed_dict

    def __call__(self, inputs, seq_len, initial_state = None, name = 'rnn',
                 reuse = tf.AUTO_REUSE):
        with tf.variable_scope("rnn_"+name, reuse = reuse):
            assert self.num_layers >0, "num_layers need larger than 0"
            assert self.num_hidden >0, "num_hidden need larger than 0"
            batch_size = get_placeholder_batch_size(inputs)
            if self.rnn_type == 'lstm':
                cells = [tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True) for n in
                         range(self.num_layers)]
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
                state_for_feed = tf.identity(state, name="state")
                self.pb_nodes = [self.initial_state.name.split(':')[0],
                                 state_for_feed.name.split(':')[0]]

            elif self.rnn_type == 'gru':
                cells = [tf.contrib.rnn.GRUCell(self.num_hidden) for n in range(self.num_layers)]
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
                state_for_feed = tf.identity(state, name="state")
                self.pb_nodes = [self.initial_state.name.split(':')[0],
                                 state_for_feed.name.split(':')[0]]

            elif self.rnn_type == 'bi_lstm' and self.num_layers == 1:
                #sigle layer lstm
                cell_fw = tf.contrib.rnn.LSTMCell(self.num_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.num_hidden)
                if initial_state == None:
                    self.initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
                    self.initial_state_bw = cell_fw.zero_state(batch_size, dtype=tf.float32)
                else:
                    assert len(initial_state) == 2, "initial_state length shoud be 2"
                    self.initial_state_fw = initial_state[0] 
                    self.initial_state_bw = initial_state[1]
                (fw_outputs,bw_outputs), state = \
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                    cell_bw=cell_bw, 
                                                    inputs=inputs, 
                                                    sequence_length=seq_len, 
                                                    initial_state_fw = self.initial_state_fw, 
                                                    initial_state_bw = self.initial_state_bw, 
                                                    dtype=tf.float32)

                self.initial_state_fw = tf.identity(self.initial_state_fw, name='initial_state_fw')
                self.initial_state_bw = tf.identity(self.initial_state_bw, name='initial_state_bw')
                state_for_feed = tf.identity(state, name="state")
                outputs = tf.concat((fw_outputs, bw_outputs), 2)
                self.pb_nodes = [self.initial_state_fw.name.split(':')[0],
                                 self.initial_state_bw.name.split(':')[0],
                                 state_for_feed.name.split(':')[0]]

            elif self.rnn_type == 'bi_lstm':
                #multi layer lstm
                fw_cells = cells = [tf.contrib.rnn.LSTMCell(self.num_hidden,
                                                 state_is_tuple=True) for n in
                                    range(self.num_layers)]
                bw_cells = cells = [tf.contrib.rnn.LSTMCell(self.num_hidden,
                                                 state_is_tuple=True) for n in
                                    range(self.num_layers)]
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
                state_for_feed = tf.identity(state, name="state")
                self.pb_nodes = [self.initial_state_fw.name.split(':')[0],
                                 self.initial_state_bw.name.split(':')[0],
                                 state_for_feed.name.split(':')[0]]

            elif self.rnn_type == 'bi_gru':
                fw_cells = [tf.contrib.rnn.GRUCell(self.num_hidden) for n in range(self.num_layers)]
                bw_cells = [tf.contrib.rnn.GRUCell(self.num_hidden) for n in range(self.num_layers)]
                stack_fw = tf.contrib.rnn.MultiRNNCell(fw_cells)
                stack_bw = tf.contrib.rnn.MultiRNNCell(bw_cells)
                if initial_state == None:
                    self.initial_state_fw = stack_fw.zero_state(batch_size, dtype=tf.float32)
                    self.initial_state_bw = stack_bw.zero_state(batch_size, dtype=tf.float32)
                else:
                    assert len(initial_state) == 2, "initial_state length shoud be 2"
                    self.initial_state_fw = initial_state[0]
                    self.initial_state_bw = initial_state[1] 
                #(fw_outputs,bw_outputs), (fw_state,bw_state) = \
                (fw_outputs,bw_outputs), state = \
                    tf.nn.bidirectional_dynamic_rnn(stack_fw,
                                                    stack_bw,
                                                    inputs, 
                                                    seq_len, 
                                                    initial_state_fw = self.initial_state_fw, 
                                                    initial_state_bw = self.initial_state_bw, 
                                                    dtype=tf.float32)
                self.initial_state_fw = tf.identity(self.initial_state_fw, name='initial_state_fw')
                self.initial_state_bw = tf.identity(self.initial_state_bw, name='initial_state_bw')
                outputs = tf.concat((fw_outputs, bw_outputs), 2)
                #state = tf.concat((fw_state, bw_state), 1)
                state_for_feed = tf.identity(state, name="state")
                self.pb_nodes = [self.initial_state_fw.name.split(':')[0],
                                 self.initial_state_bw.name.split(':')[0],
                                 state_for_feed.name.split(':')[0]]
            else:
                raise ValueError("unknown rnn type")
            return outputs,state, state_for_feed

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
    elif type == 'xavier':
        return tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    elif type =='variance_scaling':
        return tf.variance_scaling_initializer(scale=1.0,mode="fan_in",
                                                        distribution="uniform",seed=None,dtype=tf.float32)
    else:
        raise ValueError('unknown type of initializer!')

def get_train_op(global_step, optimizer_type, loss, learning_rate, var_list = None, 
                 clip_grad = 5):
    with tf.variable_scope("train_step"):
        #self.global_step = tf.Variable(0, name="global_step", trainable=False)
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
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
            #grads_and_vars_clip = zip(grads_and_vars_clip, var_list)
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


