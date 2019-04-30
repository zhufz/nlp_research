import tensorflow as tf
from tensorflow.contrib import rnn

def rnn_layer(inputs, seq_len, num_hidden, num_layers, rnn_type, keep_prob):
    assert num_layers >0, "num_layers need larger than 0"
    assert num_hidden >0, "num_hidden need larger than 0"
    if rnn_type == 'lstm':
        cells = [rnn.LSTMCell(num_hidden, state_is_tuple=True) for n in range(num_layers)]
        stack = rnn.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
        #state = state[-1][1]
    elif rnn_type == 'gru':
        cells = [rnn.GRUCell(num_hidden) for n in range(num_layers)]
        stack = rnn.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    elif rnn_type == 'bi_lstm' and num_layers == 1:
        #sigle layer lstm
        cell_fw = rnn.LSTMCell(num_hidden)
        cell_bw = rnn.LSTMCell(num_hidden)
        (fw_outputs,bw_outputs), (fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        state = tf.concat((fw_state, bw_state), 2)
    elif rnn_type == 'bi_lstm':
        #multi layer lstm
        fw_cells = cells = [rnn.LSTMCell(num_hidden, state_is_tuple=True) for n in range(num_layers)]
        bw_cells = cells = [rnn.LSTMCell(num_hidden, state_is_tuple=True) for n in range(num_layers)]
        stack_fw = rnn.MultiRNNCell(fw_cells)
        stack_bw = rnn.MultiRNNCell(bw_cells)
        (fw_outputs,bw_outputs), (fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(stack_fw,stack_bw,inputs, seq_len, dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        state = tf.concat((fw_state, bw_state), 2)
        #state = state[-1][1]
    elif rnn_type == 'bi_gru':
        fw_cells = [rnn.GRUCell(num_hidden) for n in range(num_layers)]
        bw_cells = [rnn.GRUCell(num_hidden) for n in range(num_layers)]
        stack_fw = rnn.MultiRNNCell(fw_cells)
        stack_bw = rnn.MultiRNNCell(bw_cells)
        (fw_outputs,bw_outputs), (fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(stack_fw,stack_bw,inputs, seq_len, dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        state = tf.concat((fw_state, bw_state), 2)
    else:
        raise ValueError("unknown rnn type")
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

def get_trainp_op(global_step, optimizer_type, loss, clip_grad, lr_pl):
    with tf.variable_scope("train_step"):
        #self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if optimizer_type == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=lr_pl)
        elif optimizer_type == 'Adadelta':
            optim = tf.train.AdadeltaOptimizer(learning_rate=lr_pl)
        elif optimizer_type == 'Adagrad':
            optim = tf.train.AdagradOptimizer(learning_rate=lr_pl)
        elif optimizer_type == 'RMSProp':
            optim = tf.train.RMSPropOptimizer(learning_rate=lr_pl)
        elif optimizer_type == 'Momentum':
            optim = tf.train.MomentumOptimizer(learning_rate=lr_pl, momentum=0.9)
        elif optimizer_type == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate=lr_pl)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=lr_pl)

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


