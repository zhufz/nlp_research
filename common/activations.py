#-*- coding:utf-8 -*-
import tensorflow as tf

def softplus(x, name="softplus"):
    with tf.variable_scope(name):
        return tf.nn.softplus

def swish(x, name="swish"):
    """f(x) = sigmoid(x) * x
    """
    with tf.variable_scope(name):
        return (tf.nn.sigmoid(x * 1.0) * x)

def leaky_relu(x, leak=0.2, name="leaky_relu"):
    """f(x) = max(alpha * x, x)
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)

def cube(x, name="cube_act"):
    """f(x) = pow(x, 3)
    """
    with tf.variable_scope(name):
        return tf.pow(x, 3)

def penalized_tanh(x, name="penalized_tanh"):
    """f(x) = max(tanh(x), alpha * tanh(x))
    """
    with tf.variable_scope(name):
        alpha = 0.25
        return tf.maximum(tf.tanh(x), alpha * tf.tanh(x))

def cosper(x, name="cosper_act"):
    """f(x) = cos(x) - x
    """
    with tf.variable_scope(name):
        return (tf.cos(x) - x)

def minsin(x, name="minsin_act"):
    """f(x) = min(x, xin(x))
    """
    with tf.variable_scope(name):
        return tf.minimum(x, tf.sin(x))

def tanhrev(x, name="tanhprev"):
    """f(x) = pow(atan(x), 2) - x
    """
    with tf.variable_scope(name):
        return (tf.pow(tf.atan(x), 2) - x)

def maxsig(x, name="maxsig_act"):
    """f(x) = max(x, tf.sigmiod(x))
    """
    with tf.variable_scope(name):
        return tf.maximum(x, tf.sigmoid(x))

def maxtanh(x, name="max_tanh_act"):
    """f(x) = max(x, tanh(x))
    """
    with tf.variable_scope(name):
        return tf.maximum(x, tf.tanh(x))

def get_activation(active_type = 'swish', **kwargs):
    mp = {
        "sigmoid": tf.nn.sigmoid,
        "tanh": tf.nn.tanh,
        "softsign": tf.nn.softsign,
        "relu": tf.nn.relu,
        "leaky_relu": leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "swish": swish,
        "sin": tf.sin,
        "cube": cube,
        "penalized_tanh": penalized_tanh,
        "cosper": cosper,
        "minsin": minsin,
        "tanhrev": tanhrev,
        "maxsig": maxsig,
        "maxtanh": maxtanh,
        "softplus": tf.nn.softplus,
        }
    assert active_type in mp, "%s is not in activation list"
    return mp[atype]


