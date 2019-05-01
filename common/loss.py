import tensorflow as tf
import numpy as np

def get_default_value(kwargs, key, value):
    if key in kwargs:
        return kwargs[key]
    else:
        return value

def get_loss(logits, labels, type = 'cross', labels_sparse = False,  **kwargs):
    if labels_sparse == True:
        num = logits.shape.as_list()[-1]
        labels = tf.one_hot(labels,num)

    if type == 'focal_loss':
        gamma = get_default_value(kwargs, 'gamma', 2.0)
        alpha = get_default_value(kwargs, 'alpha', 0.25)
        epsilon = get_default_value(kwargs, 'epsilon', 1e-8)
        return focal_loss(logits, labels, gamma, alpha, epsilon)
    elif type == 'sigmoid_loss':
        return sigmoid_cross_entropy(logits, labels)
    elif type == 'softmax_loss':
        return softmax_cross_entropy(logits, labels)
    elif type == 'margin_loss':
        return margin_loss(logits, labels)
    else:
        raise ValueError("unknown loss type")

def focal_loss(logits, labels, gamma=2.0, alpha=0.25, epsilon=1e-8):
    logits = tf.cast(logits, tf.float32)
    model_out = tf.add(logits, epsilon)
    ce = tf.multiply(tf.cast(labels, tf.float32), -tf.log(model_out))
    weights = tf.multiply(tf.cast(labels, tf.float32), tf.pow(tf.subtract(1.0, model_out), gamma))
    return tf.reduce_mean(tf.multiply(alpha, tf.multiply(weights, ce)))

def sigmoid_cross_entropy(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                                                labels=tf.cast(labels,tf.float32))
    loss = tf.reduce_mean(loss)
    return loss

def softmax_cross_entropy(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                labels=tf.cast(labels,tf.float32))
    loss = tf.reduce_mean(loss)
    return loss

def margin_loss(logits, labels):
    labels = tf.cast(labels,tf.float32)
    loss = labels * tf.square(tf.maximum(0., 0.9 - logits)) + \
        0.25 * (1.0 - labels) * tf.square(tf.maximum(0., logits - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss
