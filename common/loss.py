import tensorflow as tf
import numpy as np

def get_default_value(kwargs, key, value):
    if key in kwargs:
        return kwargs[key]
    else:
        return value

def get_loss(logits = None, labels = None, neg_logits = None, 
             pos_logits = None, type = 'softmax_loss', labels_sparse = False,  **kwargs):
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

    elif type == 'am_softmax_loss':
        m = get_default_value(kwargs, 'm', 0.35)
        s = get_default_value(kwargs, 's', 5)
        return am_softmax_loss(logits, labels, m ,s)

    elif type == 'margin_loss':
        return margin_loss(logits, labels)

    elif type == 'l1_loss':
        return l1_loss(logits, labels)
    elif type == 'l2_loss':
        return l2_loss(logits, labels)

    elif type == 'hinge_loss':
        margin = get_default_value(kwargs, 'margin', 0.8)
        is_distance = get_default_value(kwargs, 'is_distance', True)
        return hinge_loss(neg_logits, pos_logits, margin, is_distance)

    elif type == 'improved_triplet_loss':
        margin = get_default_value(kwargs, 'margin', 0.8)
        margin_pos = get_default_value(kwargs, 'margin_pos', 0.01)
        is_distance = get_default_value(kwargs, 'is_distance', True)
        return improved_triplet_loss(neg_logits, pos_logits, margin, 
                                     margin_pos, is_distance)
    else:
        raise ValueError("unknown loss type")

def focal_loss(logits, labels, gamma, alpha, epsilon):
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

def am_softmax_loss(labels, logits, m, s):
    logits = labels * (logits - m) + (1 - labels) * logits
    logits *= s
    loss = softmax_cross_entropy(logits, labels)
    return loss

def margin_loss(logits, labels):
    # logits = tf.nn.softmax(logits)
    labels = tf.cast(labels,tf.float32)
    loss = labels * tf.square(tf.maximum(0., 0.9 - logits)) + \
        0.25 * (1.0 - labels) * tf.square(tf.maximum(0., logits - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

def l1_loss(logits, labels):
    return tf.reduce_mean(tf.abs(logits - labels))

def l2_loss(logits, labels):
    return tf.reduce_mean(tf.square(logits - labels))

def hinge_loss(neg_logits, pos_logits, margin, is_distance):
    if is_distance:
        loss = tf.reduce_mean(tf.maximum(margin + pos_logits - neg_logits, 0.0))
    else:
        loss = tf.reduce_mean(tf.maximum(margin - pos_logits + neg_logits, 0.0))
    return loss

def improved_triplet_loss(neg_logits, pos_logits, margin, margin_pos, is_distance):
    if is_distance:
        loss = tf.reduce_mean(tf.maximum(margin + pos_logits - neg_logits, 0.0) +
                              tf.square(1 - neg_logits))
                              #tf.square(pos_logits))
    else:
        loss = tf.reduce_mean(tf.maximum(margin - pos_logits + neg_logits, 0.0) + 
                              tf.square(neg_logits))
                              #tf.square(1 - pos_logits))
    return loss
