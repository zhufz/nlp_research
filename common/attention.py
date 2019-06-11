#-*- coding:utf-8 -*-
import tensorflow as tf
#refer: https://github.com/sjvasquez/quora-duplicate-questions/blob/d93a4bd5edf8327bd5ed7900107ca18bf8bb2b2f/attend.py

def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if not dim:
        return tensor.shape.as_list()
    if dim:
        return tensor.shape.as_list()[dim]

def self_attention(a, a_lenghts, max_seq_len, scope = 'self-attention', reuse = False):
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.matmul(a, tf.transpose(a, (0, 2, 1)))
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, a_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)

def multiplicative_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                             scope='multiplicative-attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.matmul(a, tf.transpose(b, (0, 2, 1)))
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def additive_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                       scope='additive-attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        a = tf.expand_dims(a, 2)
        b = tf.expand_dims(b, 1)
        v = tf.get_variable(
            name='dot_weights',
            initializer=tf.variance_scaling_initializer(),
            shape=[hidden_units]
        )
        logits = tf.einsum('ijkl,l->ijk', tf.nn.tanh(a + b), v)
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def concat_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                     scope='concat-attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        a = tf.expand_dims(a, 2)
        b = tf.expand_dims(b, 1)
        c = tf.concat([a, b], axis=3)
        W = tf.get_variable(
            name='matmul_weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(c, -1), hidden_units]
        )
        cW = tf.einsum('ijkl,lm->ijkm', c, W)
        v = tf.get_variable(
            name='dot_weights',
            initializer=tf.ones_initializer(),
            shape=[hidden_units]
        )
        logits = tf.einsum('ijkl,l->ijk', tf.nn.tanh(cW), v)
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def dot_attention(a, b, a_lengths, b_lengths, max_seq_len):
    logits = tf.matmul(a, tf.transpose(b, (0, 2, 1)))
    logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
    attn = tf.exp(logits)
    attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
    return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def cosine_attention(a, b, a_lengths, b_lengths, max_seq_len):
    a_norm = tf.nn.l2_normalize(a, dim=2)
    b_norm = tf.nn.l2_normalize(b, dim=2)
    logits = tf.matmul(a_norm, tf.transpose(b_norm, (0, 2, 1)))
    logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
    attn = tf.exp(logits)
    attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
    return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def mask_attention_weights(weights, a_lengths, b_lengths, max_seq_len):
    a_mask = tf.expand_dims(tf.sequence_mask(a_lengths, maxlen=max_seq_len), 2)
    b_mask = tf.expand_dims(tf.sequence_mask(b_lengths, maxlen=max_seq_len), 1)
    seq_mask = tf.cast(tf.matmul(tf.cast(a_mask, tf.int32), tf.cast(b_mask, tf.int32)), tf.bool)
    return tf.where(seq_mask, weights, tf.zeros_like(weights))


def softmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
                               attention_func_kwargs={}):
    attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, **attention_func_kwargs)
    return tf.matmul(attn, b)


def maxpool_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
                               attention_func_kwargs={}):
    attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, **attention_func_kwargs)
    return tf.reduce_max(tf.einsum('ijk,ikl->ijkl', attn, b), axis=2)


def argmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
                              attention_func_kwargs={}):
    attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, **attention_func_kwargs)
    b_match_idx = tf.argmax(attn, axis=2)
    batch_index = tf.tile(tf.expand_dims(tf.range(shape(b, 0), dtype=tf.int64), 1), (1, max_seq_len))
    b_idx = tf.stack([batch_index, b_match_idx], axis=2)
    return tf.gather_nd(b, b_idx)
