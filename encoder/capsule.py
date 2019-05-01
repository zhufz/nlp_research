import tensorflow as tf
import keras
from keras import backend as K
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import pdb
#refer:https://github.com/andyweizhao/capsule_text_classification/blob/master/network.py

epsilon = 1e-9

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

def squash_v1(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def squash_v0(s, axis=-1, epsilon=1e-7, name=None):
    s_squared_norm = K.sum(K.square(s), axis, keepdims=True) + K.epsilon()
    safe_norm = K.sqrt(s_squared_norm)
    scale = 1 - tf.exp(-safe_norm)
    return scale * s / safe_norm

def routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations):
    b = keras.backend.zeros_like(u_hat_vecs[:,:,:,0])
    if i_activations is not None:
        i_activations = i_activations[...,tf.newaxis]
    for i in range(iterations):
        if False:
            leak = tf.zeros_like(b, optimize=True)
            leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
            leaky_logits = tf.concat([leak, b], axis=1)
            leaky_routing = tf.nn.softmax(leaky_logits, dim=1)
            c = tf.split(leaky_routing, [1, output_capsule_num], axis=1)[1]
        else:
            c = softmax(b, 1)
#        if i_activations is not None:
#            tf.transpose(tf.transpose(c, perm=[0,2,1]) * i_activations, perm=[0,2,1])
        outputs = squash_v1(K.batch_dot(c, u_hat_vecs, [2, 2]))
        if i < iterations - 1:
            b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])
    poses = outputs
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations

def _matmul_broadcast(x, y, name):
  """Compute x @ y, broadcasting over the first `N - 2` ranks.
  """
  with tf.variable_scope(name) as scope:
    return tf.reduce_sum(
      tf.nn.dropout(x[..., tf.newaxis] * y[..., tf.newaxis, :, :],1), axis=-2
    )


def _get_variable_wrapper(
  name, shape=None, dtype=None, initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None
):
  """Wrapper over tf.get_variable().
  """

  with tf.device('/cpu:0'):
    var = tf.get_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )
  return var


def _get_weights_wrapper(
  name, shape, dtype=tf.float32, initializer=initializers.xavier_initializer(),
  weights_decay_factor=None
):
  """Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """

  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'
    )

    tf.add_to_collection('losses', weights_wd)

  return weights


def _get_biases_wrapper(
  name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)
):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  return biases


def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name, stddev=0.1):
  """Wrapper over tf.nn.conv2d().
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0, #initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    )
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.separable_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0
    )
    pkernel = _get_weights_wrapper(
      name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0
    )
    output = tf.nn.separable_conv2d(
      input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[pointwise_shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.depthwise_conv2d(
      inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(
        name='biases', shape=[d_]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

    return output


def vec_transformationByConv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
    kernel = _get_weights_wrapper(
      name='weights', shape=[1, input_capsule_dim, output_capsule_dim*output_capsule_num], weights_decay_factor=0.0
    )
    tf.logging.info('poses: {}'.format(poses.get_shape()))
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = keras.backend.conv1d(poses, kernel)
    u_hat_vecs = keras.backend.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
    return u_hat_vecs

def vec_transformationByMat(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num, shared=True):
    inputs_poses_shape = poses.get_shape().as_list()
    poses = poses[..., tf.newaxis, :]
    poses = tf.tile(
              poses, [1, 1, output_capsule_num, 1]
            )
    if shared:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, 1, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], input_capsule_num, 1, 1, 1]
                )
    else:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, input_capsule_num, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], 1, 1, 1, 1]
                )
    tf.logging.info('poses: {}'.format(poses[...,tf.newaxis].get_shape()))
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = tf.squeeze(tf.matmul(kernel, poses[...,tf.newaxis]),axis=-1)
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
    return u_hat_vecs


class Capsule():
    def __init__(self, **kwargs):
        self.seq_length = kwargs['maxlen']
        self.embedding_size = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']

    def capsules_init(self, inputs, shape, strides, padding, pose_shape, add_bias, name):
        with tf.variable_scope(name):
            poses = _conv2d_wrapper(
              inputs,
              shape=shape[0:-1] + [shape[-1] * pose_shape],
              strides=strides,
              padding=padding,
              add_bias=add_bias,
              activation_fn=None,
              name='pose_stacked'
            )
            poses_shape = poses.get_shape().as_list()
            poses = tf.reshape(
                        poses, [
                            -1, poses_shape[1], poses_shape[2], shape[-1], pose_shape
                        ])
            beta_a = _get_weights_wrapper(
                            name='beta_a', shape=[1, shape[-1]]
            )
            poses = squash_v1(poses, axis=-1)
            activations = K.sqrt(K.sum(K.square(poses), axis=-1)) + beta_a
            tf.logging.info("prim poses dimension:{}".format(poses.get_shape()))
        return poses, activations

    def capsule_flatten(self, nets):
        poses, activations = nets
        input_pose_shape = poses.get_shape().as_list()
        poses = tf.reshape(poses, [
                        -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3], input_pose_shape[-1]])
        activations = tf.reshape(activations, [
                        -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3]])
        tf.logging.info("flatten poses dimension:{}".format(poses.get_shape()))
        tf.logging.info("flatten activations dimension:{}".format(activations.get_shape()))
        return poses, activations

    def capsule_conv_layer(self, nets, shape, strides, iterations, name):
        with tf.variable_scope(name):
            poses, i_activations = nets
            inputs_poses_shape = poses.get_shape().as_list()
            hk_offsets = [
              [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in
              range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
            ]
            wk_offsets = [
              [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in
              range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
            ]
            inputs_poses_patches = tf.transpose(
              tf.gather(
                tf.gather(
                  poses, hk_offsets, axis=1, name='gather_poses_height_kernel'
                ), wk_offsets, axis=3, name='gather_poses_width_kernel'
              ), perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches'
            )
            tf.logging.info('i_poses_patches shape: {}'.format(inputs_poses_patches.get_shape()))
            inputs_poses_shape = inputs_poses_patches.get_shape().as_list()
            inputs_poses_patches = tf.reshape(inputs_poses_patches, [
                                    -1, shape[0]*shape[1]*shape[2], inputs_poses_shape[-1]
                                    ])
            i_activations_patches = tf.transpose(
              tf.gather(
                tf.gather(
                  i_activations, hk_offsets, axis=1, name='gather_activations_height_kernel'
                ), wk_offsets, axis=3, name='gather_activations_width_kernel'
              ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
            )
            tf.logging.info('i_activations_patches shape: {}'.format(i_activations_patches.get_shape()))
            i_activations_patches = tf.reshape(i_activations_patches, [
                                    -1, shape[0]*shape[1]*shape[2]]
                                    )
            u_hat_vecs = vec_transformationByConv(
                      inputs_poses_patches,
                      inputs_poses_shape[-1], shape[0]*shape[1]*shape[2],
                      inputs_poses_shape[-1], shape[3],
                      )
            tf.logging.info('capsule conv votes shape: {}'.format(u_hat_vecs.get_shape()))
            beta_a = _get_weights_wrapper(
                    name='beta_a', shape=[1, shape[3]]
                    )
            poses, activations = routing(u_hat_vecs, beta_a, iterations, shape[3], i_activations_patches)
            poses = tf.reshape(poses, [
                        #inputs_poses_shape[0], inputs_poses_shape[1],
                        -1, inputs_poses_shape[1],
                        inputs_poses_shape[2], shape[3],
                        inputs_poses_shape[-1]]
                    )
            activations = tf.reshape(activations, [
                        #inputs_poses_shape[0],inputs_poses_shape[1],
                        -1,inputs_poses_shape[1],
                        inputs_poses_shape[2],shape[3]]
                    )
            nets = poses, activations
        tf.logging.info("capsule conv poses dimension:{}".format(poses.get_shape()))
        tf.logging.info("capsule conv activations dimension:{}".format(activations.get_shape()))
        return nets

    def capsule_fc_layer(self, nets, output_capsule_num, iterations, name):
        with tf.variable_scope(name):
            poses, i_activations = nets
            input_pose_shape = poses.get_shape().as_list()
            u_hat_vecs = vec_transformationByConv(
                          poses,
                          input_pose_shape[-1], input_pose_shape[1],
                          input_pose_shape[-1], output_capsule_num,
                          )
            tf.logging.info('votes shape: {}'.format(u_hat_vecs.get_shape()))
            beta_a = _get_weights_wrapper(
                    name='beta_a', shape=[1, output_capsule_num]
                    )
            poses, activations = routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations)
            tf.logging.info('capsule fc shape: {}'.format(poses.get_shape()))   
        return poses, activations

    def capsule_model_B(self, X):
        poses_list = []
        for _, ngram in enumerate([3,4,5]):
            with tf.variable_scope('capsule_'+str(ngram)):
                nets = _conv2d_wrapper(
                    X, shape=[ngram, self.embedding_size, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
                tf.logging.info('output shape: {}'.format(nets.get_shape()))
                nets = self.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                     padding='VALID', pose_shape=16, add_bias=True,
                                     name='primary')
                nets = self.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
                nets = self.capsule_flatten(nets)
                poses, activations = self.capsule_fc_layer(nets,
                                                           self.num_output, 3, 'fc2')
                poses_list.append(poses)
        poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
        activations = K.sqrt(K.sum(K.square(poses), 2))
        return activations

    def capsule_model_A(self, X):
        with tf.variable_scope('capsule_'+str(3)):
            nets = _conv2d_wrapper(
                    X, shape=[3, self.embedding_size, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = self.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True,
                                 name='primary')
            nets = self.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = self.capsule_flatten(nets)
            poses, activations = self.capsule_fc_layer(nets, self.num_output, 3, 'fc2') 
        return activations

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict

    def __call__(self, embed, reuse = tf.AUTO_REUSE):
        embed = tf.expand_dims(embed, -1)
        return self.capsule_model_A(embed)
