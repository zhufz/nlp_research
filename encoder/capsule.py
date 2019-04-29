import tensorflow as tf
from keras import backend as K
from utils import _conv2d_wrapper
import tensorflow.contrib.slim as slim
#refer:https://github.com/andyweizhao/capsule_text_classification/blob/master/network.py

class Capsule():
    def __init__(self, **kwargs):
        self.output_size = 128
        pass

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
                        inputs_poses_shape[0], inputs_poses_shape[1],
                        inputs_poses_shape[2], shape[3],
                        inputs_poses_shape[-1]]
                    )
            activations = tf.reshape(activations, [
                        inputs_poses_shape[0],inputs_poses_shape[1],
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
                    X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
                tf.logging.info('output shape: {}'.format(nets.get_shape()))
                nets = self.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                     padding='VALID', pose_shape=16, add_bias=True,
                                     name='primary')
                nets = self.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
                nets = self.capsule_flatten(nets)
                poses, activations = self.capsule_fc_layer(nets,
                                                           self.output_size, 3, 'fc2')
                poses_list.append(poses)
        poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
        activations = K.sqrt(K.sum(K.square(poses), 2))
        return poses

    def capsule_model_A(self, X):
        with tf.variable_scope('capsule_'+str(3)):
            nets = _conv2d_wrapper(
                    X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = self.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True,
                                 name='primary')
            nets = self.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = self.capsule_flatten(nets)
            poses, activations = self.capsule_fc_layer(nets, self.output_size, 3, 'fc2') 
        return poses

    def __call__(self, embed, reuse = tf.AUTO_REUSE):
        return capsule_model_A(embed)
