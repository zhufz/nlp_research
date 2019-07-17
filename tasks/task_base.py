#-*- coding: utf-8 -*-
import tensorflow as tf
from functools import partial
from common.lr import cyclic_learning_rate
from common.layers import get_train_op
from embedding import embedding
from encoder import encoder
from utils.data_utils import GenerateTfrecords
import logging
import re
import pdb

class TaskBase(object):
    def __init__(self, conf):
        for attr in conf:
            setattr(self, attr, conf[attr])

    def read_data(self):
        """you can load data in different formats for different task 
        """
        raise NotImplementedError('subclasses must override read_data()!')

    def prepare(self):
        """convert text_list to tfrecords
        """
        vocab_dict = embedding[self.embedding_type].build_dict(\
                                                               dict_path = self.dict_path,
                                                               text_list = self.text_list,
                                                               mode = self.mode)
        text2id = embedding[self.embedding_type].text2id
        self.gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
        self.gt.process(self.text_list, self.label_list, text2id,
                        self.encoder.encoder_fun, vocab_dict,
                        self.tfrecords_path, self.label_path, 
                        self.dev_size, self.data_type, mode = self.mode)
        logging.info("tfrecords generated!")

    def init_embedding(self):
        """init embedding object
        """
        vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.dict_path,
                                            text_list = self.text_list,
                                            mode = self.mode)
        _embedding = embedding[self.embedding_type](text_list = self.text_list,
                                                   vocab_dict = vocab_dict,
                                                   dict_path = self.dict_path,
                                                   random=self.rand_embedding,
                                                   maxlen = self.maxlen,
                                                   batch_size = self.batch_size,
                                                   embedding_size = self.embedding_size,
                                                   conf = self.conf)
        return _embedding, vocab_dict

    def train_estimator_spec(self, mode, loss, global_step, params):
        """generate optimizer which can apply different learning rate in base
        model and downstream model,and return estimatorspec for training
        """
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

    def get_train_estimator(self, model_fn, params):
        """use params and model_fn to init estimator for training
           if 'base_var' is exited in params, so base model will be initialized
        """
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        if self.use_language_model:
            init_vars = tf.train.list_variables(self.init_checkpoint_path)
            init_vars_name = []
            for x in list(init_vars):
                (name, var) = (x[0], x[1])
                init_vars_name.append(name)
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.init_checkpoint_path,
                        vars_to_warm_start=init_vars_name)
            params.update({'base_var':init_vars_name})
        else:
            ws = None
        estimator = tf.estimator.Estimator(model_fn = model_fn,
                                           config = config,
                                           params = params,
                                           warm_start_from = ws)
        return estimator

    def save_model(self, model_fn, params, get_features):
        """save model
        """
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = model_fn,
                                           config = config,
                                           params = params)
        def serving_input_receiver_fn():
            features = get_features()
            return tf.estimator.export.ServingInputReceiver(features, features)

        estimator.export_savedmodel(
            self.export_dir_path, # 目录
            serving_input_receiver_fn, # 返回ServingInputReceiver的函数
            assets_extra=None,
            as_text=False,
            checkpoint_path=None)
