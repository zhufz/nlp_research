#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import predictor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pdb
import re
import traceback
import pickle
import logging
import multiprocessing
import os,sys
from functools import partial
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

from embedding import embedding
from language_model.bert.modeling import get_assignment_map_from_checkpoint
from encoder import encoder
from utils.data_utils import *
from utils.preprocess import Preprocess
from utils.tf_utils import load_pb,write_pb
from utils.recall import Annoy
from common.layers import get_train_op
from common.loss import get_loss
from common.lr import cyclic_learning_rate
from common.metrics import metrics
from common.triplet import batch_hard_triplet_scores


class Match(object):
    def __init__(self, conf):
        self.task_type = 'match'
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.pre = Preprocess()
        self.model_loaded = False
        self.zdy = {}
        csv = pd.read_csv(self.ori_path, header = 0, sep="\t", error_bad_lines=False)

        if 'text' in csv.keys() and 'target' in csv.keys():
            #format: text \t target
            #for this format, the size for each class should be larger than 2 
            self.text_list = list(csv['text'])
            self.label_list = list(csv['target'])
            self.data_type = 'column_2'
        elif 'text_a' in csv.keys() and 'text_b' in csv.keys() and'target' in csv.keys():
            #format: text_a \t text_b \t target
            #for this format, target value can only be choosen from 0 or 1
            self.text_a_list = list(csv['text_a'])
            self.text_b_list = list(csv['text_b'])
            self.text_list = self.text_a_list + self.text_b_list
            self.label_list = list(csv['target'])
            self.data_type = 'column_3'
        else:
            raise ValueError('error format for train file')

        self.num_class = len(set(self.label_list))
        logging.info(">>>>>>>>>>>> class num:%s <<<<<<<<<<<<<<<"%self.num_class)
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in \
                          self.text_list]
        self.conf.update({
            "maxlen": self.maxlen,
            "maxlen1": self.maxlen,
            "maxlen2": self.maxlen,
            "num_class": self.num_class,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "num_output": self.num_output,
            "keep_prob": 1,
            "is_training": False,
        })
        self.encoder = encoder[self.encoder_type](**self.conf)

    def prepare(self):
        vocab_dict = embedding[self.embedding_type].build_dict(\
                                                               dict_path = self.dict_path,
                                                               text_list = self.text_list,
                                                               mode = self.mode)
        text2id = embedding[self.embedding_type].text2id
        self.gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
        self.gt.process(self.text_list, self.label_list, text2id,
                        self.encoder.encoder_fun, vocab_dict,
                        self.tfrecords_path, self.label_path, 
                        self.test_size, self.data_type)
        logging.info("tfrecords generated!")

    def create_model_fn(self):
        def init_embedding():
            vocab_dict = embedding[self.embedding_type].build_dict(\
                                                                   dict_path = self.dict_path,
                                                                   text_list = self.text_list,
                                                                   mode = self.mode)
            return  embedding[self.embedding_type](text_list = self.text_list,
                                                   vocab_dict = vocab_dict,
                                                   dict_path = self.dict_path,
                                                   random=self.rand_embedding,
                                                   maxlen = self.maxlen,
                                                   batch_size = self.batch_size,
                                                   embedding_size =
                                                   self.embedding_size,
                                                   conf = self.conf)

        def cal_loss(pred, labels, batch_size, conf):
            if self.tfrecords_mode == 'class':
                pos_scores, neg_scores = batch_hard_triplet_scores(labels, pred, is_distance = self.is_distance) # pos/neg scores
                pos_scores = tf.squeeze(pos_scores, -1)
                neg_scores = tf.squeeze(neg_scores, -1)
                #for represent, 
                #     pred is a batch of tensors which size >1
                #     we can use triplet loss(hinge loss) or contrastive loss
                #if use hinge loss, we don't need labels
                #if use other loss(contrastive loss), we need define pos/neg target before
                if self.loss_type in ['hinge_loss','improved_triplet_loss']:
                    #pairwise
                    loss = get_loss(type = self.loss_type, 
                                    pos_logits = pos_scores,
                                    neg_logits = neg_scores,
                                    **conf)
                else:
                    #pointwise
                    pos_target = tf.ones(shape = [int(self.batch_size)], dtype = tf.float32)
                    neg_target = tf.zeros(shape = [int(self.batch_size)], dtype = tf.float32)

                    pos_loss = get_loss(type = self.loss_type, logits = pos_scores, labels =
                                    pos_target, **conf)
                    neg_loss = get_loss(type = self.loss_type, logits = neg_scores, labels =
                                    neg_target, **conf)
                    loss = pos_loss + neg_loss

            elif self.tfrecords_mode in ['pair','point']:
                if self.loss_type in ['hinge_loss','improved_triplet_loss']:
                    assert self.tfrecords_mode == 'pair', "only pair mode can provide <query, pos, neg> format data"
                    #pairwise
                    if self.num_output == 1:
                        pred = tf.nn.sigmoid(pred)
                    elif self.num_output == 2:
                        pred = tf.nn.softmax(pred)[:,0]
                        pred = tf.expand_dims(pred,-1)
                    else:
                        raise ValueError('unsupported num_output, 1(sigmoid) or 2(softmax)?')
                    pos_scores = tf.strided_slice(pred, [0], [batch_size], [2])
                    neg_scores = tf.strided_slice(pred, [1], [batch_size], [2])
                    loss = get_loss(type = self.loss_type, 
                                    pos_logits = pos_scores,
                                    neg_logits = neg_scores,
                                    **conf)
                elif self.loss_type in ['sigmoid_loss']:
                    #pointwise
                    labels = tf.expand_dims(labels,axis=-1)
                    loss = get_loss(type = self.loss_type, logits = pred, labels =
                                    labels, **conf)
                else:
                    raise ValueError('unsupported loss for pair/point match')
            else:
                raise ValueError('unknown tfrecords_mode?')
            return loss

        def model_fn(features, labels, mode, params):
            #model params
            self.encoder.keep_prob = params['keep_prob']
            self.encoder.is_training = params['is_training']
            global_step = tf.train.get_or_create_global_step()

            ############# encode #################
            if not self.use_language_model:
                self.embedding = init_embedding()
                if self.tfrecords_mode == 'class':
                    self.embed_query = self.embedding(features = features, name = 'x_query')
                    output = self.encoder(self.embed_query, 
                                          name = 'x_query', 
                                          features = features)
                    output = tf.nn.l2_normalize(output, -1)

                elif self.tfrecords_mode in ['pair','point']:
                    if self.sim_mode == 'cross':
                        self.embed_query = self.embedding(features = features, name = 'x_query')
                        self.embed_sample = self.embedding(features = features, name = 'x_sample')
                        output = self.encoder(x_query = self.embed_query, 
                                              x_sample = self.embed_sample,
                                              features = features)
                    elif self.sim_mode == 'represent':
                        self.embed_query = self.embedding(features = features, name = 'x_query')
                        self.embed_sample = self.embedding(features = features, name = 'x_sample')
                        query_encode = self.encoder(self.embed_query, 
                                                    name = 'x_query', 
                                                    features = features)
                        sample_encode = self.encoder(self.embed_sample, 
                                                     name = 'x_sample', 
                                                     features = features)
                        output = self.concat(query_encode, sample_encode)
                        output = tf.layers.dense(output,
                                                1,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                                name='fc')
                    else:
                        raise ValueError('unknown sim_mode, represent or cross')
            else:
                output = self.encoder(features = features)

            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                #pdb.set_trace()
                predictions = {
                    'encode': output,
                    'pred': tf.cast(tf.greater(tf.nn.softmax(output)[:,0], 0.5),
                                    tf.int32) if self.num_output == 2 else 
                            tf.cast(tf.greater(tf.nn.sigmoid(output), 0.5), tf.int32),
                    'score': tf.nn.softmax(output)[:,0] if self.num_output == 2 else tf.nn.sigmoid(output),
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            ############### loss ##################
            loss = cal_loss(output, labels, self.batch_size, self.conf)

            ############### train ##################
            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.use_clr:
                    self.learning_rate = cyclic_learning_rate(global_step=global_step,
                                                              learning_rate = self.learning_rate, 
                                                              max_lr = self.max_learning_rate,
                                                              step_size = self.step_size,
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
            ############### eval ##################
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {}
                #{"accuracy": tf.metrics.accuracy(
                #    labels=labels, predictions=predictions["classes"])}
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return model_fn

    def create_input_fn(self, mode):
        n_cpu = multiprocessing.cpu_count()
        def train_input_fn():
            if self.tfrecords_mode  == 'class':
                #size = self.num_class
                num_classes_per_batch = 32
                assert num_classes_per_batch < self.num_class
                num_sentences_per_class = self.batch_size // num_classes_per_batch
            elif self.tfrecords_mode == 'pair':
                #data order: query,pos,query,neg
                num_sentences_per_class = 4
                num_classes_per_batch = self.batch_size // num_sentences_per_class
            elif self.tfrecords_mode  == 'point':
                #data order: query, sample(pos or neg)
                num_classes_per_batch = 2
                num_sentences_per_class = self.batch_size // num_classes_per_batch
            else:
                raise ValueError('unknown tfrecords_mode')

            #filenames = ["{}/train_class_{:04d}".format(self.tfrecords_path,i) \
            #                 for i in range(size)]
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith('train')]
            size = len(filenames)
            if size == 0:
                logging.error('no processed tfrecords data finded!')
            logging.info("tfrecords train class num: {}".format(size))
            datasets = [tf.data.TFRecordDataset(filename) for filename in filenames]
            datasets = [dataset.repeat() for dataset in datasets]
            #datasets = [dataset.shuffle(buffer_size=1000) for dataset in datasets]
            def generator():
                while True:
                    labels = np.random.choice(range(size),
                                              num_classes_per_batch,
                                              replace=False)
                    for label in labels:
                        for _ in range(num_sentences_per_class):
                            yield label

            choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
            dataset = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)
            gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
            dataset = dataset.map(lambda record: gt.parse_record(record, self.encoder),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(4*self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            ##test
            #pdb.set_trace()
            #sess = tf.Session()
            #features1,label1 = sess.run([features,label])
            #features1['x_query_pred'] = [item.decode('utf-8') for item in features1['x_query_pred'][1]]
            #features1['x_sample_pred'] = [item.decode('utf-8') for item in features1['x_sample_pred'][1]]
            return features, label

        def test_input_fn(mode):
            #filenames = ["{}/{}_class_{:04d}".format(self.tfrecords_path,mode,i) \
            #                 for i in range(self.num_class * self.test_size)]
            filenames = [os.path.join(self.tfrecords_path,item) for item in 
                         os.listdir(self.tfrecords_path) if item.startswith(mode)]
            assert self.num_class == len(filenames), "the num of tfrecords file error!"
            logging.info("tfrecords test class num: {}".format(len(filenames)))
            dataset = tf.data.TFRecordDataset(filenames)
            gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
            dataset = dataset.map(lambda record: gt.parse_record(record, self.encoder),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            return features, label

        if mode == 'train':
            return train_input_fn
        elif mode == 'test':
            return lambda : test_input_fn("test")
        elif mode == 'label':
            return lambda : test_input_fn("train")
        else:
            raise ValueError("unknown input_fn type!")

    def train(self):
        params = {
            'is_training': True,
            'keep_prob': 0.7,
        }
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
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params,
                                           warm_start_from = ws)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps =
                        self.max_steps)
        self.save()

    def save(self):
        params = {
            'is_training': False,
            'keep_prob': 1
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params)
        def serving_input_receiver_fn():
            features = {'x_query': tf.placeholder(dtype=tf.int64, 
                                                  shape=[None, self.maxlen],
                                                  name='x_query'),
                        'x_query_length': tf.placeholder(dtype=tf.int64,
                                                         shape=[None],
                                                         name='x_query_length'),
                        'label': tf.placeholder(dtype=tf.int64, 
                                                shape=[None],
                                                name='label')}
            if self.tfrecords_mode in ['pair','point']:
                features.update({'x_sample': tf.placeholder(dtype=tf.int64, 
                                                      shape=[None, self.maxlen],
                                                      name='x_sample'),
                                'x_sample_length': tf.placeholder(dtype=tf.int64,
                                                             shape=[None],
                                                             name='x_sample_length')})
            features.update(self.encoder.get_features())
            return tf.estimator.export.ServingInputReceiver(features, features)

        estimator.export_savedmodel(
            self.export_dir_path, # 目录
            serving_input_receiver_fn, # 返回ServingInputReceiver的函数
            assets_extra=None,
            as_text = False,
            checkpoint_path=None)

    def test(self):
        params = {
            'is_training': False,
            'keep_prob': 1
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params)
        predictions = estimator.predict(input_fn=self.create_input_fn("test"))
        predictions = list(predictions)

        if self.tfrecords_mode == 'class':
            predictions_vec = [item['encode'] for item in predictions]
            predictions_label = [item['label'] for item in predictions]
            refers = estimator.predict(input_fn=self.create_input_fn("label"))
            refers = list(refers) 

            refers_vec = [item['encode'] for item in refers]
            refers_label = [item['label'] for item in refers]

            right = 0
            thre_right = 0
            sum = 0

            if self.is_distance:
                scores = euclidean_distances(predictions_vec, refers_vec)
                selected_ids = np.argmin(scores, axis=-1)
            else:
                scores = cosine_similarity(predictions_vec, refers_vec)
                selected_ids = np.argmax(scores, axis=-1)
            for idx, item in enumerate(selected_ids):
                if refers_label[item] == predictions_label[idx]:
                    if self.is_distance:
                        if 1 - scores[idx][item] > self.score_thre:
                            thre_right += 1
                    else:
                        if scores[idx][item] > self.score_thre:
                            thre_right += 1
                    right += 1
                sum += 1
            print("Acc:{}".format(float(right)/sum))
            print("ThreAcc:{}".format(float(thre_right)/sum))
        elif self.tfrecords_mode == 'pair':
            #对于pair方式的评估
            scores = [item['score'] for item in predictions]
            labels = [item['label'] for item in predictions]
            #pdb.set_trace()

            #predictions
            scores = np.reshape(scores,[self.num_class*self.test_size, -1])
            pred_max_ids = np.argmax(scores, axis = -1)
            #label
            labels = np.reshape(labels,[self.num_class, -1])

            right = 0
            for idx,max_id in enumerate(pred_max_ids):
                if labels[idx][max_id] == 1:
                    right += 1
            sum = len(pred_max_ids)
            print("Acc:{}".format(float(right)/sum))

        elif self.tfrecords_mode == 'point':
            scores = [item['score'] for item in predictions]
            scores = np.reshape(scores, -1)
            scores = [0 if item < self.score_thre else 1 for item in scores]
            #pred = [item['pred'] for item in predictions]
            labels = [item['label'] for item in predictions]
            res = metrics(labels = labels, logits = np.array(scores))
            print("precision:{} recall:{} f1:{}".format(res[3],res[4],res[5]))


    def concat(self, a, b):
        tmp = tf.concat([a,b], axis = -1)
        #return tmp
        res1 = a*b
        res2 = a+b
        res3 = a-b
        return tf.concat([tmp, res1, res2, res3], axis = -1)

    def knn(self, scores, predictions_label, refers_label, k = 4):
        sorted_id = np.argsort(-scores, axis = -1)
        shape = np.shape(sorted_id)
        max_id = []
        for idx in range(shape[0]):
            mp = defaultdict(int)
            for idy in range(k):
                mp[refers_label[int(sorted_id[idx][idy])]] += 1
            max_id.append(max(mp,key=mp.get))
        return max_id
