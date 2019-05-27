import tensorflow as tf
from tensorflow.contrib import predictor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
import pdb
import traceback
import pickle
import logging
import multiprocessing
import os,sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

from embedding import embedding
from encoder import encoder
from utils.data_utils import *
from utils.preprocess import Preprocess
from utils.tf_utils import load_pb,write_pb
from utils.recall import Annoy
from common.layers import get_train_op
from common.loss import get_loss
from common.lr import cyclic_learning_rate
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
        csv = pd.read_csv(self.ori_path, header = 0, sep=",", error_bad_lines=False)
        self.text_list = list(csv['text'])
        self.label_list = list(csv['target'])
        self.num_class = len(set(self.label_list))
        logging.info(f">>>>>>>>>>>> class num:{self.num_class} <<<<<<<<<<<<<<<")
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

    def init_embedding(self):
        self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.dict_path,
                                            text_list = self.text_list,
                                            mode = self.mode)
        self.embedding = embedding[self.embedding_type](text_list = self.text_list,
                                                        vocab_dict = self.vocab_dict,
                                                        dict_path = self.dict_path,
                                                        random=self.rand_embedding,
                                                        maxlen = self.maxlen,
                                                        batch_size = self.batch_size,
                                                        embedding_size =
                                                        self.embedding_size,
                                                        conf = self.conf)

    def prepare(self):
        self.init_embedding()
        self.gt = GenerateTfrecords(self.tfrecords_mode, self.maxlen)
        self.gt.process(self.text_list, self.label_list, self.embedding.text2id,
                        self.encoder.encoder_fun, self.vocab_dict,
                        self.tfrecords_path, self.label_path, self.test_size)
        logging.info("tfrecords generated!")

    def cal_loss(self, pred, labels, pos_target, neg_target, batch_size, conf):
        if self.sim_mode == 'represent':
            pos_scores, neg_scores = batch_hard_triplet_scores(labels, pred) # pos/neg scores
            pos_scores = tf.squeeze(pos_scores, -1)
            neg_scores = tf.squeeze(neg_scores, -1)
            #for represent, 
            #     pred is a batch of tensors which size >1
            #     we can use triplet loss(hinge loss) or contrastive loss
            #if use hinge loss, we don't need labels
            #if use other loss(contrastive loss), we need define pos/neg target before
            if self.loss_type == 'hinge_loss':
                loss = get_loss(type = self.loss_type, 
                                pos_logits = pos_scores,
                                neg_logits = neg_scores,
                                **conf)
            else:
                pos_loss = get_loss(type = self.loss_type, logits = pos_scores, labels =
                                pos_target, **conf)
                neg_loss = get_loss(type = self.loss_type, logits = neg_scores, labels =
                                neg_target, **conf)
                loss = pos_loss + neg_loss

        elif self.sim_mode == 'cross':
            #for cross:
            #   pred is a batch of tensors which size == 1
            loss = get_loss(type = self.loss_type, logits = pred, labels =
                                labels, **conf)
        else:
            raise ValueError('unknown sim mode, cross or represent?')
        return loss

    def create_model_fn(self):
        def model_fn(features, labels, mode, params):
            ########### embedding #################
            if not self.use_language_model:
                self.init_embedding()
                if self.tfrecords_mode == 'class':
                    self.embed_query = self.embedding(features = features, name = 'x_query')
                else:
                    self.embed_query = self.embedding(features = features, name = 'x_query')
                    self.embed_sample = self.embedding(features = features, name = 'x_sample')
            else:
                self.embedding = None
            #############  encoder  #################
            #model params
            self.encoder.keep_prob = params['keep_prob']
            self.encoder.is_training = params['is_training']
            global_step = tf.train.get_or_create_global_step()
            if self.sim_mode == 'cross':
                if not self.use_language_model:
                    pred = self.encoder(x_query = self.embed_query, 
                                        x_sample = self.embed_sample,
                                        features = features)
                else:
                    pred = self.encoder(features = features)
            elif self.sim_mode == 'represent':
                if not self.use_language_model:
                    #features['x_query_length'] = features['length']
                    pred = self.encoder(self.embed_query, 
                                                     name = 'x_query', 
                                                     features = features)
                else:
                    pred = self.encoder(features = features)
            else:
                raise ValueError('unknown sim mode')

            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': pred,
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            ############### loss ##################
            pos_target = tf.ones(shape = [int(self.batch_size)], dtype = tf.float32)
            neg_target = tf.zeros(shape = [int(self.batch_size)], dtype = tf.float32)
            loss = self.cal_loss(pred,
                             labels,
                             pos_target,
                             neg_target,
                             self.batch_size,
                             self.conf)
            ############### train ##################
            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.use_clr:
                    self.learning_rate = cyclic_learning_rate(global_step=global_step,
                                                          learning_rate = self.learning_rate, 
                                                          mode = self.clr_mode)
                optimizer = get_train_op(global_step, 
                                         self.optimizer_type, 
                                         loss,
                                         self.learning_rate, 
                                         clip_grad = 5)
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
            if self.tfrecords_mode == 'pair':
                size = self.num_pair
                num_classes_per_batch = 2
                num_sentences_per_class = self.batch_size // num_classes_per_batch
            else:
                size = self.num_class
                num_classes_per_batch = 16
                num_sentences_per_class = self.batch_size // num_classes_per_batch

            filenames = ["{}/train_class_{:04d}".format(self.tfrecords_path,i) \
                             for i in range(size)]
            logging.info("tfrecords train class num: {}".format(len(filenames)))
            datasets = [tf.data.TFRecordDataset(filename) for filename in filenames]
            datasets = [dataset.repeat() for dataset in datasets]
            #assert self.batch_size == num_sentences_per_class* num_classes_per_batch
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
            #test
            #sess = tf.Session()
            #features,label = sess.run([features,label])
            #features['x_query_pred'] = [item.decode('utf-8') for item in
            #                           features['x_query_pred'][1]]
            return features, label

        def test_input_fn(mode):
            filenames = ["{}/{}_class_{:04d}".format(self.tfrecords_path,mode,i) \
                             for i in range(self.num_class)]
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
            'keep_prob': 0.5
        }
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(),
                                           config = config,
                                           params = params)
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
            features.update(self.encoder.features)
            return tf.estimator.export.ServingInputReceiver(features, features)

        estimator.export_savedmodel(
            self.export_dir_path, # 目录
            serving_input_receiver_fn, # 返回ServingInputReceiver的函数
            assets_extra=None,
            as_text=False,
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
        predictions_vec = [item['pred'] for item in predictions]
        predictions_label = [item['label'] for item in predictions]
        if self.tfrecords_mode == 'class':
            refers = estimator.predict(input_fn=self.create_input_fn("label"))
            refers = list(refers) 

            refers_vec = [item['pred'] for item in refers]
            refers_label = [item['label'] for item in refers]

            right = 0
            thre_right = 0
            sum = 0
            scores = cosine_similarity(predictions_vec, refers_vec)
            max_id = np.argmax(scores, axis=-1)
            #max_id = self.knn(scores, predictions_label, refers_label)
            for idx, item in enumerate(max_id):
                if refers_label[item] == predictions_label[idx]:
                    if scores[idx][item] > self.score_thre:
                        thre_right += 1
                    right += 1
                sum += 1
            print("Acc:{}".format(float(right)/sum))
            print("ThreAcc:{}".format(float(thre_right)/sum))
        else:
            #TODO: 对于pair方式的评估
            pdb.set_trace()

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
