import tensorflow as tf
from tensorflow.contrib import predictor
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


class Classify(object):
    def __init__(self, conf):
        self.task_type = 'classify'
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.graph = tf.get_default_graph()
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
                        self.tfrecords_path, self.label_path)

    def cal_loss(self, pred, labels, batch_size, conf):
        loss = get_loss(type = self.loss_type, logits = pred, labels =
                                labels, labels_sparse = True, **conf)
        return loss

    def create_model_fn(self):
        def model_fn(features, labels, mode, params):
            ########### embedding #################
            if not self.use_language_model:
                self.init_embedding()
                self.embed_query = self.embedding(features = features, name = 'x_query')
            else:
                self.embedding = None
            #############  encoder  #################
            #model params
            self.encoder.keep_prob = params['keep_prob']
            self.encoder.is_training = params['is_training']
            global_step = tf.train.get_or_create_global_step()
            if not self.use_language_model:
                pred = self.encoder(self.embed_query, 
                                    name = 'x_query',
                                    features = features)
            else:
                pred = self.encoder(features = features)
            pred = tf.nn.softmax(pred)
            ############### predict ##################
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': pred,
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            ############### loss ##################
            loss = self.cal_loss(pred,
                             labels,
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
        scores = [item['pred'] for item in predictions]
        labels = [item['label'] for item in predictions]
        max_scores = np.max(scores, axis = -1)
        max_ids = np.argmax(scores, axis = -1)
        res = np.equal(labels, max_ids)
        right = len(list(filter(lambda x:x == True, res)))
        sum = len(res)
        print("Acc:{}".format(float(right)/sum))

    def test_unit(self, text):
        if self.model_loaded == False:
            self.init_embedding()
            subdirs = [x for x in Path(self.export_dir_path).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
            self.mp_label = pickle.load(open(self.label_path, 'rb'))
            self.mp_label_rev = {self.mp_label[item]:item for item in self.mp_label}
            self.model_loaded = True
        text_list  = [text]
        text_list_pred, x_query, x_query_length = self.embedding.text2id(text_list,
                                                     self.vocab_dict,
                                                     need_preprocess = True)
        label = [0 for _ in range(len(text_list))]

        predictions = self.predict_fn({'x_query': x_query, 
                                  'x_query_length': x_query_length, 
                                  'label': label})
        scores = [item for item in predictions['pred']]
        max_scores = np.max(scores, axis = -1)
        max_ids = np.argmax(scores, axis = -1)
        ret =  (max_ids[0], max_scores[0], self.mp_label_rev[max_ids[0]])
        print(ret)
        return ret

