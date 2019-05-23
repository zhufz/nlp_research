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
from language_model.bert.modeling import get_assignment_map_from_checkpoint
from utils.data_utils import *
from utils.preprocess import Preprocess
from utils.tf_utils import load_pb,write_pb
from utils.recall import Annoy
from common.layers import get_train_op
from common.loss import get_loss
from common.lr import cyclic_learning_rate
from common.triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss


class Match(object):
    def __init__(self, conf):
        self.task_type = 'match'
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.graph = tf.get_default_graph()
        self.pre = Preprocess()
        self.model_loaded = False
        self.zdy = {}
        self.generator = PairGenerator(self.relation_path,\
                                       self.index_path,
                                       self.test_path)
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in \
                          self.generator.index_data]
        self.label_list = self.generator.label_data
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
                        self.vocab_dict, self.tfrecords_path, self.label_path)

    def cal_loss(self, loss_type, pred, labels, pos_target, neg_target, batch_size, conf):
        if loss_type == 'hinge_loss':
            loss = batch_hard_triplet_loss(labels, pred, conf['margin'])
            #loss = batch_all_triplet_loss(labels, pred, conf['margin'])
        else:
            pos = tf.strided_slice(pred, [0], [batch_size], [2])
            neg = tf.strided_slice(pred, [1], [batch_size], [2])
            pos_loss = get_loss(type = loss_type, logits = pos, labels =
                                pos_target, **conf)

            neg_loss = get_loss(type = loss_type, logits = neg, labels =
                                neg_target, **conf)
            loss = pos_loss + neg_loss
        return loss

    def create_model_fn(self):
        def model_fn(features, labels, mode, params):
            if not self.use_language_model:
                self.init_embedding()
                if self.tfrecords_mode == 'class':
                    self.embed_query = self.embedding(features = features, name = 'x_query')
                else:
                    self.embed_query = self.embedding(features = features, name = 'x_query')
                    self.embed_sample = self.embedding(features = features, name = 'x_sample')
            else:
                self.embedding = None
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
                    pred = self.encoder()
            elif self.sim_mode == 'represent':
                features['x_query_length'] = features['length']
                pred = self.encoder(self.embed_query, 
                                                 name = 'x_query', 
                                                 features = features)
            else:
                raise ValueError('unknown sim mode')

            pos_target = tf.ones(shape = [int(self.batch_size/2)], dtype = tf.float32)
            neg_target = tf.zeros(shape = [int(self.batch_size/2)], dtype = tf.float32)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': pred,
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            loss = self.cal_loss(self.loss_type,
                             pred,
                             labels,
                             pos_target,
                             neg_target,
                             self.batch_size,
                             self.conf)
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
            else:
                size = self.num_class
            filenames = ["{}/train_class_{:04d}".format(self.tfrecords_path,i) \
                             for i in range(size)]
            logging.info("tfrecords train class num: {}".format(len(filenames)))
            if self.tfrecords_mode == 'pair':
                dataset = tf.data.TFRecordDataset(filenames)
            else:

                datasets = [tf.data.TFRecordDataset(filename) for filename in filenames]
                datasets = [dataset.repeat() for dataset in datasets]

                num_sentences_per_class = 4
                num_classes_per_batch = 16
                assert self.batch_size == num_sentences_per_class* num_classes_per_batch
                def generator():
                    while True:
                        labels = np.random.choice(range(self.num_class),
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

            sess = tf.Session()
            if 'x_query_length' in features:
                features['x_query_length'] = features['x_query_length'].eval(session = sess)
            if 'x_sample_length' in features:
                features['x_sample_length'] = features['x_sample_length'].eval(session = sess)
            if 'x_query_raw' in features:
                features['x_query_raw'] = features['x_query_raw'].eval(session = sess)
                features['x_query_raw'] = [item.decode('utf-8') for item in
                                           features['x_query_raw'][1]]
            if 'x_sample_raw' in features:
                features['x_sample_raw'] = features['x_sample_raw'].eval(session = sess)
                features['x_sample_raw'] = [item.decode('utf-8') for item in
                                            features['x_sample_raw'][1]]
            try:
                self.encoder.update_features(features)
            except:
                logging.info(traceback.print_exc())
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
        estimator.train(input_fn = self.create_input_fn("train"), max_steps = 3000)
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
            x_query = tf.placeholder(dtype=tf.int64, shape=[None, self.maxlen],
                                   name='x_query')
            length = tf.placeholder(dtype=tf.int64, shape=[None], name='length')
            label = tf.placeholder(dtype=tf.int64, shape=[None], name='label')

            receiver_tensors = {'x_query': x_query, 'length': length, 'label': label}
            features = {'x_query': x_query, 'length': length, 'label': label}
            return tf.estimator.export.ServingInputReceiver(receiver_tensors,
                                                            features)
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
            sum = 0
            scores = cosine_similarity(predictions_vec, refers_vec)
            max_id = np.argmax(scores, axis=-1)
            #max_id = self.knn(scores, predictions_label, refers_label)
            for idx, item in enumerate(max_id):
                if refers_label[item] == predictions_label[idx]:
                    if scores[idx][item] > self.score_thre:
                        right += 1
                sum += 1
            print("Acc:{}".format(float(right)/sum))
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

    def test_unit(self, text):
        #######################init#########################
        if self.model_loaded == False:
            subdirs = [x for x in Path(self.export_dir_path).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
            self.init_embedding()
            self.model_loaded = True
            self.vec_list = self._get_vecs(self.predict_fn, self.text_list)
            #self.set_zdy_labels(['睡觉','我回家了','晚安','娃娃了','周杰伦','自然语言处理'],
            #                    ['打开情景模式','打开情景模式','打开情景模式',
            #                     '打开情景模式','打开情景模式','打开情景模式'])
        text_list = self.text_list
        vec_list = self.vec_list
        label_list = self.label_list

        #用于添加自定义问句(自定义优先)
        if self.zdy != {}:
            text_list = self.zdy['text_list'] + text_list
            vec_list = np.concatenate([self.zdy['vec_list'], self.vec_list], axis = 0)
            label_list = self.zdy['label_list'] + label_list
        vec = self._get_vecs(self.predict_fn, [text], need_preprocess = True)
        scores = cosine_similarity(vec, vec_list)[0]
        max_id = np.argmax(scores)
        max_score = scores[max_id]
        max_similar = text_list[max_id]
        print(label_list[max_id], max_score, max_similar)
        return label_list[max_id], max_score

    def set_zdy_labels(self, text_list, label_list):
        self.zdy['text_list'] = text_list
        self.zdy['vec_list'] = self._get_vecs(self.predict_fn, 
                                              text_list,
                                              need_preprocess = True)
        self.zdy['label_list'] = label_list

    def _get_vecs(self, predict_fn, text_list, need_preprocess = False):
        #根据batches数据生成向量
        text_list_pred, x_query, x_query_length = self.embedding.text2id(text_list,
                                                     self.vocab_dict,
                                                     need_preprocess)
        label = [0 for _ in range(len(text_list))]

        predictions = predict_fn({'x_query': x_query, 
                                  'length': x_query_length, 
                                  'label': label})
        return predictions['pred']
