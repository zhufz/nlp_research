import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pdb
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
from common.triplet_loss import batch_hard_triplet_loss


class Match(object):
    def __init__(self, conf):
        self.task_type = 'match'
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.graph = tf.get_default_graph()
        self.pre = Preprocess()
        self.generator = PairGenerator(self.relation_path,\
                                       self.index_path,
                                       self.test_path)
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in \
                          self.generator.index_data]
        self.label_list = self.generator.label_data

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
        self.gt = GenerateTfrecords()
        self.gt.process(self.text_list, self.label_list, self.embedding.text2id,
                        self.vocab_dict, self.tfrecords_path, self.label_path)
    def train(self):
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.checkpoint_path)
        estimator = tf.estimator.Estimator(model_fn = self.create_model_fn(), config=config)
        estimator.train(input_fn = self.create_input_fn("train"), max_steps = 3000)
        predictions = estimator.predict(input_fn=self.create_input_fn("test"))
        predictions = list(predictions)
        predictions_vec = [item['pred'] for item in predictions]
        predictions_label = [item['label'] for item in predictions]

        refers = estimator.predict(input_fn=self.create_input_fn("label"))
        refers = list(refers) 

        refers_vec = [item['pred'] for item in refers]
        refers_label = [item['label'] for item in refers]

        mp_label = pickle.load(open(self.label_path,'rb'))
        mp_label_rev = {mp_label[item]:item for item in mp_label}

        right = 0
        sum = 0
        scores = cosine_similarity(predictions_vec, refers_vec)
        max_id = np.argmax(scores, axis=-1)
        #max_id = self.knn(scores, predictions_label, refers_label)
        for idx, item in enumerate(max_id):
            if refers_label[item] == predictions_label[idx]:
                right += 1
            sum += 1
        print("Acc:{}".format(float(right)/sum))

    def test(self):
        pass

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

    def create_model_fn(self):
        def model_fn(features, labels, mode, params):
            self.init_embedding()
            if not self.use_language_model:
                self.embed_query = self.embedding(features = features, name = 'x_query')
            else:
                self.embedding = None
            #model params
            params = self.conf
            params.update({
                "maxlen":self.maxlen,
                "maxlen1":self.maxlen,
                "maxlen2":self.maxlen,
                "num_class":self.num_class,
                "embedding_size":self.embedding_size,
                "keep_prob":features['keep_prob'],
                "is_training": features['is_training'],
                "batch_size": self.batch_size,
                "num_output": self.num_output
            })

            self.global_step = tf.train.get_or_create_global_step()
            self.pred = self.sim(self.sim_mode, params, features) #encoder
            self.pos_target = tf.ones(shape = [int(self.batch_size/2)], dtype = tf.float32)
            self.neg_target = tf.zeros(shape = [int(self.batch_size/2)], dtype = tf.float32)

            if mode == tf.estimator.ModeKeys.TRAIN:
                self.loss = self.cal_loss(self.loss_type,
                                      self.pred,
                                      labels,
                                      self.pos_target,
                                      self.neg_target,
                                      self.batch_size,
                                      self.conf)
                if self.use_clr:
                    self.learning_rate = cyclic_learning_rate(global_step=self.global_step,
                                                          learning_rate = self.learning_rate, 
                                                          mode = self.clr_mode)
                self.optimizer = get_train_op(self.global_step, 
                                               self.optimizer_type, 
                                               self.loss,
                                               self.learning_rate, 
                                               clip_grad = 5)
                return tf.estimator.EstimatorSpec(mode, loss=self.loss,
                                                      train_op=self.optimizer)
            # Add evaluation metrics (for EVAL mode)
            #if mode == tf.estimator.ModeKeys.EVAL:
                #eval_metric_ops = {
                #    "accuracy": tf.metrics.accuracy(
                #        labels=labels, predictions=predictions["classes"])}
                #return tf.estimator.EstimatorSpec(
                #    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
            #predicted_classes = tf.argmax(logits, 1)
            elif mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': self.pred,
                    'label': features['label']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        return model_fn

    def create_input_fn(self, mode):
        n_cpu = multiprocessing.cpu_count()
        def parse_record(record, keep_prob, is_training):
            keys_to_features = {
                "input": tf.FixedLenFeature([self.maxlen], tf.int64),
                "length": tf.FixedLenFeature([1], tf.int64),
                "label": tf.FixedLenFeature([1], tf.int64),
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            # Perform additional preprocessing on the parsed data.
            input = tf.reshape(parsed['input'], [self.maxlen])
            label = tf.reshape(parsed['label'], [1])
            length = tf.reshape(parsed['length'], [1])
            return {'x_query':input, 
                    'length':length[0], 
                    'keep_prob': keep_prob, 
                    'is_training': is_training, 
                    'label': label[0]} , label[0]

        def train_input_fn():
            filenames = ["{}/train_class_{:04d}".format(self.tfrecords_path,i) \
                             for i in range(self.num_class)]
            assert self.num_class == len(filenames), "the num of tfrecords file error!"
            logging.info("tfrecords train class num: {}".format(len(filenames)))
            datasets = [tf.data.TFRecordDataset(filename) for filename in filenames]
            datasets = [dataset.repeat() for dataset in datasets]

            num_classes_per_batch = 8
            num_sentences_per_class = 8
            batch_size = num_classes_per_batch * num_sentences_per_class
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
            dataset = dataset.map(lambda record: parse_record(record, 0.5, True),
                                  num_parallel_calls=n_cpu)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(4*batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next()
            return features, label

        def test_input_fn(mode):
            filenames = ["{}/{}_class_{:04d}".format(self.tfrecords_path,mode,i) \
                             for i in range(self.num_class)]
            assert self.num_class == len(filenames), "the num of tfrecords file error!"
            logging.info("tfrecords test class num: {}".format(len(filenames)))
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(lambda record: parse_record(record, 1, False),
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


    def cal_loss(self, loss_type, pred, labels, pos_target, neg_target, batch_size, conf):
        if loss_type == 'hinge_loss':
            loss = batch_hard_triplet_loss(labels, pred, conf['margin'])
        else:
            pos = tf.strided_slice(pred, [0], [batch_size], [2])
            neg = tf.strided_slice(pred, [1], [batch_size], [2])
            pos_loss = get_loss(type = loss_type, logits = pos, labels =
                                pos_target, **conf)

            neg_loss = get_loss(type = loss_type, logits = neg, labels =
                                neg_target, **conf)
            loss = pos_loss + neg_loss
        return loss

    def sim(self, mode, params, features):
        if mode == 'cross':
            return self.cross_sim(params, features)
        elif mode == 'represent':
            return self.represent_sim(params, features)
        else:
            raise ValueError('unknown sim mode')

    def cross_sim(self, params, features):
        #cross based match model
        self.encoder = encoder[self.encoder_type](**params)
        if not self.use_language_model:
            pred = self.encoder(x_query = self.embed_query, 
                                x_sample = self.embed_sample,
                                features = features)
        else:
            pred = self.encoder()
        return pred

    def represent_sim(self, params, features):
        #representation based match model
        self.encoder = encoder[self.encoder_type](**params)
        features['x_query_length'] = features['length']
        self.encode_query = self.encoder(self.embed_query, 
                                         'x_query', 
                                         features = features)
        return self.encode_query
