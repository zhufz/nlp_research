import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pdb
import logging
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


class Match(object):
    def __init__(self, conf):
        self.task_type = 'match'
        self.conf = conf
        for attr in conf:
            setattr(self, attr, conf[attr])
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        self.pre = Preprocess()
        self.generator = PairGenerator(self.relation_path,\
                                       self.index_path,
                                       self.test_path)
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in \
                          self.generator.index_data]
        self.label_list = self.generator.label_data
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

        self.gt = GenerateTfrecords()
        self.gt.process(self.text_list, self.label_list, self.embedding.text2id, self.vocab_dict, self.conf)

            #_, x1_batch, x1_len_batch = self.embedding.text2id(batch,
            #                                         self.vocab_dict,
            #                                         need_preprocess = False)
    def create_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            if not self.use_language_model:

                #define embedding object by embedding_type

                self.embed_query = self.embedding(features = features, name = 'x_query')
                self.embed_sample = self.embedding(features = features, name = 'x_sample')
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
                "keep_prob":self.keep_prob,
                "is_training": self.is_training,
                "batch_size": self.batch_size,
                "num_output": self.num_output
            })

            self.pred = self.sim(self.sim_mode, params) #encoder
            self.output_nodes = self.pred.name.split(':')[0]
            self.pos_target = tf.ones(shape = [int(self.batch_size/2)], dtype = tf.float32)
            self.neg_target = tf.zeros(shape = [int(self.batch_size/2)], dtype = tf.float32)
            self.loss = self.loss(self.loss_type,
                                  self.pred,
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
        return model_fn


    def create_input_fn(self):
        filenames = ["{}/class_{:04d}".format(self.tfrecords_path,i) \
                         for i in range(self.num_class)]
        datasets = tf.data.TFRecordDataset(filenames)

        def parser(record):
            keys_to_features = {
                "input": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.string),
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            # Perform additional preprocessing on the parsed data.
            input = tf.decode_raw(parsed["input"], tf.int64)
            label = tf.decode_raw(parsed["label"], tf.int64)
            return {'input':input, 'label':label} , label

        def input_fn():
            num_classes_per_batch = 8
            num_sentences_per_class = 8
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

            batch_size = num_classes_per_batch * num_images_per_class
            dataset = dataset.map(parser)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            features, label = iterator.get_next() return features, label return input_fn

    def train(self):
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.model_path)
        estimator = tf.estimator.Estimator(self.create_model_fn(), config=config)
        estimator.train(self.create_input_fn())

    def loss(self, loss_type, pred, pos_target, neg_target, batch_size, conf):
        pos = tf.strided_slice(pred, [0], [batch_size], [2])
        neg = tf.strided_slice(pred, [1], [batch_size], [2])
        if loss_type == 'hinge_loss':
            loss = get_loss(type = loss_type, neg_logits = neg, pos_logits =
                                pos, **conf)
        else:
            pos_loss = get_loss(type = loss_type, logits = pos, labels =
                                pos_target, **conf)

            neg_loss = get_loss(type = loss_type, logits = neg, labels =
                                neg_target, **conf)
            loss = pos_loss + neg_loss
        return loss

    def sim(self, mode, params):
        if mode == 'cross':
            return self.cross_sim(params)
        elif mode == 'represent':
            return self.represent_sim(params)
        else:
            raise ValueError('unknown sim mode')

    def cross_sim(self, params):
        #cross based match model
        self.encoder = encoder[self.encoder_type](**params)
        if not self.use_language_model:
            pred = self.encoder(x_query = self.embed_query, x_sample = self.embed_sample)
        else:
            pred = self.encoder()
        return pred

    def represent_sim(self, params):
        #representation based match model
        self.encoder = encoder[self.encoder_type](**params)
        self.encode_query = self.encoder(self.embed_query, 'x_query')
        self.encode_sample = self.encoder(self.embed_sample, 'x_sample')
        pred = self.cosine_similarity(self.encode_query, self.encode_sample)
        self.rep_nodes = self.encode_query.name.split(':')[0]
        return pred

    def cosine_similarity(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        return score

