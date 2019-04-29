import sys,os
import yaml
import time
import copy
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.preprocess import Preprocess
from embedding import embedding
from encoder import encoder
from utils.data_utils import *
from utils.tf_utils import load_pb,write_pb
import pdb

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


class NER(object):
    def __init__(self, conf):
        self.conf = conf
        self.task_type = 'NER'

        self.batch_size = self.conf['batch_size']
        self.num_class = self.conf['num_tag']
        self.learning_rate = self.conf['learning_rate']

        self.embedding_type = self.conf['embedding']
        self.encoder_type = self.conf['encoder']
        self.epoch_num = self.conf['num_epochs']
        self.tag2label = tag2label
        self.shuffle = True
        self.CRF = True

        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        self.pre = Preprocess()
        self.text_list, self.label_list = load_ner_data(self.conf['train_path'])
        self.trans_label_list(self.label_list, self.tag2label)

        self.text_list = [self.pre.get_dl_input_by_text(text) for text in self.text_list]

        #build vocabulary map using training data
        self.vocab_dict = embedding[self.embedding_type].build_dict(dict_path = self.conf['dict_path'], 
                                                              text_list = self.text_list)

        #define embedding object by embedding_type
        self.embedding = embedding[self.embedding_type](text_list = self.text_list,
                                                        vocab_dict = self.vocab_dict,
                                                        dict_path = self.conf['dict_path'],
                                                        random=self.conf['rand_embedding'],
                                                        batch_size = self.conf['batch_size'])
        self.embed = self.embedding('x')
        #self.y = tf.placeholder(tf.int32, [None], name="y")
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        #self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        #self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        #model params
        params = {
            "maxlen":self.embedding.maxlen,
            "embedding_size":self.embedding.size,
            "keep_prob":self.keep_prob,
            "is_training": self.is_training,
            "batch_size": self.batch_size,
            "num_output": self.num_class
        }
        params.update(conf)
        self.encoder = encoder[self.encoder_type](**params)
        self.out = self.encoder(self.embed, 'query', ner_flag = True)
        self.output_nodes = self.out.name.split(':')[0]
        self.loss(self.out)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def loss(self, out):
        out_shape = tf.shape(out)
        self.logits = tf.reshape(out, [-1, out_shape[1], self.num_class])
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)


#    def add_summary(self, sess):
#        """
#        :param sess:
#        :return:
#        """
#        self.merged = tf.summary.merge_all()
#        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def trans_label_list(self, label_list, tag2label):
        for idx,labels in enumerate(label_list):
            for idy,label in enumerate(labels):
                label_list[idx][idy] = tag2label[label_list[idx][idy]]

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        label_list = []
        batches = batch_iter(sent, self.batch_size, self.epoch_num, shuffle=False)
        for batch in batches:
            seqs, labels = zip(*batch)
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def train(self):
        train_data = zip(self.text_list, self.label_list)
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_iter(train_data, self.batch_size, self.epoch_num, shuffle = True)

        for step, batch in enumerate(batches):
            x_batch, labels = zip(*batch)
            sys.stdout.write(' processing: {}.'.format(step + 1) + '\r')
            step_num = step + 1

            _, x_batch, len_batch = self.embedding.text2id(x_batch, self.vocab_dict, need_preprocess = False)
            feed_dict = {self.sequence_lengths: len_batch}
            feed_dict[self.labels],_ = self.embedding.pad_sequences(labels)
            feed_dict.update(self.embedding.feed_dict(x_batch,'x'))
            feed_dict.update(self.encoder.feed_dict(query = len_batch))

            loss_train, step_num_ = self.sess.run([self.loss, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0:
                print('{} , step {}, loss: {:.4}, global_step: {}'.format(\
                    start_time, 
                    step + 1,
                    loss_train, 
                    step_num))
        print('===========validation / test===========')



        self.dev_text_list, self.dev_label_list = load_ner_data(self.conf['test_path'])
        self.dev_text_list = [self.pre.get_dl_input_by_text(text) for text in self.dev_text_list]
        dev_data = zip(self.dev_text_list, self.dev_label_list)
        self.trans_label_list(self.dev_label_list, self.tag2label)
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(self.sess, dev_data)
        self.evaluate(label_list_dev, seq_len_list_dev, dev_data)


    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        batches = batch_iter(dev, self.batch_size, self.epoch_num, shuffle=False)
        for batch in batches:
            seqs, labels = zip(*batch)
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        _, x_batch, len_batch = self.embedding.text2id(seqs, self.vocab_dict, need_preprocess = False)
        feed_dict = {self.sequence_lengths: len_batch}
        feed_dict.update(self.embedding.feed_dict(x_batch,'x'))
        feed_dict.update(self.encoder.feed_dict(query = len_batch))
        #feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            print(_)

