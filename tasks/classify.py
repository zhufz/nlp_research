import sys,os
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.preprocess import Preprocess
from embedding import embedding
from encoder import encoder
from utils.data_utils import *
from utils.tf_utils import load_pb,write_pb
from tensorflow.python.platform import gfile

import pdb

class Classify(object):
    def __init__(self, conf):
        self.conf = conf
        self.task_type = 'classify'

        self.batch_size = self.conf['batch_size']
        self.num_class = self.conf['num_class']
        self.learning_rate = self.conf['learning_rate']

        self.embedding_type = self.conf['embedding']
        self.encoder_type = self.conf['encoder']

        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        self.pre = Preprocess()
        self.text_list, self.label_list = load_classify_data(self.conf['train_path'])
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
        self.y = tf.placeholder(tf.int32, [None], name="y")

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
        self.out = self.encoder(self.embed)
        self.output_nodes = self.out.name.split(':')[0]
        self.loss(self.out)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def load_data(self, mode = 'train'):
        print("Building dataset...")
        if mode == 'train':
            class_mp, class_mp_rev = generate_class_mp(self.label_list, self.conf['classes_path'])
            y = [class_mp[item] for item in self.label_list]
            train_x, valid_x, train_y, valid_y = \
                train_test_split(self.text_list, y, test_size=0.05)
            return zip(train_x, train_y), zip(valid_x, valid_y)
        else:

            class_mp, class_mp_rev = load_class_mp(self.conf['classes_path'])
            text_list, label_list = load_classify_data(self.conf['test_path'])
            y = [class_mp[item] for item in label_list]
            return text_list, y

    def loss(self, out):
        with tf.name_scope("output"):
            self.predictions = tf.argmax(tf.nn.softmax(out, axis=1, name="scores"), -1, output_type=tf.int32,
                                         name = 'predictions')
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def get_batch_len(self, x_batch):
        return [len(x) for x in x_batch]

    def train(self):
        self.train_data, self.valid_data = self.load_data(mode = 'train')
        self.train_data = list(self.train_data)
        self.valid_data = list(self.valid_data)
        train_batches = batch_iter(self.train_data, self.batch_size, self.conf['num_epochs'])
        num_batches_per_epoch = (len(self.train_data) - 1) // self.batch_size + 1
        max_accuracy = 0
        for batch in train_batches:
            x_batch, y_batch = zip(*batch)
            len_batch = self.get_batch_len(x_batch)
            _, x_batch = self.embedding.text2id(x_batch, self.vocab_dict, need_preprocess = False)
            train_feed_dict = {
                self.y: y_batch,
                self.is_training: True
            }
            train_feed_dict.update(self.embedding.feed_dict(x_batch,'x'))
            train_feed_dict.update(self.encoder.feed_dict(len = len_batch))

            _, step, loss = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict=train_feed_dict)
            if step % 100 == 0:
                print("step {0}: loss = {1}".format(step, loss))
            if step % 1000 == 0:
                # Test accuracy with validation data for each epoch.
                valid_batches = batch_iter(self.valid_data, self.batch_size, 1, shuffle=False)
                sum_accuracy, cnt = 0, 0
                for valid_batch in valid_batches:

                    valid_x_batch, valid_y_batch = zip(*valid_batch)
                    len_batch = self.get_batch_len(valid_x_batch)
                    _, valid_x_batch = self.embedding.text2id(valid_x_batch, self.vocab_dict, need_preprocess = False)
                    valid_feed_dict = {
                        self.y: valid_y_batch,
                        self.is_training: False
                    }
                    valid_feed_dict.update(self.embedding.feed_dict(valid_x_batch,'x'))
                    valid_feed_dict.update(self.encoder.feed_dict(len = len_batch))
                    accuracy = self.sess.run(self.accuracy, feed_dict=valid_feed_dict)
                    sum_accuracy += accuracy
                    cnt += 1
                valid_accuracy = sum_accuracy / cnt
                print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))
                # Save model
                if valid_accuracy > max_accuracy:
                    max_accuracy = valid_accuracy
                    self.saver.save(self.sess,
                                    #"{0}/{1}/{2}.ckpt".format(self.conf['path'],
                                    "{0}/{1}.ckpt".format(self.conf['checkpoint_path'],
                                                              #self.task_type,
                                                              self.task_type),
                                    global_step=step)
                    write_pb(self.conf['checkpoint_path'],
                             self.conf['model_path'],
                             ['is_training','output/predictions','accuracy/accuracy',self.output_nodes])
                    print("Model is saved.\n")
                else:
                    print(f"train finished! accuracy: {max_accuracy}")
                    #sys.exit(0)

    def test(self):
        graph = load_pb(self.conf['model_path'])
        sess = tf.Session(graph=graph)

        self.y = graph.get_operation_by_name("y").outputs[0]
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]
        self.accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        #self.scores = graph.get_tensor_by_name("output/scores:0")
        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")
        self.predictions = graph.get_tensor_by_name("output/predictions:0")

        mp, mp_rev = load_class_mp(self.conf['classes_path'])

        test_x, test_y = self.load_data("test")
        pred_y = []
        scores = []
        batches = batch_iter(zip(test_x, test_y), self.conf['batch_size'], 1, shuffle=False)
        sum_accuracy, cnt = 0, 0
        right, all = 0, 0
        vocab_dict = embedding[self.embedding_type].build_dict(self.conf['dict_path'],
                                                      mode = 'test')
        all_test_x = []
        all_test_y = []
        for batch in batches:
            batch_x, batch_y = zip(*batch)
            len_batch = self.get_batch_len(batch_x)
            preprocess_x, batch_x_id = self.embedding.text2id(batch_x, vocab_dict, need_preprocess = True)
            feed_dict = {
                self.y: batch_y,
                self.is_training: False
            }
            feed_dict.update(self.embedding.pb_feed_dict(graph, batch_x_id, 'x'))
            feed_dict.update(self.encoder.pb_feed_dict(graph, len = len_batch))
            accuracy_out, predictions_out, scores_out = sess.run([self.accuracy,
                                                                  self.predictions,
                                                                  self.scores],
                                                                 feed_dict=feed_dict)
            max_scores = [scores_out[idx][predictions_out[idx]] \
                          for idx in range(len(predictions_out))]
            sum_accuracy += accuracy_out
            cnt += 1
            pred_y += list(predictions_out)
            scores += list(max_scores)
            all_test_x += list(batch_x)
            all_test_y += list(batch_y)

            for idx in range(len(predictions_out)):
                if predictions_out[idx] == int(batch_y[idx]) and max_scores[idx]> 0.4:
                    right += 1
                all += 1
        dt = pd.DataFrame({'text': all_test_x,
                       'target': [mp_rev[int(item)] for item in
                                 all_test_y] ,
                       'pred': [mp_rev[item] for item in
                                pred_y],
                       'score': scores })
        dt.to_csv(self.conf['test_path']+'.result.csv',index=False,sep=',')
        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
        print("Test Thre Accuracy : {0}".format(right / all))

    def predict(self):
        predict_file = self.conf['predict_path']
        graph = load_pb(self.conf['model_path'])
        sess = tf.Session(graph=graph)

        self.y = graph.get_operation_by_name("y").outputs[0]
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]

        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")
        #self.scores = graph.get_tensor_by_name("output/scores:0")
        self.predictions = graph.get_tensor_by_name("output/predictions:0")

        vocab_dict = embedding[self.embedding_type].build_dict(self.conf['dict_path'],mode = 'test')
        mp, mp_rev = load_class_mp(self.conf['classes_path']) 
        with open(predict_file) as f:
            lines = [line.strip() for line in f.readlines()]
            batches = batch_iter(lines, self.conf['batch_size'], 1, shuffle=False)
            scores = []
            predicts = []
            for batch_x in batches:
                len_batch = self.get_batch_len(batch_x)
                feed_dict = {
                    self.is_training: False
                }
                preprocess_x, batch_x = self.embedding.text2id(batch_x, vocab_dict)
                feed_dict.update(self.embedding.pb_feed_dict(graph, batch_x, 'x'))
                feed_dict.update(self.encoder.pb_feed_dict(graph, len = len_batch))
                predictions_out, scores_out = sess.run([self.predictions,
                                                            self.scores],
                                                            feed_dict=feed_dict)
                max_scores = [scores_out[idx][predictions_out[idx]] \
                              for idx in range(len(predictions_out))]

                predicts += list(predictions_out)
                scores += list(max_scores)

            predicts = [mp_rev[item] for item in predicts]

            dt = pd.DataFrame({'text': lines,
                               'pred': predicts,
                               'score': scores })
            dt.to_csv(self.conf['predict_path']+'.result.csv',index=False,sep=',')

    def test_unit(self, text):
        graph = load_pb(self.conf['model_path'])
        sess = tf.Session(graph=graph)

        self.y = graph.get_operation_by_name("y").outputs[0]
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]

        #self.scores = graph.get_tensor_by_name("output/scores:0")
        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")
        self.predictions = graph.get_tensor_by_name("output/predictions:0")

        vocab_dict = embedding[self.embedding_type].build_dict(self.conf['dict_path'],mode = 'test')
        mp, mp_rev = load_class_mp(self.conf['classes_path']) 
        batches = batch_iter([text], self.conf['batch_size'], 1, shuffle=False)
        for batch_x in batches:
            len_batch = self.get_batch_len(batch_x)
            feed_dict = {
                self.is_training: False
            }
            preprocess_x, batch_x = self.embedding.text2id(batch_x, vocab_dict)
            feed_dict.update(self.embedding.pb_feed_dict(graph, batch_x, 'x'))
            feed_dict.update(self.encoder.pb_feed_dict(graph, len = len_batch))
            predictions_out, scores_out = sess.run([self.predictions,
                                                        self.scores],
                                                        feed_dict=feed_dict)
            max_scores = [scores_out[idx][predictions_out[idx]] \
                          for idx in range(len(predictions_out))]
        print("preprocess: {}".format(preprocess_x))
        print("class:{}, score:{}, class_id:{}".format(
            mp_rev[predictions_out[0]],
            max_scores[0],
            predictions_out[0]))
        return mp_rev[predictions_out[0]], max_scores[0]
