import tensorflow as tf
from embedding import embedding
from encoder import encoder
import pdb
import os,sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)

from utils.data_utils import *
from utils.preprocess import Preprocess
from utils.tf_utils import load_pb,write_pb


class Match(object):
    def __init__(self, conf):
        self.task_type = 'match_cross'
        self.conf = conf
        self.learning_rate =conf['learning_rate']
        self.embedding_type = conf['embedding']
        self.encoder_type = conf['encoder']
        self.batch_size = conf['batch_size']
        self.margin = conf['margin']
        self.num_class = conf['num_class']
        self.score_thre = conf['score_thre']
        self.loss_type = conf['loss_type']
        self.num_output = conf['num_output']

        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, float(conf['keep_prob']), 1.0)

        self.pre = Preprocess()
        self.generator = PairGenerator(self.conf['relation_path'],\
                                       self.conf['index_path'],
                                       self.conf['test_path'])
        self.text_list = [self.pre.get_dl_input_by_text(text) for text in \
                          self.generator.index_data]
        self.label_list = self.generator.label_data
        self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                            dict_path = self.conf['dict_path'],
                                            text_list = self.text_list,
                                            mode = self.conf['mode'])

        #define embedding object by embedding_type
        self.embedding = embedding[self.embedding_type](text_list = self.text_list,
                                                        vocab_dict = self.vocab_dict,
                                                        dict_path = self.conf['dict_path'],
                                                        random=self.conf['rand_embedding'],
                                                        batch_size = self.conf['batch_size'])

        self.embed_query = self.embedding('x_query')
        self.embed_sample = self.embedding('x_sample')

        #model params
        params = {
            "maxlen":self.embedding.maxlen,
            "maxlen1":self.embedding.maxlen,
            "maxlen2":self.embedding.maxlen,
            "num_class":self.num_class,
            "embedding_size":self.embedding.size,
            "keep_prob":self.keep_prob,
            "is_training": self.is_training,
            "batch_size": self.batch_size,
            "num_output": self.num_output
        }
        params.update(self.conf)

        self.pred = self.sim(self.conf['sim_mode'], params) #encoder
        self.output_nodes = self.pred.name.split(':')[0]
        self.pos_target = tf.ones(shape = [int(self.batch_size/2)], dtype = tf.float32)
        self.neg_target = tf.zeros(shape = [int(self.batch_size/2)], dtype = tf.float32)
        self.loss = self.loss(self.loss_type, self.pred)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("logs/match_log", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def loss(self, loss_type, pred):
        if loss_type == 'pairwise':
            return self.pairwise(pred)
        elif loss_type == 'pointwise':
            return self.pointwise(pred)
        else:
            raise ValueError('unknown loss type')

    def pairwise(self, pred):
        pos = tf.strided_slice(pred, [0], [self.batch_size], [2])
        neg = tf.strided_slice(pred, [1], [self.batch_size], [2])
        loss = tf.reduce_mean(tf.maximum(self.margin + neg - pos, 0.0))
        return loss

    def pointwise(self, pred):
        pos = tf.strided_slice(pred, [0], [self.batch_size], [2])
        neg = tf.strided_slice(pred, [1], [self.batch_size], [2])
        pos_loss = tf.reduce_mean(\
                    tf.nn.sigmoid_cross_entropy_with_logits(labels = self.pos_target,
                                                       logits = pos))
        neg_loss = tf.reduce_mean(\
                    tf.nn.sigmoid_cross_entropy_with_logits(labels = self.neg_target,
                                                       logits = neg))
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
        self.encoder = encoder[self.conf['encoder']](**params)
        pred = self.encoder(x_query = self.embed_query, x_sample = self.embed_sample)
        return pred

    def represent_sim(self, params):
        #representation based match model
        self.encoder = encoder[self.conf['encoder']](**params)
        self.encode_query = self.encoder(self.embed_query, 'x_query')
        self.encode_sample = self.encoder(self.embed_sample, 'x_sample')
        pred = self.cosine_similarity(self.encode_query, self.encode_sample)
        return pred

    def cosine_similarity(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        return score

    def predict_prob(self, data_list):
        x1 = [item[0] for item in data_list]
        x2 = [item[1] for item in data_list]
        _, x1, x1_len = self.embedding.text2id(x1, self.vocab_dict,
                                             need_preprocess = False)
        _, x2, x2_len = self.embedding.text2id(x2, self.vocab_dict,
                                             need_preprocess = False)
        feed_dict = {
            self.is_training: False
        }
        feed_dict.update(self.embedding.feed_dict(x1,'x_query'))
        feed_dict.update(self.embedding.feed_dict(x2,'x_sample'))
        feed_dict.update(self.encoder.feed_dict(x_query = x1_len, x_sample = x2_len))
        pred = self.sess.run([self.pred], feed_dict=feed_dict)
        return pred

    def train(self):
        train_batches = self.generator.get_batch(self.text_list,
                                                 self.batch_size,
                                                 self.conf['num_epochs'],
                                                 self.embedding.maxlen,
                                                 self.embedding.maxlen,
                                                 task = self,
                                                 mode =self.conf['batch_mode'])

        max_accuracy = -1
        for batch in train_batches:
            x1_batch, x2_batch = batch
            _, x1_batch, x1_len_batch = self.embedding.text2id(x1_batch, self.vocab_dict,
                                                 need_preprocess = False)
            _, x2_batch, x2_len_batch = self.embedding.text2id(x2_batch, self.vocab_dict,
                                                 need_preprocess = False)

            train_feed_dict = {
                self.is_training: True
            }
            train_feed_dict.update(self.embedding.feed_dict(x1_batch,'x_query'))
            train_feed_dict.update(self.embedding.feed_dict(x2_batch,'x_sample'))
            train_feed_dict.update(\
                        self.encoder.feed_dict(x_query = x1_len_batch, 
                                               x_sample = x2_len_batch))
            _, step, loss = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict=train_feed_dict)
            #print(loss)

            if step % 100 == 0:
                print("step {0}: loss = {1}".format(step, loss))
            if step % 1000 == 0:
                # Save model
                self.saver.save(self.sess,
                                #"{0}/{1}/{2}.ckpt".format(self.conf['path'],
                                "{0}/{1}.ckpt".format(self.conf['checkpoint_path'],
                                                          #self.task_type,
                                                          self.task_type),
                                global_step=step)
                write_pb(self.conf['checkpoint_path'],self.conf['model_path'],["is_training", self.output_nodes])
                acc = self.test()
                if acc > max_accuracy:
                    max_accuracy = acc
                else:
                    print(f'train finished! accuracy: {acc}')
                    sys.exit(0)

    def knn(self, pred, k = 5):
        sorted_id = np.argsort(-pred,axis=0)
        mp = defaultdict(int)
        for idx in range(k):
            mp[int(sorted_id[idx])] += 1
        max_id = max(mp,key=mp.get)
        return max_id


    def test_step(self, batch, embedding, encoder, vocab_dict,  graph, sess, run_param):
        x1_batch,x2_batch,labels_batch = batch
        _, x1_batch, x1_len_batch = embedding.text2id(x1_batch,
                                             vocab_dict,
                                             need_preprocess = False)
        _, x2_batch, x2_len_batch = embedding.text2id(x2_batch,
                                             vocab_dict,
                                             need_preprocess = False)
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        test_feed_dict = {
            is_training: False
        }
        test_feed_dict.update(embedding.pb_feed_dict(graph,x1_batch,'x_query'))
        test_feed_dict.update(embedding.pb_feed_dict(graph,x2_batch,'x_sample'))
        #test_feed_dict.update(encoder.pb_feed_dict(graph, len = [x1_len_batch,x2_len_batch]))
        test_feed_dict.update(encoder.pb_feed_dict(graph, 
                                                   x_query = x1_len_batch, 
                                                   x_sample = x2_len_batch))
        pred = sess.run(run_param, feed_dict=test_feed_dict)
        return pred, labels_batch

    def test(self):
        # Test accuracy with validation data for each epoch.
        test_batches = self.generator.get_test_batch(self.text_list,
                                                     self.embedding.maxlen,
                                                     self.embedding.maxlen)
        graph = load_pb(self.conf['model_path'])
        sess = tf.Session(graph=graph)
        #self.scores = graph.get_operation_by_name(self.output_nodes)
        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")
        sum, rig, thre_rig = 0, 0, 0
        for batch in test_batches:
            pred, labels = self.test_step(batch, self.embedding, self.encoder, self.vocab_dict,
                           graph, sess, self.scores)
            assert len(pred) == len(labels), "len(pred)!=len(labels)!"
            max_id = self.knn(pred)
            max_score = pred[max_id]
            sum += 1
            if labels[max_id] == 1:
                rig += 1
                if max_score >= self.score_thre:
                    thre_rig +=1

        acc = float(rig) / sum
        thre_acc = float(thre_rig) / sum
        print("\nTest Accuracy = {}\n".format(acc))
        print("\nTest Thre Accuracy = {}\n".format(thre_acc))
        return acc


    def test_unit(self, text):
        text = self.pre.get_dl_input_by_text(text)
        test_batches = self.generator.get_test_batch(self.text_list,
                                                     self.embedding.maxlen,
                                                     self.embedding.maxlen,
                                                     query = text)
        graph = load_pb(self.conf['model_path'])
        sess = tf.Session(graph=graph)

        #self.scores = graph.get_operation_by_name(self.output_nodes)
        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")

        preds = []
        for batch in test_batches:
            pred, labels = self.test_step(batch, self.embedding, self.encoder, self.vocab_dict,
                           graph, sess, self.scores)
            preds += list(pred)
        preds = np.exp(preds)/sum(np.exp(preds))
        max_id = self.knn(pred)
        #max_id = np.argmax(preds)
        max_score = preds[max_id]
        print(self.text_list[max_id], self.label_list[max_id], max_score)
        return self.label_list[max_id], max_score



