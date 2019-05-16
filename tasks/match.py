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
        if not self.use_language_model:
            self.vocab_dict = embedding[self.embedding_type].build_dict(\
                                                dict_path = self.dict_path,
                                                text_list = self.text_list,
                                                mode = self.mode)

            #define embedding object by embedding_type
            self.embedding = embedding[self.embedding_type](text_list = self.text_list,
                                                            vocab_dict = self.vocab_dict,
                                                            dict_path = self.dict_path,
                                                            random=self.rand_embedding,
                                                            maxlen = self.maxlen,
                                                            batch_size = self.batch_size,
                                                            embedding_size =
                                                            self.embedding_size,
                                                            conf = self.conf)

            self.embed_query = self.embedding('x_query')
            self.embed_sample = self.embedding('x_sample')
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
        self.sess = tf.Session()
        #self.writer = tf.summary.FileWriter("logs/match_log", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

        if self.use_language_model:
            tvars = tf.trainable_variables()
            init_checkpoint = self.init_checkpoint_path
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

        if self.mode == 'test_one':
            self.cached_vecs = self._generate_labels_vec()
            self.recall = Annoy(self.cached_vecs)

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
                                                 self.num_epochs,
                                                 self.maxlen,
                                                 self.maxlen,
                                                 task = self,
                                                 mode =self.batch_mode)

        max_acc = -1
        for batch in train_batches:
            x1_batch, x2_batch = batch
            train_feed_dict = {
                self.is_training: True
            }

            if not self.use_language_model:
                _, x1_batch, x1_len_batch = self.embedding.text2id(x1_batch, self.vocab_dict,
                                                     need_preprocess = False)
                _, x2_batch, x2_len_batch = self.embedding.text2id(x2_batch, self.vocab_dict,
                                                     need_preprocess = False)
                train_feed_dict.update(self.embedding.feed_dict(x1_batch,'x_query'))
                train_feed_dict.update(self.embedding.feed_dict(x2_batch,'x_sample'))
                train_feed_dict.update(\
                            self.encoder.feed_dict(x_query = x1_len_batch, 
                                                   x_sample = x2_len_batch))
            else:
                train_feed_dict.update(self.encoder.feed_dict(x1_batch, x2_batch))

            _, step, loss = self.sess.run([self.optimizer, self.global_step, self.loss], feed_dict=train_feed_dict)
            #logging.info(loss)

            if step % (self.valid_step/10) == 0:
                logging.info("step {0}: loss = {1}".format(step, loss))
            if step % (self.valid_step) == 0:
                #validation
                test_batches = self.generator.get_test_batch(self.text_list,
                                                             self.maxlen,
                                                             self.maxlen)
                sum, rig, thre_rig = 0, 0, 0
                for x1_batch,x2_batch,labels_batch in test_batches:
                    test_feed_dict = {
                        self.is_training: False
                    }
                    if not self.use_language_model:
                        _, x1_batch, x1_len_batch = self.embedding.text2id(x1_batch,
                                                             self.vocab_dict,
                                                             need_preprocess = False)
                        _, x2_batch, x2_len_batch = self.embedding.text2id(x2_batch,
                                                             self.vocab_dict,
                                                             need_preprocess = False)

                        test_feed_dict.update(self.embedding.feed_dict(x1_batch,'x_query'))
                        test_feed_dict.update(self.embedding.feed_dict(x2_batch,'x_sample'))
                        test_feed_dict.update(\
                                    self.encoder.feed_dict(x_query = x1_len_batch, 
                                                           x_sample = x2_len_batch))
                    else:
                        test_feed_dict.update(self.encoder.feed_dict(x1_batch, x2_batch))
                    pred = self.sess.run(self.pred, feed_dict=test_feed_dict)


                    assert len(pred) == len(labels_batch), "len(pred)!=len(labels_batch)!"
                    max_id = self.knn(pred)
                    max_score = pred[max_id]
                    sum += 1
                    if labels_batch[max_id] == 1:
                        rig += 1
                        if max_score >= self.score_thre:
                            thre_rig +=1

                acc = float(rig) / sum
                logging.info("\nValid Accuracy = {}\n".format(acc))

                #acc = self.test()
                if acc > max_acc:
                    max_acc = acc
                    # Save model
                    self.saver.save(self.sess,
                                    "{0}/{1}.ckpt".format(
                                        self.checkpoint_path,
                                        self.task_type),
                                    global_step=step)
                else:
                    self.save_pb()
                    logging.info(f'train finished! max valid accuracy: {max_acc}')
                    if self.early_stop == True:
                        sys.exit(0)

    def save_pb(self):
        write_pb(self.checkpoint_path,self.model_path,["is_training", self.output_nodes])

    def knn(self, pred, k = 5):
        sorted_id = np.argsort(-pred,axis=0)
        mp = defaultdict(int)
        for idx in range(k):
            mp[int(sorted_id[idx])] += 1
        max_id = max(mp,key=mp.get)
        return max_id


    def test_step(self, batch, embedding, encoder, vocab_dict,  graph, sess,
                  run_param, is_training):
        x1_batch,x2_batch,labels_batch = batch
        test_feed_dict = {
            is_training: False
        }
        if not self.use_language_model:
            _, x1_batch, x1_len_batch = embedding.text2id(x1_batch,
                                                 vocab_dict,
                                                 need_preprocess = False)
            _, x2_batch, x2_len_batch = embedding.text2id(x2_batch,
                                                 vocab_dict,
                                                 need_preprocess = False)
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            test_feed_dict.update(embedding.pb_feed_dict(graph,x1_batch,'x_query'))
            test_feed_dict.update(embedding.pb_feed_dict(graph,x2_batch,'x_sample'))
            #test_feed_dict.update(encoder.pb_feed_dict(graph, len = [x1_len_batch,x2_len_batch]))
            test_feed_dict.update(encoder.pb_feed_dict(graph, 
                                                       x_query = x1_len_batch, 
                                                       x_sample = x2_len_batch))
        else:
            test_feed_dict.update(encoder.feed_dict(x1_batch, x2_batch))
        pred = sess.run(run_param, feed_dict=test_feed_dict)
        return pred, labels_batch

    def test(self):
        # Test accuracy with validation data for each epoch.
        test_batches = self.generator.get_test_batch(self.text_list,
                                                     self.maxlen,
                                                     self.maxlen)
        if not os.path.exists(self.model_path):
            self.save_pb()
        graph = load_pb(self.model_path)
        sess = tf.Session(graph=graph)
        #self.scores = graph.get_operation_by_name(self.output_nodes)
        self.scores = graph.get_tensor_by_name(self.output_nodes+":0")
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]
        sum, rig, thre_rig = 0, 0, 0
        for batch in test_batches:
            pred, labels = self.test_step(batch, self.embedding, self.encoder, self.vocab_dict,
                           graph, sess, self.scores, self.is_training)
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
        logging.info("\nTest Accuracy = {}\n".format(acc))
        logging.info("\nTest Thre Accuracy = {}\n".format(thre_acc))
        return acc


    def test_unit(self, text, use_recall = True):
        vec = self._get_raw_texts_vec([text])[0]
        if use_recall:
            id_list, score_list = self.recall(vec)
            vecs = [self.cached_vecs[idx] for idx in id_list]
            cosine_dis = cosine_similarity([vec], vecs)[0]
            max_id = id_list[np.argmax(cosine_dis)]
        else:
            cosine_dis = cosine_similarity([vec], self.cached_vecs)[0]
            max_id = np.argmax(cosine_dis)
        max_score = np.max(cosine_dis)
        logging.info(f"{self.text_list[max_id]}, {self.label_list[max_id]}, {max_score}")
        return self.label_list[max_id], max_score

    def _get_raw_texts_vec(self, text_list):
        #获取原始句子的句向量
        for idx, text in enumerate(text_list):
            text_list[idx] = self.pre.get_dl_input_by_text(text)
        vec = self._get_vec_by_batches([text_list])
        return vec

    def _generate_labels_vec(self):
        #生成所有训练样本的句向量
        assert self.sim_mode == 'represent', "only represent mode supports cache label vectors"
        vecs = self._get_vec_by_batches([self.text_list])
        return vecs

    def _get_vec_by_batches(self, batches):
        #根据batches数据生成向量
        if not os.path.exists(self.model_path):
            self.save_pb()
        graph = load_pb(self.model_path)
        sess = tf.Session(graph=graph)
        rep = graph.get_tensor_by_name(self.rep_nodes+":0")
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        feed_dict = {is_training: False}
        vecs = []
        for batch in batches:
            feed_dict = {
                is_training: False
            }
            _, x1_batch, x1_len_batch = self.embedding.text2id(batch,
                                                     self.vocab_dict,
                                                     need_preprocess = False)
            feed_dict.update(self.embedding.pb_feed_dict(graph,x1_batch,'x_query'))
            feed_dict.update(self.encoder.pb_feed_dict(graph, x_query = x1_len_batch))
            vecs = sess.run(rep, feed_dict=feed_dict)
        return vecs

