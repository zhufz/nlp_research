import tensorflow as tf
import argparse
import pdb
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from utils.data_utils import batch_iter,load_class_mp
from tasks import dl_tasks
import yaml


class Test:
    def __init__(self):
        self.conf = yaml.load(open('task.yml'))
        self.task_type = self.conf['task_type']
        checkpoint_file = tf.train.latest_checkpoint("{}/{}".format(self.conf['path'],self.task_type))
        graph = tf.get_default_graph()
        self.sess = tf.Session()

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.sess, checkpoint_file)

        self.x = graph.get_operation_by_name("x").outputs[0]
        self.y = graph.get_operation_by_name("y").outputs[0]
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]
        self.accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        self.scores = graph.get_tensor_by_name("output/scores:0")
        self.predictions = graph.get_tensor_by_name("output/predictions:0")

        self.model = dl_tasks['classify']()
        self.mp, self.mp_rev = load_class_mp(self.conf['classes_path'])

    def process(self):
        test_x, test_y = self.model.load_data("test")
        pdb.set_trace()
        batches = batch_iter(zip(test_x, test_y), self.conf['batch_size'], 1)
        sum_accuracy, cnt = 0, 0
        right, all = 0, 0
        for batch in batches:
            batch_x, batch_y = zip(*batch)
            feed_dict = {
                self.x: batch_x,
                self.y: batch_y,
                self.is_training: False
            }
            accuracy_out, predictions_out, scores_out = self.sess.run([self.accuracy,
                                                                  self.predictions,
                                                                  self.scores],
                                                                 feed_dict=feed_dict)
            max_scores = [scores_out[idx][predictions_out[idx]] \
                          for idx in range(len(predictions_out))]
            sum_accuracy += accuracy_out
            cnt += 1

            for idx in range(len(predictions_out)):
                if predictions_out[idx] == batch_y[idx] and max_scores[idx]> 0.4:
                    right += 1
                all += 1
        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
        print("Test Thre Accuracy : {0}".format(right / all))

    def process_unit(self, text):
        test_x = self.model.embedding.text2id([text])
        batches = batch_iter(test_x, self.conf['batch_size'], 1)
        for batch_x in batches:
            feed_dict = {
                self.x: batch_x,
                self.is_training: False
            }
            #pdb.set_trace()
            predictions_out, scores_out = self.sess.run([self.predictions,
                                                        self.scores],
                                                        feed_dict=feed_dict)
            max_scores = [scores_out[idx][predictions_out[idx]] \
                          for idx in range(len(predictions_out))]
        print("class:{}, score:{}, class_id:{}".format(
            self.mp_rev[predictions_out[0]],
            max_scores[0],
            predictions_out[0]))


if __name__ == '__main__':
    test = Test()
    #test.process()
    while True:
        a = input('input:')
        test.process_unit(a)
