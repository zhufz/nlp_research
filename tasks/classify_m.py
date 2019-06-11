#-*- coding:utf-8 -*-
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
#import lightgbm as lgb
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(ROOT_PATH)
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from utils.preprocess import Preprocess
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pdb


class ClassifyM():
    def __init__(self):
        self.preprocess = Preprocess()
        self.thre = 0.5

    def load(self, train_path, test_path):
        #load train_data
        csv = pd.read_csv(train_path)
        train_list = self.preprocess.process(csv['text'])
        self.labels = {item: idx for idx,item in enumerate(set(csv['intent']))}
        self.output_label()
        self.labels_rev = {self.labels[item]:item for item in self.labels}
        self.labels_rev[-1] = '未知'

        print("class_num:",len(self.labels))
        self.labels_num = len(self.labels)
        y_train = [self.labels[item] for item in csv['intent']]
        #train data weight
        self.vectorizer = TfidfVectorizer()
        train_weight = self.vectorizer.fit_transform([' '.join(item) for item in
                                                      train_list])
        #load test_data
        self.result_path = test_path + ".result.csv"
        csv = pd.read_csv(test_path)
        test_list = self.preprocess.process(csv['text'])
        y_test = [self.labels[item] for item in csv['intent']]  #int label
        #test data weight
        test_weight = self.vectorizer.transform([' '.join(item) for item in
                                                 test_list])

        self.data = {}
        self.data['x_train'] = train_weight
        self.data['y_train'] = y_train
        self.data['x_test'] = test_weight
        self.data['y_test'] = y_test
        self.data['raw_test_list'] = csv['text']
        self.data['test_list'] = test_list

    def output_label(self):
        with open('data/label.txt','w') as f:
            for item in self.labels:
                f.write('{}\t{}\n'.format(item, self.labels[item]))

    def train(self):
        ### fit model for train data
        self.model = XGBClassifier(learning_rate=0.1,
                              n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                              max_depth=6,               # 树的深度
                              min_child_weight = 1,      # 叶子节点最小权重
                              gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                              subsample=0.8,             # 随机选择80%样本建立决策树
                              colsample_btree=0.8,       # 随机选择80%特征建立决策树
                              objective='multi:softmax', # 指定损失函数
                              scale_pos_weight=1,        # 解决样本个数不平衡的问题
                              random_state=27            # 随机数
                              )
        print("开始训练！")
        self.model.fit(self.data['x_train'],
                  self.data['y_train'],
                  eval_set = [(self.data['x_test'],self.data['y_test'])],
                  eval_metric = "mlogloss",
                  early_stopping_rounds = 10,
                  verbose = True)

        ### model evaluate

        predictions = self.model.predict_proba(self.data['x_test'])
        y_pred = np.argmax(predictions, 1)
        scores = [predictions[idx][y_pred[idx]] for idx in range(len(y_pred))]
        for idx in range(len(y_pred)):
            if scores[idx] < self.thre:
                y_pred[idx] = -1
        accuracy = accuracy_score(self.data['y_test'],y_pred)
        print("accuarcy: %.2f%%" % (accuracy*100.0))

        dt = pd.DataFrame({'text':self.data['raw_test_list'],
                           'feature':self.data['test_list'], 
                           'target':[self.labels_rev[item] for item in
                                     self.data['y_test']] ,
                           'pred': [self.labels_rev[item] for item in 
                                    y_pred],
                           'score': scores })
        dt.to_csv(self.result_path,index=False,sep=',')

    def test(self, text):
        test_list = self.preprocess.process([text])
        test_weight = self.vectorizer.transform([' '.join(item) for item in
                                                 test_list])
        predictions = self.model.predict_proba(test_weight)
        pred = np.argmax(predictions, 1)
        print(self.labels_rev[pred[0]],predictions[0][pred[0]])

    def test(self, file):
        lines = [line.strip() for line in open(file).readlines()]
        test_list = self.preprocess.process(lines)
        test_weight = self.vectorizer.transform([' '.join(item) for item in
                                                 test_list])
        predictions = self.model.predict_proba(test_weight)
        pred = np.argmax(predictions, 1)
        pdb.set_trace()
        with open(file+'.res','w') as f_w:
            for idx,line in enumerate(lines):
                f_w.write("{}\t{}\t{}\n".format(line,
                                                self.labels_rev[pred[idx]],
                                                predictions[idx][pred[idx]]))

    def process(self,train_path, test_path):
        self.load(train_path, test_path)
        self.train()

if __name__ == '__main__':
    cls = ClassifyM()
    cls.process('data/intent_train.csv', 'data/intent_test.csv')
    #while True:
    #    a = input("input:")
    #    cls.test(a)
    #cls.test('data_raw/intent.data')


