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
from task_base import TaskBase
import numpy as np
import time
import pdb


class XGB(TaskBase):
    def __init__(self, conf):
        super(XGB, self).__init__(conf)
        self.preprocess = Preprocess()
        self.vectorizer = TfidfVectorizer()
        self.thre = 0.5
        self.read_data()

    def output_label(self):
        with open(self.dict_path,'w') as f:
            for item in self.labels:
                f.write('{}\t{}\n'.format(item, self.labels[item]))

    def read_data(self):
        #load train_data
        csv = pd.read_csv(self.ori_path, header = 0, sep="\t", error_bad_lines=False)
        train_list = self.preprocess.process(csv['text'])
        self.labels = {item: idx for idx,item in enumerate(set(csv['target']))}
        self.labels_rev = {self.labels[item]:item for item in self.labels}
        self.labels_rev[-1] = '未知'
        self.output_label()
        print("class_num:",len(self.labels))
        #train data weight
        X = self.vectorizer.fit_transform([' '.join(item) for item in train_list])
        y = [self.labels[item] for item in csv['target']]

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size = self.test_size, 
                                                            random_state=0)


        self.data = {}
        self.data['x_train'] = X_train
        self.data['y_train'] = y_train
        self.data['x_test'] = X_test
        self.data['y_test'] = y_test

    def train(self):
        ### fit model for train data
        self.model = XGBClassifier(learning_rate=0.3,
                              n_estimators=100,         # 树的个数--1000棵树建立xgboost
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
        #for idx in range(len(y_pred)):
        #    if scores[idx] < self.thre:
        #        y_pred[idx] = -1
        accuracy = accuracy_score(self.data['y_test'],y_pred)
        print("accuarcy: %.2f%%" % (accuracy*100.0))

        #dt = pd.DataFrame({'text':self.data['raw_test_list'],
        #                   'feature':self.data['test_list'], 
        #                   'target':[self.labels_rev[item] for item in
        #                             self.data['y_test']] ,
        #                   'pred': [self.labels_rev[item] for item in 
        #                            y_pred],
        #                   'score': scores })
        #dt.to_csv(self.result_path,index=False,sep=',')

    def test(self, mode = 'test'):
        assert os.path.exists(file), "file [%s] not existed!"%file
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

