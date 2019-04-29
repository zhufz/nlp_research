import pandas as pd
import sys
from collections import defaultdict

class Convert():
    def __init__(self):
        pass


    def process(self):
        """
        针对训练语料和测试语料格式进行调整，增加类别文件，类别转出序号
        """
        train_path = './data_raw/intent_train.csv.feature'
        test_path = './data_raw/intent_test.csv.feature'
        res_path = "./data/train.csv"
        test_res_path = "./data/test.csv"
        class_path = "./data/classes"

        dt = pd.read_csv(train_path)
        dt_test = pd.read_csv(test_path)
        classes = set(dt['intent'])
        class_mp = {}
        class_mp_rev = {}
        for idx,item in enumerate(classes):
            class_mp[item] = idx
            class_mp_rev[idx] = item

        with open(class_path,'w') as f_w:
            for idx in range(len(class_mp)):
                f_w.write("{}\n".format(class_mp_rev[idx]))



        target = [class_mp[item]+1 for item in dt['intent']]
        target_test = [class_mp[item]+1 for item in dt_test['intent']]
        dt_res = pd.DataFrame({'class':target,'content':dt['text']})
        dt_test_res = pd.DataFrame({'class':target_test,'content':dt_test['text']})


        dt_res.to_csv(res_path,index=False,sep=',')
        dt_test_res.to_csv(test_res_path,index=False,sep=',')
        print("convert finished!")


if __name__ == '__main__':
    split = Convert()
    split.process()
    #split.process(sys.argv[1])






