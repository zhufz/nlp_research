from tasks import dl_tasks
from tests.test_unit import Test
import yaml
import os,sys
import pdb
import time
import logging
from utils.generate_data import GenerateData

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

class Run():
    def __init__(self, init_log = False):
        if init_log:
            self.init_logging('log')

    def init_logging(self, logFilename):
        logging.basicConfig(
                        level    = logging.DEBUG,
                        #format   = '%(asctime)s\t%(filename)s,line %(lineno)s\t%(levelname)s: %(message)s',
                        format   = '%(asctime)s\t%(levelname)s: %(message)s',
                        datefmt  = '%Y/%m/%d %H:%M:%S',
                        filename = logFilename,
                        filemode = 'w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s\t%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def _change_path(self, conf):
        path_root = os.path.join(ROOT_PATH, conf['path_root'])
        for k,v in conf.items():
            if k.endswith('_path'):
                conf[k] = os.path.join(path_root, conf[k])

    def read_config_type(self, conf):
        #读取config信息，对应不同的参数
        if "config" and "config_type" in conf:
            config_type = conf['config_type']
            for k,v in (conf['config'][config_type]).items():
                conf[k] = v
        del conf['config']

    def read_conf(self, task_type):
        base_yml = os.path.join(ROOT_PATH, "conf/model/base.yml")
        task_yml = os.path.join(ROOT_PATH, f"conf/model/{task_type}.yml")
        assert os.path.exists(task_yml),'fmodel {task_type} does not exists!'
        conf = yaml.load(open(task_yml))
        self.read_config_type(conf)
        base = yaml.load(open(base_yml))
        #相对路径->绝对路径
        self._change_path(base)
        self._change_path(conf)
        #加载base信息
        for k,v in base.items():
            conf[k] = v
        #更新encoder_type信息
        for k,v in conf.items():
            if type(v) == str and (v.find('{encoder_type}')) != -1:
                conf[k] = v.replace("{encoder_type}", conf['encoder_type'])
        #创建相关目录
        model_path = '/'.join(conf['model_path'].split('/')[:-1])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if 'tfrecords_path' in conf:
            tfrecords_path = conf['tfrecords_path']
            if not os.path.exists(tfrecords_path):
                os.makedirs(tfrecords_path)
        #使用指令中的额外参数更新conf
        if len(sys.argv) >1:
            #additional params from cmd
            for idx, arg in enumerate(sys.argv):
                if idx ==0:continue
                if arg.find("=") == -1:continue
                key,value = arg.split('=')
                if value.isdigit():value = int(value)
                conf[key] = value
        return conf


if __name__ == '__main__':
    assert len(sys.argv) > 1,"task type missed, classify, match, ner...?"
    task_type = sys.argv[1]

    run = Run(init_log = True)
    conf = run.read_conf(task_type)

    logging.info(conf)

    if conf['mode'] == 'train':
        cl = dl_tasks[task_type](conf)
        cl.train()
        cl.test()
    elif conf['mode'] == 'test':
        cl = dl_tasks[task_type](conf)
        cl.test()
    elif conf['mode'] == 'predict':
        cl = dl_tasks[task_type](conf)
        cl.predict()
    elif conf['mode'] in ['test_one','test_unit']:
        #cl = dl_tasks[task_type](conf)
        conf['task_type'] = task_type
        cl = Test(conf)
        while True:
            a = input('input:')
            start = time.time()
            cl.test_unit(a)
            end = time.time()
            consume = end-start
            print(f'consume: {consume}')
    elif conf['mode'] == 'prepare':
        split = GenerateData(conf)
        if task_type in ['match','classify']:
            cl = dl_tasks[task_type](conf)
            cl.prepare()
        else:
            raise ValueError('unknown task type for prepare data step!')
    else:
        raise ValueError('unknown mode!')
