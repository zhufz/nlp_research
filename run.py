from tasks import dl_tasks
import yaml
import os,sys
import pdb
import time
import logging

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

def init_logging(logFilename):
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

def change_path(conf):
    path_root = os.path.join(ROOT_PATH, conf['path_root'])
    for k,v in conf.items():
        if k.endswith('_path'):
            conf[k] = os.path.join(path_root, conf[k])

def read_conf(task_type):
    init_logging('log')
    base_yml = "conf/model/base.yml"
    task_yml = f"conf/model/{task_type}.yml"
    assert os.path.exists(task_yml),'fmodel {task_type} does not exists!'

    conf = yaml.load(open(task_yml))
    base = yaml.load(open(base_yml))

    #相对路径->绝对路径
    change_path(base)
    change_path(conf)

    #加载base信息
    for k,v in base.items():
        conf[k] = v
    if "config" and "config_type" in conf:
        config_type = conf['config_type']
        for k,v in (conf['config'][config_type]).items():
            conf[k] = v
    #更新encoder_type信息
    for k,v in conf.items():
        if type(v) == str and (v.find('{encoder_type}')) != -1:
            conf[k] = v.replace("{encoder_type}", conf['encoder_type'])
    #创建相关目录
    model_path = '/'.join(conf['model_path'].split('/')[:-1])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #使用指令中的额外参数更新conf
    if len(sys.argv) >1:
        #additional params from cmd
        for idx, arg in enumerate(sys.argv):
            if idx ==0:continue
            if arg.find("=") == -1:continue
            key,value = arg.split('=')
            if value.isdigit():value = int(value)
            conf[key] = value
    return conf,task_type


if __name__ == '__main__':
    if len(sys.argv) <1:
        print("task type missed, classify, match, ner...?")
    task_type = sys.argv[1]

    conf, task_type = read_conf(task_type)
    cl = dl_tasks[task_type](conf)
    logging.info(conf)
    if conf['mode'] == 'train':
        cl.train()
        cl.test()
    elif conf['mode'] == 'test':
        cl.test()
    elif conf['mode'] == 'predict':
        cl.predict()
    elif conf['mode'] == 'test_one':
        while True:
            a = input('input:')
            start = time.time()
            cl.test_unit(a)
            end = time.time()
            consume = end-start
            print(f'consume: {consume}')
