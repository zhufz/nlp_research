from tasks import dl_tasks
import yaml
import os,sys
import pdb
import time

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

def change_path(conf):
    path_root = os.path.join(ROOT_PATH, conf['path_root'])
    for k,v in conf.items():
        if k.endswith('_path'):
            conf[k] = os.path.join(path_root, conf[k])

if __name__ == '__main__':

    conf = yaml.load(open('task.yml'))
    task_type = conf['task_type']
    base = conf['base']
    conf = conf[task_type]
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
    #创建目录
    model_path = '/'.join(conf['model_path'].split('/')[:-1])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if len(sys.argv) >1:
        #additional params from cmd
        for idx, arg in enumerate(sys.argv):
            if idx ==0:continue
            key,value = arg.split('=')
            if value.isdigit():value = int(value)
            conf[key] = value

    print("conf:",conf)
    cl = dl_tasks[task_type](conf)
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
