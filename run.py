from tasks import dl_tasks
import yaml
import os,sys
import pdb

ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

if __name__ == '__main__':

    conf = yaml.load(open('task.yml'))
    task_type = conf['task_type']
    conf = conf[task_type]
    if "config" and "config_type" in conf:
        config_type = conf['config_type']
        for k,v in (conf['config'][config_type]).items():
            conf[k] = v
    for k,v in conf.items():
        if k.endswith('_path'):
            conf[k] = os.path.join(ROOT_PATH, conf[k])

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
            cl.test_unit(a)
