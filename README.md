# nlp_research


## 介绍
  
    本框架支持的NLP任务包括 分类、匹配、序列标注、文本生成等.
    - 对于分类任务，目前支持多分类、多标签分类，通过选择不同的loss即可。
    - 对于匹配任务，目前已支持交互模型和表示模型。

## 数据

    训练数据:
    对于分类任务的数据使用csv格式，csv头部包括列名‘target’和‘text’;
    对于匹配任务的数据使用csv格式，csv头部包括列名‘target’,‘text’ 或者 ‘target’,‘text_a’,‘text_b’

    预训练数据(目前在分类和匹配任务上已支持):
    如果使用到bert作为预训练，请提前运行"sh scripts/prepare.sh"

## 快速开始

    [依赖]
         环境：python3+tensorflow 1.10
         pip3 install --user -r requirements.txt
         
    [分类]
         1.生成tfrecords数据，训练:
            python3 run.py classify mode=prepare
            python3 run.py classify 
           或者直接使用脚本:
            sh scripts/restart.sh classify
         
         2.测试：python3 run.py classify model=test
           单个测试：python3 run.py classify model=test_one
    [匹配]
         1.生成tfrecords数据，训练:
             python3 run.py match mode=prepare
             python3 run.py match mode=train
           或者直接使用脚本:
             sh scripts/restart.sh match
         2.测试：python3 run.py match model=test
           单个测试：python3 run.py match model=test_one
    [序列标注]
        ...
        sh scripts/restart.sh ner
    [翻译]    
        ...
        sh scripts/restart.sh translation
## 任务

    各类任务的参数分别定义在conf/model/的，以任务名命名的yml文件中"conf/model/{task}.yml"
    目前已支持的常见任务如下：
    1. classify, 训练
    2. match   , 匹配
    3. ner     , 序列标注
    4. seq2seq , 翻译任务

## 模块

    1. encoder
        cnn
        fasttext
        text_cnn
        dcnn
        dpcnn
        vdcnn
        rnn        
        rcnn
        attention_rnn
        capsule
        esim
        han
        matchpyramid
        abcnn
        transformer
  
    2. common 
        loss
        attention
        lr
        ...
    
    3. utils
        data process
## 联系

    如果有任何问题，欢迎发邮件到zfz1015@outlook.com
    
  
