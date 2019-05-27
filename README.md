# nlp_research


## Introduction
  
    nlp一些任务集成：classify，match，sequence tagging，translation...

## Data
train datas:

    数据统一放在data下，
    对于分类和匹配任务，训练文件每行格式为"文本\t类别".
    对于ner任务，参考示例数据

language model data:

    如果需要使用bert预训练模型，请先下载预训练模型：sh scripts/prepare.sh

## Quickly start
    [依赖]
         pip3 install --user -r requirements.txt
    [分类]
         生成语料：python3 run.py classify classify
         训练：python3 run.py classify 
         测试：python3 run.py classify model=test
         单个测试：python3 run.py classify model=test_one
    [匹配]
         生成tfrecords语料: python3 run.py match mode=prepare
         训练: python3 run.py match mode=train
         测试：python3 run.py match model=test
         单个测试：python3 run.py match model=test_one
## Task

    环境：python3+tensorflow 1.10
    各类任务的参数分别定义在conf/model/的，以任务名命名的yml文件中！
    目前已支持的常见任务如下：
    1. classify 分类任务
    2. match    匹配任务 
    3. ner      标注任务
    4. seq2seq  文本生成

## Module

    1. encoder（存放编码器）
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
  
    2. common（存放一些通用网络中间层）
        loss
        attention
        lr
        ...
    
    3. utils
        数据处理工具
