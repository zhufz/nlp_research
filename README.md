# nlp_research


## 介绍
nlp一些任务集成：classify，match，sequence tagging，translation...

## 快速开始

1. 分类：

    模型配置文件: conf/model/classify.yml

    训练：python3 run.py classify

    测试：python3 run.py classify model=test

    单个测试：python3 run.py classify model=test_one


## 数据
  数据统一放在data下，

  对于分类和匹配任务，训练文件每行格式为"文本\t类别"，然后调用generate_ml_data.sh进行数据生成！

  对于ner任务，参考示例数据

## 任务
环境：python3+tensorflow 1.10

任务参数定义在task.yml中，具体执行代码在tasks文件夹中。
目前已支持三种常见任务：

1. classify

    分类任务

2. match

    匹配任务

3. ner

    标注任务

4. seq2seq

    文本生成

## 模块说明

1. encoder

    存放编码器，目前已支持的编码器列表：
    
    abcnn
    
    attention_rnn
    
    capsule
    
    cnn
    
    dcnn
    
    dpcnn
    
    esim
    
    fast_attention_text
    
    fasttext
    
    han
    
    matchpyramid
    
    rcnn
    
    rnn
    
    text_cnn
    
    transformer
    
    vdcnn
    

2. common

    存放一些通用网络中间层

3. utils

    数据处理工具

