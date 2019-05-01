# nlp_research


## 介绍
nlp一些任务集成：classify，match，sequence tagging, translation...

## 任务
环境：python3+tensorflow 1.10

任务参数定义在task.yml中,具体执行代码在tasks文件夹中。
目前已支持三种常见任务：

1. classify

    分类任务

2. match

    匹配任务

3. ner

    标注任务

## 模块说明

1. encoder

    存放编码器

2. common

    存放一些通用网络中间层

3. utils

    数据处理工具

## 计划
1.nlp_research初版，支持分类、匹配模型（已完成）

2.集成序列标注模型（已完成）

3.小样本学习支持

4.集成bert

5.强化学习（基于任务型对话）








