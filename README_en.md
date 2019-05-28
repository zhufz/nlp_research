# nlp_research


## Introduction
  
    Nlp tasks supported in this lab：classify，match，sequence tagging，translation...

## Data
train datas:

    we use csv formats data as training data, and its head contains ['target','text']

language model data:

    if you use bert model as pretrained model, run "sh scripts/prepare.sh" first

## Quickly start
    [requirements]
         pip3 install --user -r requirements.txt
    [classify]
         1.tfrecords prepare and train:
            python3 run.py classify mode=prepare
            python3 run.py classify 
           or:
            sh scripts/restart.sh classify
         
         2.test：python3 run.py classify model=test
           test one：python3 run.py classify model=test_one
    [match]
         1.tfrecords prepare and train:
             python3 run.py match mode=prepare
             python3 run.py match mode=train
            or:
             sh scripts/restart.sh match
         2.test：python3 run.py match model=test
           test one：python3 run.py match model=test_one
## Task

    Environments：python3+tensorflow 1.10
    The parameters for each type of task are defined in the yml file named "conf/model/{task}.yml".
    Common tasks currently supported are:
    1. classify, used to train classify model
    2. match   , used to train sentence match model
    3. ner     , used to train sequence tagging model
    4. seq2seq , used to train sentence generation model

## Module

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
## Contact

    if you have any problems, you can send email to zfz1015@outlook.com
    
  
