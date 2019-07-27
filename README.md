# nlp_research


## 介绍
  
    本项目支持的NLP任务包括 分类、匹配、序列标注、文本生成等.
    - 对于分类任务，目前支持多分类、多标签分类，通过选择不同的loss即可。
    - 对于匹配任务，目前已支持交互模型和表示模型。
    - 对于NER任务，目前已支持rnn+crf,idcnn+crf以及bert+crf
## 数据

    训练数据(目前data下均内置了样例数据):
    （1）对于分类任务的数据使用csv格式，csv头部包括列名‘target’和‘text’;
    （2）对于匹配任务的数据使用csv格式，csv头部包括列名‘target’,‘text’ 或者 ‘target’,‘text_a’,‘text_b’
    （3）对于NER任务的数据，参考"data/ner/train_data",或者使用其它格式的数据的话，修改task/ner.py中的read_data方法即可。
    预训练数据(目前在分类和匹配任务上已支持):
    - 如果使用到bert作为预训练(直接下载google训练好的模型即可)，直接运行"sh scripts/prepare.sh"
    - 如果使用elmo作为预训练，需要准备一份corpus.txt训练语料放在language_model/bilm_tf/data/目录下
          然后执行指令进行预训练： 
                cd language_model/bilm_tf
                sh start.sh
## 快速开始
    [依赖]
         环境：python3+tensorflow 1.10(python2.7已支持)
         pip3 install --user -r requirements.txt
         
    各类任务的参数定义在conf/model/内的以任务名命名的yml文件中"conf/model/***.yml"
    目前已支持的常见任务如下：       
    [分类]
         1.生成tfrecords数据，训练:
            python3 run.py classify.yml mode=train
           或者直接使用脚本:
            sh scripts/restart.sh classify.yml
         
         2.测试：
           单个测试：python3 run.py classify.yml model=test_one
    [匹配]
         1.生成tfrecords数据，训练:
             python3 run.py match.yml mode=train
           或者直接使用脚本:
             sh scripts/restart.sh match.yml
         2.测试：
            单个测试：python3 run.py match.yml model=test_one
    [序列标注]
        ...
        sh scripts/restart.sh ner.yml
    [翻译]    
        ...
        sh scripts/restart.sh translation.yml

## 模块

    1. encoder
        cnn
        fasttext
        text_cnn
        dcnn
        idcnn
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
        
## 参考文献
    1. [2014 dcnn]A Convolutional Neural Network for Modelling Sentences
    2. [2014 textcnn] Convolutional Neural Networks for Sentence Classification
    3. [2015 charCNN] Character-level Convolutional Networks for TextClassification
    4. [2016 HAN] Hierarchical Attention Networks for Document Classification
    5. [2016-fasttext]Bag of Tricks for Efficient Text Classification
    6. [2017 vdcnn] Very Deep Convolutional Networks for Text Classification
    7. [2017_ACL dpcnn] Deep Pyramid Convolutional Neural Networks for Text Categorization
    8. [2018] Investigating Capsule Networks with Dynamic Routing for Text Classification
    9. [2018_ACL] Disconnected Recurrent Neural Networks for Text Categorization
    10.[2018] Investigating Capsule Networks with Dynamic Routing for Text Classification
    11.[2018] Topic Memory Networks for Short Text Classification
    12.Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    13.[2016] A Deep Relevance Matching Model for Ad-hoc Retrieval
    14.[2016] ABCNN Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
    15.[2016] Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function
    16.[2016] Text Matching as Image Recognition
    17.[2017 ACL,smn] Sequential Matching Network A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots
    18.[2017 bimpm] Bilateral Multi-Perspective Matching for Natural Language Sentences
    19.[2017 esim] Enhanced LSTM for Natural Language Inference
    20.[2017] IRGAN A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
    21.[2018] MIX Multi-Channel Information Crossing for Text Matching
    22.[2017-NIPS] Attention-is-all-you-need
    23.[2018-AAAI] DiSAN Directional Self-Attention Network for RNNCNN-Free Language Understanding
    24.[2018-ICLR]Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling
    25.[2018] An Introductory Survey on Attention Mechanisms in NLP Problems
    26.[2018] Universal Transformers
    27.[2018 naacl ELMo]  Deep contextualized word representations
    28.[2018 iclr quick-thoughts]An efficient framework for learning sentence representations
    29.[2017 subword] EnrichingWord Vectors with Subword Information
    30.[2018] Universal Language Model Fine-tuning for Text Classification


## 联系

    如果有任何问题，欢迎发邮件到zfz1015@outlook.com
    如果觉得我的工作对您有所帮助，请不要吝啬右上角的小星星哦！
    欢迎Watch/Fork/Star！也欢迎一起建设这个项目！
    
  
