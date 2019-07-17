#-*- coding:utf-8 -*-
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

from encoder_base import EncoderBase
encoder = {}
#sentence encoder

encoder["dcnn"] = getattr(__import__('dcnn'),'DCNN')
encoder["vdcnn"] = getattr(__import__('vdcnn'),'VDCNN')
encoder["idcnn"] = getattr(__import__('idcnn'),'IDCNN')
encoder["dpcnn"] = getattr(__import__('dpcnn'),'DPCNN')
encoder["text_cnn"] = getattr(__import__('text_cnn'),'TextCNN')
encoder["rcnn"] = getattr(__import__('rcnn'),'RCNN')
encoder["rnn"] = getattr(__import__('rnn'),'RNN')
encoder["attention_rnn"] = getattr(__import__('attention_rnn'),'AttentionRNN')
encoder["transformer"] = getattr(__import__('transformer'),'Transformer')
encoder["fasttext"] = getattr(__import__('fasttext'),'FastText')
encoder["fast_attention_text"] = getattr(__import__('fast_attention_text'),'FastAttentionText')
encoder["han"] = getattr(__import__('han'),'HAN')
encoder["capsule"] = getattr(__import__('capsule'),'Capsule')
#translation:
encoder["seq2seq"] = getattr(__import__('seq2seq'),'Seq2seq')

#with pretrain language model
#such as elmo, bert...
encoder["bert"] = getattr(__import__('bert'), 'Bert')

#pair sentence encoder
encoder["match_pyramid"] = getattr(__import__('match_pyramid'),'MatchPyramid')
encoder["abcnn"] = getattr(__import__('abcnn'),'ABCNN')
encoder["esim"] = getattr(__import__('esim'),'ESIM')



