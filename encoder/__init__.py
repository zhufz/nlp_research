import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

encoder = {}
#sentence encoder
from cnn import CNN
from dcnn import DCNN
from dpcnn import DPCNN
from vdcnn import VDCNN
from rcnn import RCNN
from rnn import RNN
from attention_rnn import AttentionRNN
from text_cnn import TextCNN
from transformer import Transformer
from fasttext import FastText
from fast_attention_text import FastAttentionText
from han import HAN

encoder["cnn"] = CNN
encoder["dcnn"] = DCNN
encoder["vdcnn"] = VDCNN
encoder["dpcnn"] = DPCNN
encoder["text_cnn"] = TextCNN
encoder["rcnn"] = RCNN
encoder["rnn"] = RNN
encoder["attention_rnn"] = AttentionRNN
encoder["transformer"] = Transformer
encoder["fasttext"] = FastText
encoder["fast_attention_text"] = FastAttentionText
encoder["han"] = HAN


#pair sentence encoder
from match_pyramid import MatchPyramid
from abcnn import ABCNN
from esim import ESIM

encoder["match_pyramid"] = MatchPyramid
encoder["abcnn"] = ABCNN
encoder["esim"] = ESIM



