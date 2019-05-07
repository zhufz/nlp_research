import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)
from classify import Classify
from classify_m import ClassifyM
from match import Match
from ner import NER
from seq2seq import Seq2seq

dl_tasks = {}
dl_tasks['classify'] = Classify
dl_tasks['match'] = Match
dl_tasks['ner'] = NER
dl_tasks['seq2seq'] = Seq2seq


ml_tasks = {}
ml_tasks['classify_m'] = ClassifyM
