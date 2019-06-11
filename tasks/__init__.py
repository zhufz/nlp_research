#-*- coding:utf-8 -*-
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)
from classify import Classify
from classify_m import ClassifyM
from match import Match
from ner import NER
from seq_generate import SeqGenerate
from translation import Translation

dl_tasks = {}
dl_tasks['classify'] = Classify
dl_tasks['match'] = Match
dl_tasks['ner'] = NER
dl_tasks['seq_generate'] = SeqGenerate
dl_tasks['translation'] = Translation


ml_tasks = {}
ml_tasks['classify_m'] = ClassifyM
