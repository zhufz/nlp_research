#-*- coding:utf-8 -*-
import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)
from classify import Classify
from xgb import XGB
from match import Match
from ner import NER
from translation import Translation

tasks = {}
tasks['classify'] = Classify
tasks['match'] = Match
tasks['ner'] = NER
tasks['translation'] = Translation

tasks['xgb'] = XGB
