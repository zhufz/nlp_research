#-*- coding:utf-8 -*-
import sys,os,pdb
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)

tests = {}
tests['classify'] =  getattr(__import__('test_classify'),'TestClassify')
tests['match'] =  getattr(__import__('test_match'),'TestMatch')
tests['ner'] =  getattr(__import__('test_ner'),'TestNER')
tests['translation'] =  getattr(__import__('test_translation'),'TestTranslation')
