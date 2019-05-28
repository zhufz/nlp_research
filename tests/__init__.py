import sys,os
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(ROOT_PATH)
from tests.test import TestClassify, TestMatch

tests = {}
tests['classify'] = TestClassify
tests['match'] = TestMatch
