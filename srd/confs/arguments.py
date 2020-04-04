# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

sys.path.append(os.path.abspath('.'))

from pathlib import Path

local = False
PROJECT_DIR = Path(os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1]))

# -* original data *-
DATA_DIR = PROJECT_DIR.joinpath('data')
INPUT_DIR = DATA_DIR.joinpath('input')

# docid	text
CONTEXT_PATH = INPUT_DIR.joinpath('NCPPolicies_context_20200301.csv')
# id	docid	question	answer
TRAIN_PATH = INPUT_DIR.joinpath('NCPPolicies_train_20200301.csv')
# id	question
TEST_PATH = INPUT_DIR.joinpath('NCPPolicies_test.csv')
# id	docid	answer
SUBMIT_PATH = INPUT_DIR.joinpath('submit_example.csv')
