# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

sys.path.append(os.path.abspath('.'))

project_dir = '/home/wjunneng/Ubuntu/2020-DataFountain-Outbreak-Assistant'

# -* original data *-
data_dir = os.path.join(project_dir, 'data')
input_dir = os.path.join(data_dir, 'input')

# docid	text
context_path = os.path.join(input_dir, 'NCPPolicies_context_20200301.csv')
# id	docid	question	answer
train_path = os.path.join(input_dir, 'NCPPolicies_train_20200301.csv')
# id	question
test_path = os.path.join(input_dir, 'NCPPolicies_test.csv')
# id	docid	answer
submit_path = os.path.join(input_dir, 'submit_example.csv')

# -* generate data *-
output_dir = os.path.join(data_dir, 'output')
fold_dir = os.path.join(data_dir, 'fold')
fold_report_path = os.path.join(fold_dir, 'report.txt')
local_dir = os.path.join(data_dir, 'local')
local_report_path = os.path.join(local_dir, 'report.txt')

local = True
if local:
    if os.path.exists(local_dir) is False:
        os.makedirs(local_dir)
    train_0301_path = os.path.join(local_dir, 'train.json')
    dev_0301_path = os.path.join(local_dir, 'dev.json')
    test_0301_path = os.path.join(local_dir, 'test.json')
else:
    train_0301_path = os.path.join(fold_dir, 'train.json')
    dev_0301_path = os.path.join(fold_dir, 'dev.json')
    test_0301_path = os.path.join(fold_dir, 'test.json')
