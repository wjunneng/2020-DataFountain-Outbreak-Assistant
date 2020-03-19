# -*- coding:utf-8 -*-
import os
import sys

# 项目地址
sys.path.append('/home/wjunneng/Ubuntu/2020-DataFountain-Outbreak-Assistant')

import pandas as pd
import io
import random
import json

random.seed(1)

from src.confs import arguments
from src.libs.utils import re_find_fake_answer, re_find_fake_answer_I, load_context

if not os.path.exists(arguments.fold_dir):
    os.makedirs(arguments.fold_dir)

if os.path.exists(arguments.fold_report_path):
    os.system('rm report.txt')


def analyzefile(file):
    df = pd.read_csv(file, delimiter="\t", error_bad_lines=False)
    os.system("touch " + os.path.basename(arguments.fold_report_path))

    os.system("echo '-------------information about " + os.path.basename(
        arguments.fold_report_path) + " set------------' >> " + os.path.basename(
        arguments.fold_report_path))

    os.system("echo 'the row number of " + os.path.basename(arguments.fold_report_path) + " is " + str(
        df.shape[0]) + "' >> " + os.path.basename(
        arguments.fold_report_path))

    os.system("echo '\n-------------the describe of " + os.path.basename(
        arguments.fold_report_path) + " is ------------------ \n" + str(
        df.describe()) + "' >> " + os.path.basename(arguments.fold_report_path))

    os.system("echo '\n--------------the info of " + os.path.basename(
        arguments.fold_report_path) + " data --------------- \n' >> " + os.path.basename(arguments.fold_report_path))

    print(arguments.fold_report_path)
    f = open(arguments.fold_report_path, 'a')
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    f.write(info)
    f.write("\n\n\n")
    f.close()


def generate_train_dev_file(context_path, train_path, train_0301_path, dev_0301_path):
    """
    :param context_path:
    :param train_path:
    :return:
    """
    # docid2context: docid text
    docid2context = load_context(context_path)

    if os.path.exists(train_0301_path) and os.path.exists(dev_0301_path):
        os.remove(train_0301_path)
        os.remove(dev_0301_path)

    train_0301 = open(train_0301_path, 'a')
    dev_0301 = open(dev_0301_path, 'a')

    train_docid_set = set()
    dev_docid_set = set()

    index = 0
    # line: id	docid	question	answer
    with open(file=train_path, mode='r') as file:
        first_line = True
        for line in file:
            index += 1
            if first_line:
                first_line = False
                continue

            line = line.strip().split('\t')
            assert len(line) == 4
            id = line[0]
            docid = line[1]
            question = line[2]
            answer = line[3]

            if docid not in train_docid_set:
                train_docid_set.add(docid)
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))

                train_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            elif docid in train_docid_set and docid not in dev_docid_set:
                dev_docid_set.add(docid)
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))
                dev_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            else:
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))
                train_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')


def generate_test():
    pass


if __name__ == '__main__':
    pass

    # # """
    # # 分析文件
    # # """
    # analyzefile(arguments.context_path)
    # analyzefile(arguments.train_path)
    # analyzefile(arguments.test_path)
    # # max / mean / min
    # stat_length(filename=arguments.context_path)

    # 速度特别慢
    generate_train_dev_file(context_path=arguments.context_path, train_path=arguments.train_path,
                            train_0301_path=arguments.train_0301_path, dev_0301_path=arguments.dev_0301_path)
