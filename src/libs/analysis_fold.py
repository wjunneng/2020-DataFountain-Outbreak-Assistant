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
from src.libs.utils import get_data_frame, re_find_fake_answer_I, load_context, load_query_docids

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
    生成训练/验证集
    :param context_path:
    :param train_path:
    :param train_0301_path:
    :param dev_0301_path:
    :return:
    """
    # docid2context: docid text
    docid2context = get_data_frame(context_path)
    docid2context = dict(zip(docid2context['docid'], docid2context['text']))

    if os.path.exists(train_0301_path) and os.path.exists(dev_0301_path):
        os.remove(train_0301_path)
        os.remove(dev_0301_path)

    train_0301 = open(train_0301_path, 'a')
    dev_0301 = open(dev_0301_path, 'a')

    train_docid_set = set()
    dev_docid_set = set()

    # line: id	docid	question	answer
    with open(file=train_path, mode='r') as file:
        first_line = True
        for line in file:
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
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid].replace(' ', ''),
                      'question': question.replace(' ', ''),
                      'answer': {'text': answer.replace(' ', '')}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))
                # 注意此出可以优化， 当前暂时跳过不计
                if rv['answer']['span'] is None:
                    continue

                train_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            elif docid in train_docid_set and docid not in dev_docid_set:
                dev_docid_set.add(docid)
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid].replace(' ', ''),
                      'question': question.replace(' ', ''),
                      'answer': {'text': answer.replace(' ', '')}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))
                # 注意此出可以优化， 当前暂时跳过不计
                if rv['answer']['span'] is None:
                    continue

                dev_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            else:
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid].replace(' ', ''),
                      'question': question.replace(' ', ''),
                      'answer': {'text': answer.replace(' ', '')}}

                # type_1
                # rv = re_find_fake_answer(json.dumps(rv))
                # type_2
                rv = re_find_fake_answer_I(json.dumps(rv))
                # 注意此出可以优化， 当前暂时跳过不计
                if rv['answer']['span'] is None:
                    continue

                train_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')

    train_0301.close()
    dev_0301.close()


def generate_test(context_path, test_path, test_0301_path, query_docids_path):
    """
    生成测试集
    :param context_path:
    :param test_path:
    :param test_0301_path:
    :param query_docids_path:
    :return:
    """
    # docid2context: docid text
    docid2context = load_context(context_path)
    question2docid = load_query_docids(query_docids_path)

    if os.path.exists(test_0301_path):
        os.remove(test_0301_path)

    test_0301 = open(test_0301_path, 'a')

    with open(file=test_path, mode='r') as file:
        first_line = True
        for line in file:
            if first_line:
                first_line = False
                continue

            line = line.strip().split('\t')
            assert len(line) == 2
            id = line[0]
            question = line[1]
            docid = question2docid[question]
            context = docid2context[docid]

            rv = {'id': id, 'docid': docid, 'context': context, 'question': question}

            test_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')

    test_0301.close()


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

    # 生成训练验证集 速度特别慢
    generate_train_dev_file(context_path=arguments.context_path, train_path=arguments.train_path,
                            train_0301_path=arguments.train_0301_path, dev_0301_path=arguments.dev_0301_path)

    # 生成测试集
    # generate_test(context_path=arguments.context_path, test_path=arguments.test_path,
    #               test_0301_path=arguments.test_0301_path, query_docids_path=arguments.query_docids_path)
