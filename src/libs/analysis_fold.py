# -*- coding:utf-8 -*-
import os
import sys

# 项目地址
sys.path.append('/home/wjunneng/Ubuntu/2020-DataFountain-Outbreak-Assistant')

import pandas as pd
import io
import random
import json
from collections import Counter

random.seed(1)

from src.confs import arguments

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


def stat_length(filename):
    """
    :param filename:
    :return:
    """
    df = pd.read_csv(filename, sep='\t', error_bad_lines=False)
    df['context_length'] = df['text'].apply(len)
    print(f"context length: {df['context_length'].mean()}")
    print(f"context length: {df['context_length'].max()}")
    print(f"context length: {df['context_length'].min()}")


def load_context(filename):
    """
    :param filename:
    :return:
    """
    docid2context = {}
    f = True
    for line in open(filename):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        docid2context[r[0]] = r[1]
    return docid2context


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def re_find_fake_answer(line):
    sample = json.loads(line)
    best_match_score = 0
    best_match_span = [-1, -1]
    if sample is not None:
        for start_tidx in range(len(sample['context'])):
            if sample['context'][start_tidx] not in sample['answer']['text']:
                continue
            for end_tidx in range(len(sample['context']) - 1, start_tidx - 1, -1):
                span_tokens = sample['context'][start_tidx: end_tidx + 1]
                match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                            [sample['answer']['text']])
                if match_score > best_match_score:
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
        if best_match_score > 0:
            sample['answer']['span'] = best_match_span

    return sample


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
            print(index)
            if first_line:
                first_line = False
                continue

            line = line.strip().split('\t')
            id = line[0]
            docid = line[1]
            question = line[2]
            answer = line[3]

            if docid not in train_docid_set:
                train_docid_set.add(docid)
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                rv = re_find_fake_answer(json.dumps(rv))
                train_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            elif docid in train_docid_set and docid not in dev_docid_set:
                dev_docid_set.add(docid)
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                rv = re_find_fake_answer(json.dumps(rv))
                dev_0301.write(json.dumps(rv, ensure_ascii=False) + '\n')
            else:
                rv = {'id': id, 'docid': docid, 'context': docid2context[docid], 'question': question,
                      'answer': {'text': answer}}

                rv = re_find_fake_answer(json.dumps(rv))
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
