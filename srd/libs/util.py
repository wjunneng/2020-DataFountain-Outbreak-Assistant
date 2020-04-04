# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])

import json
import pandas as pd


def get_data_frame(path):
    """
    读取原始数据，返回DataFrame数据类型
    """
    data = {}
    if os.path.splitext(path)[-1] == '.csv':
        with open(path, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file.readlines()):
                line = line.strip('\n')
                if index == 0:
                    for char in line.split('\t'):
                        data[char] = []
                    continue

                r = line.strip().split('\t')
                keys = list(data.keys())

                if len(r) > len(keys):
                    r[1] = '\t'.join(r[1:])
                    r = r[:2]

                for index in range(len(keys)):
                    data[keys[index]].append(r[index])

    elif os.path.splitext(path)[-1] == '.json':
        with open(path, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file.readlines()):
                if index == 0:
                    for key, value in eval(line).items():
                        if key == 'answer':
                            start_index = int(value['span'][0])
                            end_index = int(value['span'][1])
                            # # 注意此出可以优化， 当前暂时跳过不计
                            # if start_index >= end_index:
                            #     continue
                            answer = value['text']
                            answer = answer.strip('\n')
                            data['start_index'] = [start_index]
                            data['end_index'] = [end_index]
                            data['answer'] = [answer]
                            continue

                        data[key] = [value]
                else:
                    for key, value in eval(line).items():
                        if key == 'answer':
                            start_index = int(value['span'][0])
                            end_index = int(value['span'][1])
                            # # 注意此出可以优化， 当前暂时跳过不计
                            # if start_index >= end_index:
                            #     continue
                            answer = value['text']
                            answer = answer.strip('\n')
                            data['start_index'].append(start_index)
                            data['end_index'].append(end_index)
                            data['answer'].append(str(answer))
                            continue

                        data[key].append(value)

    return pd.DataFrame(data)


def re_find_fake_answer_I(line):
    """
    寻找开始/结束位置
    :param line:
    :return:
    """
    sample = json.loads(line)
    if sample is not None:
        context = sample['context']
        answer = sample['answer']['text']
        if answer in context:
            start_index = context.index(answer)
            end_index = start_index + len(answer) - 1
            sample['answer']['span'] = [start_index, end_index]
        else:
            sample['answer']['span'] = None

    return sample


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


if __name__ == '__main__':
    # train_csv_path = '/home/wjunneng/Ubuntu/NLP/2020_4/COVID19_qa_baseline/data/train.csv'
    # test_csv_path = '/home/wjunneng/Ubuntu/NLP/2020_4/COVID19_qa_baseline/data/test.csv'
    # passage_csv_path = '/home/wjunneng/Ubuntu/NLP/2020_4/COVID19_qa_baseline/data/passage.csv'
    #
    # train_csv_data = get_data_frame(train_csv_path)
    # test_csv_data = get_data_frame(test_csv_path)
    # passage_csv_data = get_data_frame(passage_csv_path)

    """
        {'text': '开通应急物资快速审批通道，把审批医疗器械生产“两证”的时间从一年压缩至半个月。', 'span': [-1, -1]}
        {'text': '2020/1/31', 'span': [-1, -1]}
        {'text': '全面加强一线疫情防控工作。', 'span': [-1, -1]}
    """
    # train_json_path = '/home/wjunneng/Ubuntu/NLP/2020_4/COVID19_qa_baseline/data/train.json'
    # get_data_frame(train_json_path)

    """
        {'text': '北京市园林绿化局北京市市场监督管理局\u3000\u3000北京市农业农村局', 'span': [-1, -1]}
        {'text': '事项1：经市经济和信息化局认定的55家北京市小型微型企业创业创新示范基地中非国有资产的基地。事项2：在疫情期间为相关小微企业开展融资担保服务且降低小微企业担保费率的本市政府性担保、再担保机构。事项3：为中小微企业提供中小微企业服务券。事项4、5：受疫情影响的、有相关需求的我市中小微企业', 'span': [-1, -1]}
    """
    # dev_json_path = '/home/wjunneng/Ubuntu/NLP/2020_4/COVID19_qa_baseline/data/dev.json'
    # get_data_frame(dev_json_path)

    # generate_train_dev_file(context_path='./data/passage1.csv',
    #                         train_path='./data/train.csv',
    #                         train_0301_path='./data/train.json',
    #                         dev_0301_path='./data/dev.json')
