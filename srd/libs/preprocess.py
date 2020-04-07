# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])

import argparse
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from srd.libs.util import get_data_frame, generate_train_dev_file


def dispose_train(train_df, passage_df, clean_train_dir):
    s, e, a = [], [], []
    for i, val in enumerate(train_df[['id', 'docid', 'question', 'answer']].values):
        passage = passage_df[passage_df['docid'] == val[1]].values[0][-1]
        ans_l = len(val[-1])
        start = passage.find(val[-1])
        end = start + ans_l
        s.append(start)
        e.append(end)
        # 如果训练集中存在答案不能和文档匹配情况
        if val[-1] != passage[start:end]:
            a.append(i)

    train_df['start_index'] = s
    train_df['end_index'] = e
    train_df.drop(a, inplace=True)
    train_df.to_csv(clean_train_dir, index=False, header=True)


def clean_data(train_dir, clean_train_dir, passage_dir, clean_passage_dir, test_dir, clean_test_dir):
    # clean passage file
    passage = get_data_frame(passage_dir)
    for i in passage.index:
        passage.iloc[i, -1] = passage.iloc[i, -1].replace(' ', '')

    passage.to_csv(clean_passage_dir, index=False, header=True)
    # clean train file
    train = get_data_frame(train_dir)

    for i in train.index:
        train.iloc[i, 2] = train.iloc[i, 2].replace(' ', '')
        train.iloc[i, 3] = train.iloc[i, 3].replace(' ', '')

    dispose_train(train, passage, clean_train_dir)
    # clean test file
    test = get_data_frame(test_dir)
    for i in test.index:
        test.iloc[i, -1] = test.iloc[i, -1].replace(' ', '')

    test.to_csv(clean_test_dir, index=False, header=True)
    print('clean done')


class ElasticObj:
    def __init__(self, index_name, index_type, ip="127.0.0.1"):
        """
        :param index_name: 索引名称
        :param index_type: 索引类型
        """
        self.index_name = index_name
        self.index_type = index_type
        # 无用户名密码状态
        self.es = Elasticsearch([ip])

    def bulk_Index_Data(self, csvfile):
        """
        用bulk将批量数据存储到es
        :return:
        """
        df = pd.read_csv(csvfile)
        doc = []
        for item in df.values:
            dic = {}
            dic['docid'] = item[0]
            dic['passage'] = item[1]
            doc.append(dic)
        ACTIONS = []
        i = 0
        for line in doc:
            action = {
                "_index": self.index_name,
                "_type": self.index_type,
                "_source": {
                    "docid": line['docid'],
                    "passage": line['passage']}
            }
            i += 1
            ACTIONS.append(action)
            # 批量处理
        print('index_num:', i)
        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)

    def create_index(self):
        """
        创建索引,创建索引名称为ott，类型为ott_type的索引
        :param ex: Elasticsearch对象
        :return:
        """
        # 创建映射
        _index_mappings = {
            "mappings": {
                self.index_type: {
                    "properties": {
                        "passage": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_max_word"
                        },
                        "docid": {
                            "type": "text"
                        }
                    }
                }
            }
        }
        if self.es.indices.exists(index=self.index_name) is not True:
            res = self.es.indices.create(index=self.index_name, body=_index_mappings)
            print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--passage_dir", default=None, type=str, required=True,
                        help="The passage data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_dir", default=None, type=str, required=True,
                        help="The test data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_json_path", default=None, type=str, required=True,
                        help="The train json path. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dev_json_path", default=None, type=str, required=True,
                        help="The dev json path. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--clean_train_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--clean_passage_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--clean_test_dir", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--es_index", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--es_ip", default=None, type=str, required=True,
                        help="")
    args = parser.parse_args()
    clean_data(args.train_dir, args.clean_train_dir, args.passage_dir, args.clean_passage_dir, args.test_dir,
               args.clean_test_dir)
    # 建立ES，把文档批量导入索引节点
    obj = ElasticObj(args.es_index, "_doc", ip=args.es_ip)
    obj.create_index()
    obj.bulk_Index_Data(args.clean_passage_dir)

    generate_train_dev_file(context_path=args.passage_dir,
                            train_path=args.train_dir,
                            train_0301_path=args.train_json_path,
                            dev_0301_path=args.dev_json_path)
