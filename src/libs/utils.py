# -*- coding:utf-8 -*-
import collections
import os
import re
import torch
import json
from glob import glob
from collections import Counter

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_all_variables(rank=0):
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True if rank == 0 else False)


# import ipdb
def get_assigment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    new_variable_names = set()
    unused_variable_names = set()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            if 'adam' not in name:
                unused_variable_names.add(name)
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name in name_to_variable:
        if name not in initialized_variable_names:
            new_variable_names.add(name)
    return assignment_map, initialized_variable_names, new_variable_names, unused_variable_names


# loading weights
def init_from_checkpoint(init_checkpoint, tvars=None, rank=0):
    if not tvars:
        tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names, new_variable_names, unused_variable_names \
        = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if rank == 0:
        # 显示成功加载的权重
        for t in initialized_variable_names:
            if ":0" not in t:
                print("Loading weights success: " + t)

        # 显示新的参数
        print('New parameters:', new_variable_names)

        # 显示初始化参数中没用到的参数
        print('Unused parameters', unused_variable_names)


def check_args(args):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    if rank == 0:
        print("Total param num：" + str(k))


def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def torch_save_model(model, output_dir, scores, max_save_num=1):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    saved_pths = glob(os.path.join(output_dir, '*.pth'))
    saved_pths.sort()
    while len(saved_pths) >= max_save_num:
        if os.path.exists(saved_pths[0].replace('//', '/')):
            os.remove(saved_pths[0].replace('//', '/'))
            del saved_pths[0]

    save_prex = "checkpoint_score"
    for k in scores:
        save_prex += ('_' + k + '-' + str(scores[k])[:6])
    save_prex += '.pth'

    torch.save(model_to_save.state_dict(),
               os.path.join(output_dir, save_prex))
    print("Saving model checkpoint to %s", output_dir)


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
        # 注意 text列 可能会被划分多个
        docid2context[r[0]] = '\t'.join(r[1:])
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
    """
    寻找开始/结束位置
    :param line:
    :return:
    """
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
            sample['answer']['span'] = [-1, -1]

    return sample
