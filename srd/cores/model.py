# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])

from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn


class BertForQuestionAnswering(BertPreTrainedModel):
    """
        机器阅读问答模型
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # start/end
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        # [batch_size, max_sequence_length, hidden_size]
        sequence_output = outputs[0]
        # [batch_size, hidden_size]
        pooled_output = outputs[1]

        # predict start & end position
        sequence_output = self.dropout(sequence_output)
        # # torch.Size([2, 512, 768])
        # print('sequence_output.shape: {}'.format(sequence_output.shape))
        qa_logits = self.qa_outputs(sequence_output)
        # # qa_logits.shape: torch.Size([2, 512, 2])
        # print('qa_logits.shape: {}'.format(qa_logits.shape))
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        # # torch.Size([2, 512])
        # print(start_logits.shape)
        # # torch.Size([2, 512])
        # print(end_logits.shape)
        # # torch.Size([2, 2])
        # print(classifier_logits.shape)

        if labels is not None:
            start_labels, end_labels, class_labels = labels
            start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_logits, start_labels)
            end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_logits, end_labels)
            class_loss = nn.CrossEntropyLoss()(classifier_logits, class_labels)
            outputs = start_loss + end_loss + 2 * class_loss
        else:
            outputs = (start_logits, end_logits, classifier_logits)

        return outputs
