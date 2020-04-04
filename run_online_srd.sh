#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export DATA_DIR=$CURRENT_DIR/data
export SRD_DIR=$CURRENT_DIR/srd
export FOLD_DIR=$DATA_DIR/fold
export INPUT_DIR=$DATA_DIR/input
export OUTPUT_DIR=$DATA_DIR/output
export PREV_TRAINED_MODEL=$DATA_DIR/prev_trained_model

# 记得开启elasticsearch
#python $SRD_DIR/libs/preprocess.py \
#  --passage_dir $INPUT_DIR/NCPPolicies_context_20200301.csv \
#  --train_dir $INPUT_DIR/NCPPolicies_train_20200301.csv \
#  --test_dir $INPUT_DIR/NCPPolicies_test.csv \
#  --train_json_path $FOLD_DIR/train.json \
#  --dev_json_path $FOLD_DIR/dev.json \
#  --clean_passage_dir $FOLD_DIR/passage.csv \
#  --clean_train_dir $FOLD_DIR/train.csv \
#  --clean_test_dir $FOLD_DIR/test.csv \
#  --es_index passages \
#  --es_ip localhost

python $SRD_DIR/run.py \
  --model_name_or_path $PREV_TRAINED_MODEL/chinese_roberta_wwm_ext_pytorch \
  --do_train \
  --do_eval \
  --es_index passages \
  --es_ip localhost \
  --data_dir $DATA_DIR \
  --train_dir $FOLD_DIR/train.csv \
  --test_dir $FOLD_DIR/test.csv \
  --train_json_path $FOLD_DIR/train.json \
  --dev_json_path $FOLD_DIR/dev.json \
  --passage_dir $FOLD_DIR/passage.csv \
  --output_dir $OUTPUT_DIR \
  --max_seq_length 512 \
  --max_question_length 96 \
  --eval_steps 50 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --learning_rate 1e-5 \
  --train_steps 100

#python $SRD_DIR/run.py \
#  --model_name_or_path $PREV_TRAINED_MODEL/chinese_roberta_wwm_ext_pytorch \
#  --do_test \
#  --k 10 \
#  --es_index passages \
#  --es_ip localhost \
#  --data_dir $DATA_DIR \
#  --test_dir $FOLD_DIR/test.csv \
#  --train_json_path $FOLD_DIR/train.json \
#  --dev_json_path $FOLD_DIR/dev.json \
#  --passage_dir $FOLD_DIR/passage.csv \
#  --output_dir $OUTPUT_DIR \
#  --max_seq_length 512 \
#  --max_question_length 96 \
#  --eval_steps 50 \
#  --per_gpu_train_batch_size 16 \
#  --per_gpu_eval_batch_size 64 \
#  --learning_rate 1e-5 \
#  --train_steps 1000
