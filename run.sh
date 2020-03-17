#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=albert_xlarge
export DATA_DIR=$CURRENT_DIR/data
export INPUT_DIR=$DATA_DIR/input
export OUTPUT_DIR=$DATA_DIR/output
export BERT_DIR=$DATA_DIR/prev_trained_model/$MODEL_NAME

#python src/tools/convert_tf_checkpoint_to_pytorch.py \
#  --tf_checkpoint_path=$BERT_DIR/model.ckpt-best.index \
#  --bert_config_file=$BERT_DIR/albert_config.json \
#  --pytorch_dump_path=$BERT_DIR/pytorch_albert_model.pth \
#  --is_albert \

#TASK_NAME='CMRC2018'
TASK_NAME="Outbreak Assistant"
python run.py \
  --gpu_ids="0" \
  --train_epochs=2 \
  --n_batch=2 \
  --lr=3e-5 \
  --warmup_rate=0.1 \
  --max_seq_length=128 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab_chinese.txt \
  --bert_config_file=$BERT_DIR/albert_config.json \
  --init_restore_dir=$BERT_DIR/pytorch_albert_model.pth \
  --train_file=$INPUT_DIR/train.json \
  --dev_file=$INPUT_DIR/dev.json \
  --train_dir=$INPUT_DIR/train_features.json \
  --dev_dir1=$INPUT_DIR/dev_examples.json \
  --dev_dir2=$INPUT_DIR/dev_features.json \
  --checkpoint_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/

#python test_mrc.py \
#  --gpu_ids="0" \
#  --n_batch=8 \
#  --max_seq_length=512 \
#  --task_name=$TASK_NAME \
#  --vocab_file=$BERT_DIR/vocab.txt \
#  --bert_config_file=$BERT_DIR/bert_config.json \
#  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#  --test_dir1=$DATA_DIR/$TASK_NAME/test_examples.json \
#  --test_dir2=$DATA_DIR/$TASK_NAME/test_features.json \
#  --test_file=$DATA_DIR/$TASK_NAME/cmrc2018_test_2k.json \
