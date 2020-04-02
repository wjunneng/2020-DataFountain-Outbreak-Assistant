#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=albert_xlarge
export DATA_DIR=$CURRENT_DIR/data
export SRC_DIR=$CURRENT_DIR/src
export FOLD_DIR=$DATA_DIR/fold
export INPUT_DIR=$DATA_DIR/input
export OUTPUT_DIR=$DATA_DIR/output
export BERT_DIR=$DATA_DIR/prev_trained_model/$MODEL_NAME

#python src/tools/convert_tf_checkpoint_to_pytorch.py \
#  --tf_checkpoint_path=$BERT_DIR/model.ckpt-best.index \
#  --bert_config_file=$BERT_DIR/albert_config.json \
#  --pytorch_dump_path=$BERT_DIR/pytorch_albert_model.pth \
#  --is_albert \

#TASK_NAME='CMRC2018'
TASK_NAME="Outbreak_Assistant"
python $SRC_DIR/run.py \
  --gpu_ids="0" \
  --train_epochs=2 \
  --eval_epochs=1 \
  --n_batch=4 \
  --lr=3e-5 \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab_chinese.txt \
  --bert_config_file=$BERT_DIR/albert_config.json \
  --init_restore_dir=$BERT_DIR/pytorch_albert_model.pth \
  --train_file=$FOLD_DIR/train.json \
  --dev_file=$FOLD_DIR/dev.json \
  --train_dir=$FOLD_DIR/train_features.json \
  --dev_dir1=$FOLD_DIR/dev_examples.json \
  --dev_dir2=$FOLD_DIR/dev_features.json \
  --checkpoint_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/

python $SRC_DIR/run_test.py \
  --gpu_ids="0" \
  --n_batch=4 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab_chinese.txt \
  --bert_config_file=$BERT_DIR/albert_config.json \
  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --test_dir1=$FOLD_DIR/test_examples.json \
  --test_dir2=$FOLD_DIR/test_features.json \
  --test_file=$FOLD_DIR/test.json
