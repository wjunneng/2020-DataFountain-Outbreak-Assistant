python preprocess.py \
  --train_dir data/train.csv \
  --clean_train_dir data/train1.csv \
  --passage_dir data/passage.csv \
  --clean_passage_dir data/passage1.csv \
  --test_dir data/test.csv \
  --clean_test_dir data/test1.csv \
  --es_index passages \
  --es_ip localhost

python run_qa.py \
  --model_name_or_path ../chinese_wwm_ext_pytorch \
  --do_train \
  --do_eval \
  --es_index passages \
  --es_ip localhost \
  --data_dir ./data/data_0 \
  --test_dir ./data/test1.csv \
  --passage_dir ./data/passage1.csv \
  --output_dir ./output/ \
  --max_seq_length 512 \
  --max_question_length 96 \
  --eval_steps 50 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --train_steps 1000

python run_qa.py \
  --model_name_or_path ../chinese_wwm_ext_pytorch \
  --do_test \
  --k 10 \
  --es_index passages \
  --es_ip localhost \
  --data_dir ./data/data_0 \
  --test_dir ./data/test1.csv \
  --passage_dir ./data/passage1.csv \
  --output_dir ./output \
  --max_seq_length 512 \
  --max_question_length 96 \
  --eval_steps 50 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 1e-5 \
  --train_steps 1000
