export PYTHONPATH=./google-bert
export WORKDIR=./google-bert
export BERT_BASE_DIR=./models/chinese_L-12_H-768_A-12
export DATASET=./data/chnsenticorp
export OUTPUT_DIR=./models/fine-tuning-model
export TASK_NAME=chnsenticorp
export CUDA_VISIBLE_DEVICES=0

python $WORKDIR/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --num_train_epochs=6.0 \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR