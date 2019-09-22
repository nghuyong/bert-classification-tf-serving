export PYTHONPATH=./google-bert
export WORKDIR=./google-bert
export BERT_BASE_DIR=./models/chinese_L-12_H-768_A-12
export DATASET=./data/chnsenticorp
export OUTPUT_DIR=./models/fine-tuning-model
export EXPORT_MODEL_DIR=./models/export-model
export TASK_NAME=chnsenticorp
export CUDA_VISIBLE_DEVICES=0


python $WORKDIR/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_predict=true \
  --data_dir=$DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR \
  --export_model_dir=$EXPORT_MODEL_DIR