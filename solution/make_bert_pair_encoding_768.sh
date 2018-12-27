#!/bin/sh

date

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BERT_BASE_DIR=../chinese_L-12_H-768_A-12
export WSDM_DIR=../data

CUDA_VISIBLE_DEVICES=6 nohup python extract_features_wsdm.py   --input_file=../data/train.csv   --output_file=../data/train_meanpool768_layer_2.jsonl   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-2   --max_seq_length=156   --batch_size=48  > ../log/train_meanpool768_layer_2.log </dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python extract_features_wsdm.py   --input_file=../data/test.csv   --output_file=../data/test_meanpool768_layer_2.jsonl   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-2   --max_seq_length=156   --batch_size=32 > ../log/test_meanpool768_layer_2.log </dev/null 2>&1 &
