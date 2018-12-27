#!/bin/sh

date

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BERT_BASE_DIR=../chinese_L-12_H-768_A-12
export WSDM_DIR=../data

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_nohidden/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy >> ../log/2e5_3epo_156ml_weighted_extra33_nohidden.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_nohidden_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy >> ../log/2e5_3epo_156ml_weighted_extra33_nohidden_si.log 2>&1 &
