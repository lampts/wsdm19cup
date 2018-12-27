#!/bin/sh

echo -----------------------------------------------------------------------------------------------------
echo "SINGLE GPU solution, run steps in sequence, so time consuming, estimated several days"
echo -----------------------------------------------------------------------------------------------------
date

# assuming we have 1 single GPU
export CUDA_VISIBLE_DEVICES=0
export BERT_BASE_DIR=../chinese_L-12_H-768_A-12
export WSDM_DIR=../data

echo "(*) make bert pair encoding in 768 dimensions (estimated ~ 5 hours for train set), outputs are big (~4GB)"
CUDA_VISIBLE_DEVICES=0 nohup python extract_features_wsdm.py   --input_file=../data/train.csv   --output_file=../data/train_meanpool768_layer_2.jsonl   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-2   --max_seq_length=156   --batch_size=48  > ../log/train_meanpool768_layer_2.log </dev/null 2>&1
wait
CUDA_VISIBLE_DEVICES=0 nohup python extract_features_wsdm.py   --input_file=../data/test.csv   --output_file=../data/test_meanpool768_layer_2.jsonl   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-2   --max_seq_length=156   --batch_size=32 > ../log/test_meanpool768_layer_2.log </dev/null 2>&1
wait


echo "(*) make manual 33 handcrafted features"
python make_handcrafted_33_features.py
wait

echo "(*) sanity check the number of lines for train and test sets"
train_line_no=$(cat ../data/train_meanpool768_layer_2.jsonl | wc -l)
test_line_no=$(cat ../data/test_meanpool768_layer_2.jsonl | wc -l)

if [ "$train_line_no" -eq 320552 ] && [ "$test_line_no" -eq 80126 ]; then
  python make_bert768_svd_knn_31_features.py
fi
wait

echo "(*) concatenate 64 manual features"
if [ -e  ../data/X_33_norm.npy ] && [ -e ../data/Xmin_norm.npy ]; then
  python make_concatenate_64_features.py
fi
wait

echo "(*) train 9 trees and 1 logistic regression"
python train_predict_trees_batch1.py
wait

python train_predict_trees_batch2.py
wait

python train_predict_trees_batch3.py
wait

echo "(*) train 18 nnets"
### batch 1 ###
CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v2.py   --task_name=WSDM   --do_train=true   --do_eval=false   --data_dir=$WSDM_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=156   --train_batch_size=32  --predict_batch_size=32  --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=../output/output_2e5_3epo_156ml/ --do_predict=true --swap_input=false --use_class_weights=false >> ../log/2e5_3epo_156ml.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v2.py   --task_name=WSDM   --do_train=true   --do_eval=false   --data_dir=$WSDM_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=156   --train_batch_size=32  --predict_batch_size=32  --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=../output/output_2e5_3epo_156ml_si/ --do_predict=true --swap_input=true --use_class_weights=false >> ../log/2e5_3epo_156ml_si.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v2.py   --task_name=WSDM   --do_train=true   --do_eval=false   --data_dir=$WSDM_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=156   --train_batch_size=32  --predict_batch_size=32  --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=../output/output_2e5_3epo_156ml_weighted/ --do_predict=true --swap_input=false --use_class_weights=true >> ../log/2e5_3epo_156ml_weighted.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v2.py   --task_name=WSDM   --do_train=true   --do_eval=false   --data_dir=$WSDM_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=156   --train_batch_size=32  --predict_batch_size=32  --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=../output/output_2e5_3epo_156ml_weighted_si/ --do_predict=true --swap_input=true --use_class_weights=true >> ../log/2e5_3epo_156ml_weighted_si.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_156ml_weighted_extra33_1layer156/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_5epo_156ml_weighted_extra33_1layer156.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_156ml_weighted_extra33_1layer156_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_5epo_156ml_weighted_extra33_1layer156_si.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_1layer156/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_3epo_156ml_weighted_extra33_1layer156.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_1layer156_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_3epo_156ml_weighted_extra33_1layer156_si.log 2>&1 
wait

### batch 2 ###

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_156ml_weighted_extra64_1layer156/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_5epo_156ml_weighted_extra64_1layer156.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_156ml_weighted_extra64_1layer156_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=156 >> ../log/2e5_5epo_156ml_weighted_extra64_1layer156_si.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=168 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_168ml_weighted_extra64_1layer256/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_5epo_168ml_weighted_extra64_1layer256.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=168 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=../output/output_2e5_5epo_168ml_weighted_extra64_1layer256_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_5epo_168ml_weighted_extra64_1layer256_si.log 2>&1 
wait

## rerun
CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=176 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=4.0 --output_dir=../output/output_2e5_4epo_176ml_weighted_extra64_1layer256/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_4epo_176ml_weighted_extra64_1layer256.log 2>&1
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=176 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=4.0 --output_dir=../output/output_2e5_4epo_176ml_weighted_extra64_1layer256_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_4epo_176ml_weighted_extra64_1layer256_si.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=182 --train_batch_size=16 --predict_batch_size=16 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_182ml_weighted_extra64_1layer256/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_3epo_182ml_weighted_extra64_1layer256.log 2>&1
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=182 --train_batch_size=16 --predict_batch_size=16 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_182ml_weighted_extra64_1layer256_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=64 --extra_train_tensor=$WSDM_DIR/X_64.npy --extra_test_tensor=$WSDM_DIR/X_test_64.npy --extra_num_layers=1 --extra_hidden_units=256 >> ../log/2e5_3epo_182ml_weighted_extra64_1layer256_si.log 2>&1
wait

### batch 3 ###

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_nohidden/ --do_predict=true --swap_input=false --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy >> ../log/2e5_3epo_156ml_weighted_extra33_nohidden.log 2>&1 
wait

CUDA_VISIBLE_DEVICES=0 nohup python run_classifier_v3.py --task_name=WSDM --do_train=true --do_eval=false --data_dir=$WSDM_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=156 --train_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../output/output_2e5_3epo_156ml_weighted_extra33_nohidden_si/ --do_predict=true --swap_input=true --use_class_weights=true --extra_dense_size=33 --extra_train_tensor=$WSDM_DIR/X_33_norm.npy --extra_test_tensor=$WSDM_DIR/X_test_33_norm.npy >> ../log/2e5_3epo_156ml_weighted_extra33_nohidden_si.log 2>&1 
wait

echo "(*) consolidate nnets results"
sh consolidate_nnet_results.sh
wait

echo "(*) make ensembling"
python make_ensembling.py
wait

date
echo -----------------------------------------------------------------------------------------------------
echo "Please check file final_score.csv and submit it."
echo -----------------------------------------------------------------------------------------------------
echo "DONE. HAPPY MODELING <3"
echo "Contact laampt@gmail.com for further information."

