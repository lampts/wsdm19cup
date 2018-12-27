#!/bin/sh

date

python make_handcrafted_33_features.py

wait

python make_bert768_svd_knn_31_features.py

wait

python make_concatenate_64_features.py

date
