import pandas as pd, numpy as np

ids = []

test = pd.read_csv("../data/test.csv", usecols=['id'])
for i, r in test.iterrows():
    ids.append(r['id'])
    
def get_score(fname):
    scores1 = []
    with open(fname) as f:
        for line in f:
            score = np.array([float(n) for n in line.replace('\n','').split('\t')])
            scores1.append(score)

    return np.array(scores1)

print("(*) get nnets scores")
nnet_score1 = get_score("../sub/output_2e5_3epo_156ml_si_test_results.tsv")
nnet_score2 = get_score("../sub/output_2e5_3epo_156ml_test_results.tsv")
nnet_score3 = get_score("../sub/output_2e5_3epo_156ml_weighted_extra33_1layer156_si_test_results.tsv")
nnet_score4 = get_score("../sub/output_2e5_3epo_156ml_weighted_extra33_1layer156_test_results.tsv")
nnet_score5 = get_score("../sub/output_2e5_3epo_156ml_weighted_extra33_nohidden_si_test_results.tsv")
nnet_score6 = get_score("../sub/output_2e5_3epo_156ml_weighted_extra33_nohidden_test_results.tsv")
nnet_score7 = get_score("../sub/output_2e5_3epo_156ml_weighted_si_test_results.tsv")
nnet_score8 = get_score("../sub/output_2e5_3epo_156ml_weighted_test_results.tsv")
nnet_score9 = get_score("../sub/output_2e5_3epo_182ml_weighted_extra64_1layer256_si_test_results.tsv")
nnet_score10 = get_score("../sub/output_2e5_3epo_182ml_weighted_extra64_1layer256_test_results.tsv")
nnet_score11 = get_score("../sub/output_2e5_4epo_176ml_weighted_extra64_1layer256_si_test_results.tsv")
nnet_score12 = get_score("../sub/output_2e5_4epo_176ml_weighted_extra64_1layer256_test_results.tsv")
nnet_score13 = get_score("../sub/output_2e5_5epo_156ml_weighted_extra33_1layer156_si_test_results.tsv")
nnet_score14 = get_score("../sub/output_2e5_5epo_156ml_weighted_extra33_1layer156_test_results.tsv")
nnet_score15 = get_score("../sub/output_2e5_5epo_156ml_weighted_extra64_1layer156_si_test_results.tsv")
nnet_score16 = get_score("../sub/output_2e5_5epo_156ml_weighted_extra64_1layer156_test_results.tsv")
nnet_score17 = get_score("../sub/output_2e5_5epo_168ml_weighted_extra64_1layer256_si_test_results.tsv")
nnet_score18 = get_score("../sub/output_2e5_5epo_168ml_weighted_extra64_1layer256_test_results.tsv")

print("(*) get tree scores")
tree_score1 = get_score("../sub/cpu_test_results_cb_d6_it1000_lr002_l2lr3_rsm1_bc128_mcc2_rs24_knn31_extra33_boc_svd168_weighted_cv10_bestmodel.tsv")
tree_score2 = get_score("../sub/cpu_test_results_cb_d6_it1000_lr002_l2lr3_rsm1_bc128_mcc2_rs24_knn31_extra33_boc_svd168_weighted_cv5_bestmodel.tsv")
tree_score3 = get_score("../sub/cpu_test_results_lgb_nl32_mcs32_md5_lr003_ne1000_ff08_bf08_rs2424_knn31_sumbow_svd124.tsv")
tree_score4 = get_score("../sub/cpu_test_results_lgb_nl32_mcs32_md7_lr003_ne1000_ff08_bf08_rs2424_knn31_sumbow.tsv")
tree_score5 = get_score("../sub/cpu_test_results_lgb_nl32_mcs32_md8_lr002_ne600_ff08_bf08_rs24_knn31_extra33_boc_svd168.tsv")
tree_score6 = get_score("../sub/cpu_test_results_lgb_nl32_mcs32_md8_lr002_ne600_ff08_bf08_rs24_knn31_extra33_boc_svd168_cv10.tsv")
tree_score7 = get_score("../sub/cpu_test_results_lgb_nl32_mcs32_md8_lr005_ne1000_ff05_bf05_rs4200_knn31.tsv")
tree_score8 = get_score("../sub/cpu_test_results_lgb_nl32_md8_lr005_ne1000_ff05_bf05_rs4200_svd168.tsv")
tree_score9 = get_score("../sub/gpu_test_results_lgb_weighted_cv10_nl32_md7_lr005_ne1000_ff05_bf05_rs4201_stopword_mdil32.tsv")
lr_score1 = get_score("../sub/cpu_test_results_lr_C1_sag_rs42_knn31_extra33_boc_svd168_cv5.tsv")

print("(*) ensembling 9 trees")
scores = tree_score1 + tree_score2 + tree_score3 + tree_score4 + tree_score5 + tree_score6 + tree_score7 + tree_score8 + tree_score9
scores /= 9
scores = np.argmax(scores, axis=1)
sub = pd.DataFrame({'Id': ids, 'Pred': scores})
labels = {0: 'agreed', 1: 'disagreed', 2: 'unrelated'}
sub['Category'] = sub.Pred.map(labels)
sub[['Id', 'Category']].to_csv(f"../sub/ens_9trees_v1.csv", index=None)

print("(*) ensembling 9 trees and lr")
scores = tree_score1 + tree_score2 + tree_score3 + tree_score4 + tree_score5 + tree_score6 + tree_score7 + tree_score8 + tree_score9 + lr_score1
scores /= 10
scores = np.argmax(scores, axis=1)
sub = pd.DataFrame({'Id': ids, 'Pred': scores})
labels = {0: 'agreed', 1: 'disagreed', 2: 'unrelated'}
sub['Category'] = sub.Pred.map(labels)
sub[['Id', 'Category']].to_csv(f"../sub/ens_9trees_lr_v1.csv", index=None)

print("(*) ensembling 18 nnets")
scores = nnet_score1 + nnet_score2 + nnet_score3 + nnet_score4 + nnet_score5 + nnet_score6 + nnet_score7 + nnet_score8 + nnet_score9 +\
         nnet_score10 + nnet_score11 + nnet_score12 + nnet_score13 + nnet_score14 + nnet_score15 + nnet_score16 + nnet_score17 + nnet_score18
scores /= 18
scores = np.argmax(scores, axis=1)
sub = pd.DataFrame({'Id': ids, 'Pred': scores})
labels = {0: 'agreed', 1: 'disagreed', 2: 'unrelated'}
sub['Category'] = sub.Pred.map(labels)
sub[['Id', 'Category']].to_csv(f"../sub/ens_18nnets_v1.csv", index=None)

print("(*) ensembling 18 nnets 9 trees 1 lr")
scores = nnet_score1 + nnet_score2 + nnet_score3 + nnet_score4 + nnet_score5 + nnet_score6 + nnet_score7 + nnet_score8 + nnet_score9 +\
         nnet_score10 + nnet_score11 + nnet_score12 + nnet_score13 + nnet_score14 + nnet_score15 + nnet_score16 + nnet_score17 + nnet_score18 +\
         tree_score1 + tree_score2 + tree_score3 + tree_score4 + tree_score5 + tree_score6 + tree_score7 + tree_score8 + tree_score9 + lr_score1

scores /= 28
scores = np.argmax(scores, axis=1)
sub = pd.DataFrame({'Id': ids, 'Pred': scores})
labels = {0: 'agreed', 1: 'disagreed', 2: 'unrelated'}
sub['Category'] = sub.Pred.map(labels)
sub[['Id', 'Category']].to_csv(f"../sub/final_score.csv", index=None)

print("DONE.")

