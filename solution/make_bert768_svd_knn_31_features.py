import numpy as np, pandas as pd
from make_knn_feats import load_bert_file, NearestNeighborsFeats
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import time as time
import gc

st = time.time()
print("(*) load bert 768-d pair encoding")
m = load_bert_file("../data/train_meanpool768_layer_2.jsonl", size=320552)
mtest = load_bert_file("../data/test_meanpool768_layer_2.jsonl", size=80126)
print(f"Time {time.time() - st:.02f}s")

print("(*) svd 168-d transformation")
svd = TruncatedSVD(n_components=168, random_state=42)
x_svd = svd.fit_transform(m)
xtest_svd = svd.transform(mtest)
print(f"Time {time.time() - st:.02f}s")

del m, mtest
gc.collect()

print("(*) load raw data")
train = pd.read_csv("../data/train.csv", usecols=["title1_zh", "title2_zh", "label"]).fillna("")
y = train.label.values
y2 = np.array([0 if i=='agreed' else 1 if i == 'disagreed' else 2 for i in y])
print(f"Time {time.time() - st:.02f}s")

print("(*) make knn features")
NNF = NearestNeighborsFeats(n_jobs=32, k_list=[3,8,32], metric='minkowski')
NNF.fit(x_svd, y2)

Xmin_test = NNF.predict(xtest_svd)
print("test shape", Xmin_test.shape)

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
NNF = NearestNeighborsFeats(n_jobs=32, k_list=[3,8,32], metric='minkowski')
Xmin = cross_val_predict(NNF,x_svd,y2,cv=skf,n_jobs=1)
print("train shape", Xmin.shape)
print(f"Time {time.time() - st:.02f}s")

print("(*) save data")
np.savetxt("../data/Xmin.npy", np.nan_to_num(Xmin), fmt='%.6e')
np.savetxt("../data/Xmin_test.npy", np.nan_to_num(Xmin_test), fmt='%.6e')
print(f"Time {time.time() - st:.02f}s")

Xmin = np.nan_to_num(Xmin)
Xmin_test = np.nan_to_num(Xmin_test)

print("(*) normalize data")
def normalize_x(x):
    m, s = x.mean(axis=0), x.std(axis=0)
    s = s + 1e-5
    x = (x - m ) / s
    return x, m, s

Xmin, m, s= normalize_x(Xmin)
Xmin_test = (Xmin_test - m) / s

Xmin = np.nan_to_num(Xmin)
Xmin_test = np.nan_to_num(Xmin_test)
np.savetxt("../data/Xmin_norm.npy", Xmin, fmt='%.6e')
np.savetxt("../data/Xmin_test_norm.npy", Xmin_test, fmt='%.6e')
print(f"Time {time.time() - st:.02f}s")
