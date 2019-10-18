from sklearn.base import BaseEstimator, ClassifierMixin	
from sklearn.neighbors import NearestNeighbors	
from multiprocessing import Pool	
import numpy as np	

import json	
def load_bert_file(fname="../data/test_meanpool768_layer_2.jsonl", size=80126):	
    m = np.zeros((size, 768))	
    with open(fname) as f:	
        for line in f:	
            j = json.loads(line)	
            m[j['linex_index']] = np.array(j['features'])	
    return m	

class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):	
    '''	
        This class should implement KNN features extraction 	
    '''	
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):	
        self.n_jobs = n_jobs	
        self.k_list = k_list	
        self.metric = metric	

        if n_neighbors is None:	
            self.n_neighbors = max(k_list) 	
        else:	
            self.n_neighbors = n_neighbors	

        self.eps = eps        	
        self.n_classes_ = n_classes	

    def fit(self, X, y):	
        '''	
            Set's up the train set and self.NN object	
        '''	
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 	
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 	
                                      metric=self.metric, 	
                                      n_jobs=1, 	
                                      algorithm='brute' if self.metric=='cosine' else 'auto')	
        self.NN.fit(X)	

        # Store labels 	
        self.y_train = y	

        # Save how many classes we have	
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_	


    def predict(self, X):       	
        '''	
            Produces KNN features for every object of a dataset X	
        '''	
        if self.n_jobs == 1:	
            test_feats = []	
            for i in range(X.shape[0]):	
                test_feats.append(self.get_features_for_one(X[i:i+1]))	
        else:	
            '''	
                 *Make it parallel*	
                     Number of threads should be controlled by `self.n_jobs`  	
                     	
                     	
                     You can use whatever you want to do it	
                     For Python 3 the simplest option would be to use 	
                     `multiprocessing.Pool` (but don't use `multiprocessing.dummy.Pool` here)	
                     You may try use `joblib` but you will most likely encounter an error, 	
                     that you will need to google up (and eventually it will work slowly)	
                     	
                     For Python 2 I also suggest using `multiprocessing.Pool` 	
                     You will need to use a hint from this blog 	
                     http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html	
                     I could not get `joblib` working at all for this code 	
                     (but in general `joblib` is very convenient)	
                     	
            '''	

            # YOUR CODE GOES HERE	
            # test_feats =  # YOUR CODE GOES HERE	
            # YOUR CODE GOES HERE	
#             test_feats = Parallel(n_jobs=self.n_jobs)(delayed(self.get_features_for_one)(x) for x in X)	
            p = Pool(self.n_jobs)	
            test_feats = p.map(self.get_features_for_one,[[x] for x in X])	
            p.close()	
            p.join()	

            # assert False, 'You need to implement it for n_jobs > 1'	



        return np.vstack(test_feats)	


    def get_features_for_one(self, x):	
        '''	
            Computes KNN features for a single object `x`	
        '''	

        NN_output = self.NN.kneighbors(x)	

        # Vector of size `n_neighbors`	
        # Stores indices of the neighbors	
        # NN_output = distances,indices	
        neighs = NN_output[1][0]	

        # Vector of size `n_neighbors`	
        # Stores distances to corresponding neighbors	
        neighs_dist = NN_output[0][0] 	

        # Vector of size `n_neighbors`	
        # Stores labels of corresponding neighbors	
        neighs_y = self.y_train[neighs] 	

        ## ========================================== ##	
        ##              YOUR CODE BELOW	
        ## ========================================== ##	

        # We will accumulate the computed features here	
        # Eventually it will be a list of lists or np.arrays	
        # and we will use np.hstack to concatenate those	
        return_list = [] 	
	
        return return_list