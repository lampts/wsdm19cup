import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from fuzzywuzzy import fuzz
import distance
import json, string, re
from collections import defaultdict
import networkx as nx
import jieba
import time as time

def preprocess(x):
        x = str(x).lower()
        re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        x = re_tok.sub(r' \1 ', x)
        return x
    
def extract_bow(train_file="../data/train.csv", test_file="../data/test.csv", analyzer='char', ngram_range=(1, 1), stop_words=[], min_df=1, max_features=10000,use_idf=True, to_preprocess=True):
    """return 4 tensors of train_q1,q2 and test_q1,q2"""
    df_train = pd.read_csv(train_file, usecols=['title1_zh', 'title2_zh']).fillna("")
    df_test = pd.read_csv(test_file, usecols=['title1_zh', 'title2_zh']).fillna("")
    df = pd.DataFrame()
    df['text'] = pd.Series(df_train['title1_zh'].tolist() + df_train['title2_zh'].tolist() + df_test['title1_zh'].tolist() + df_test['title2_zh'].tolist()).unique()
        
    if to_preprocess:
        df['text'] = df['text'].map(lambda x: preprocess(x))
        df_train['title1_zh'] = df_train['title1_zh'].apply(preprocess)
        df_train['title2_zh'] = df_train['title2_zh'].apply(preprocess)
        df_test['title1_zh'] = df_test['title1_zh'].apply(preprocess)
        df_test['title2_zh'] = df_test['title2_zh'].apply(preprocess)
        
    if analyzer == 'char':
        vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words, min_df=min_df, max_features=max_features, use_idf=use_idf)
    else:
        vect = TfidfVectorizer(analyzer=analyzer, tokenizer=jieba.cut, ngram_range=ngram_range, stop_words=stop_words, min_df=min_df, max_features=max_features, use_idf=use_idf)
    vect.fit(df["text"].tolist())
    return vect.transform(df_train.title1_zh),vect.transform(df_train.title2_zh), vect.transform(df_test.title1_zh),vect.transform(df_test.title2_zh), vect

class NLPExtractor():
    def __init__(self, stopwords):
        self.STOP_WORDS = stopwords
        self.SAFE_DIV = 0.0001
        self.MAX_SEQUENCE_LENGTH = 25
        
    def __calc_distances__(self, v1s, v2s, is_sparse=True):
        if is_sparse:
            dcosine     = np.array([cosine(x.toarray(), y.toarray())       for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dcityblock  = np.array([cityblock(x.toarray(), y.toarray())    for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dcanberra  = np.array([canberra(x.toarray(), y.toarray())     for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            deuclidean = np.array([euclidean(x.toarray(), y.toarray())    for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dminkowski  = np.array([minkowski(x.toarray(), y.toarray(), 3) for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dbraycurtis = np.array([braycurtis(x.toarray(), y.toarray())   for (x, y) in zip(v1s, v2s)]).reshape((-1,1))

            dskew_q1 = [skew(x.toarray().ravel()) for x in v1s]
            dskew_q2 = [skew(x.toarray().ravel()) for x in v2s]
            dkur_q1  = [kurtosis(x.toarray().ravel()) for x in v1s]
            dkur_q2  = [kurtosis(x.toarray().ravel()) for x in v2s]

            dskew_diff = np.abs(np.array(dskew_q1) - np.array(dskew_q2)).reshape((-1,1))
            dkur_diff  = np.abs(np.array(dkur_q1) - np.array(dkur_q2)).reshape((-1,1))
        else:
            dcosine     = np.array([cosine(x, y)       for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dcityblock  = np.array([cityblock(x, y)    for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dcanberra  = np.array([canberra(x, y)     for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            deuclidean = np.array([euclidean(x, y)    for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dminkowski  = np.array([minkowski(x, y, 3) for (x, y) in zip(v1s, v2s)]).reshape((-1,1))
            dbraycurtis = np.array([braycurtis(x, y)   for (x, y) in zip(v1s, v2s)]).reshape((-1,1))

            dskew_q1 = [skew(x) for x in v1s]
            dskew_q2 = [skew(x) for x in v2s]
            dkur_q1  = [kurtosis(x) for x in v1s]
            dkur_q2  = [kurtosis(x) for x in v2s]

            dskew_diff = np.abs(np.array(dskew_q1) - np.array(dskew_q2)).reshape((-1,1))
            dkur_diff  = np.abs(np.array(dkur_q1) - np.array(dkur_q2)).reshape((-1,1))
        return np.hstack((dcosine,dcityblock,dcanberra,deuclidean,dminkowski,dbraycurtis,dskew_diff,dkur_diff))
    
    def extract_distances_sparse(self, train_file="../data/train.csv", test_file="../data/test.csv"):
        v1s,v2s,v1s_test, v2s_test, vect = extract_bow(train_file=train_file, test_file=test_file, analyzer='word', ngram_range=(1,2), min_df=2, stop_words=self.STOP_WORDS)
        return self.__calc_distances__(v1s,v2s), self.__calc_distances__(v1s_test,v2s_test)
    
    def __is_numeric__(self,s):
        return any(i.isdigit() for i in s)
    
    def __preprocess__(self,x):
        x = str(x).lower()
        re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        x = re_tok.sub(r' \1 ', x)
        return x
    
    def __prepare__(self,q):
        q = self.__preprocess__(q)
        new_q = []
        surplus_q = []
        numbers_q = []
        new_xitrum = True
        for w in list(jieba.cut(q))[::-1]:
            if w not in self.STOP_WORDS:
                if new_xitrum:
                    new_q = ["__xitrum__"] + new_q
                    new_xitrum = False
                if self.__is_numeric__(w):
                    numbers_q = [w] + numbers_q
                else:
                    surplus_q = [w] + surplus_q
            else:
                new_xitrum = True
            if len(new_q) == self.MAX_SEQUENCE_LENGTH:
                break
        new_q = " ".join(new_q)
        return new_q, set(surplus_q), set(numbers_q)

    ### jaccard
    def extract_extra_features(self,df):
        q1s = np.array([""] * len(df), dtype=object)
        q2s = np.array([""] * len(df), dtype=object)
        features = np.zeros((len(df), 4))

        for i, (q1, q2) in enumerate(list(zip(df["title1_zh"], df["title2_zh"]))):
            q1s[i], surplus1, numbers1 = self.__prepare__(q1)
            q2s[i], surplus2, numbers2 = self.__prepare__(q2)
            features[i, 0] = len(surplus1.intersection(surplus2))
            features[i, 1] = len(surplus1.union(surplus2))
            features[i, 2] = len(numbers1.intersection(numbers2))
            features[i, 3] = len(numbers1.union(numbers2))

        return q1s, q2s, features

    def __get_token_features__(self,q1,q2):
        token_features = [0.0]*10

        q1_tokens = self.__preprocess__(q1).split()
        q2_tokens = self.__preprocess__(q2).split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])

        q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])

        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features


    def __get_longest_substr_ratio__(self,a,b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def extract_stat_features(self,df):
        df["title1_zh"] = df["title1_zh"].fillna("").apply(self.__preprocess__)
        df["title2_zh"] = df["title2_zh"].fillna("").apply(self.__preprocess__)

        print("token features...")
        token_features = df.apply(lambda x: self.__get_token_features__(x["title1_zh"], x["title2_zh"]), axis=1)
        df["cwc_min"]       = list(map(lambda x: x[0], token_features))
        df["cwc_max"]       = list(map(lambda x: x[1], token_features))
        df["csc_min"]       = list(map(lambda x: x[2], token_features))
        df["csc_max"]       = list(map(lambda x: x[3], token_features))
        df["ctc_min"]       = list(map(lambda x: x[4], token_features))
        df["ctc_max"]       = list(map(lambda x: x[5], token_features))
        df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        df["first_word_eq"] = list(map(lambda x: x[7], token_features))
        df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        df["mean_len"]      = list(map(lambda x: x[9], token_features))

        print("fuzzy features..")
        df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["title1_zh"], x["title2_zh"]), axis=1)
        df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["title1_zh"], x["title2_zh"]), axis=1)
        df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["title1_zh"], x["title2_zh"]), axis=1)
        df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["title1_zh"], x["title2_zh"]), axis=1)
        df["longest_substr_ratio"]  = df.apply(lambda x: self.__get_longest_substr_ratio__(x["title1_zh"], x["title2_zh"]), axis=1)
        
        if 'label' in df.columns.tolist():
            return df.drop(["title1_zh", "title2_zh", "label"], axis=1).values
        else:
            return df.drop(["title1_zh", "title2_zh"], axis=1).values
        
class GraphFeatureExtractor():
    def __init__(self, n_cores=10, freq_upper_bound=10, neighbor_upper_bound=5):
        self.NB_CORES = n_cores
        self.FREQ_UPPER_BOUND = freq_upper_bound
        self.NEIGHBOR_UPPER_BOUND = neighbor_upper_bound

    def create_question_hash(self, train_df, test_df):
        train_qs = np.dstack([train_df["title1_zh"], train_df["title2_zh"]]).flatten()
        test_qs = np.dstack([test_df["title1_zh"], test_df["title2_zh"]]).flatten()
        all_qs = np.append(train_qs, test_qs)
        all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
        all_qs.reset_index(inplace=True, drop=True)
        question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
        return question_dict

    def get_hash(self, df, hash_dict):
        df["qid1"] = df["title1_zh"].map(hash_dict)
        df["qid2"] = df["title2_zh"].map(hash_dict)
        return df.drop(["title1_zh", "title2_zh"], axis=1)

    def get_kcore_dict(self,df):
        g = nx.Graph()
        g.add_nodes_from(df.qid1)
        edges = list(df[["qid1", "qid2"]].to_records(index=False))
        g.add_edges_from(edges)
        g.remove_edges_from(g.selfloop_edges())

        df_output = pd.DataFrame(data=list(g.nodes()), columns=["qid"])
        df_output["kcore"] = 0
        for k in range(2, self.NB_CORES + 1):
            ck = set(nx.k_core(g, k=k).nodes())
            print("kcore", k)
            df_output.ix[df_output.qid.isin(ck), "kcore"] = k

        return df_output.to_dict()["kcore"]

    def get_kcore_features(self, df, kcore_dict):
        df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
        df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
        return df


    def convert_to_minmax(self, df, col):
        sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
        df["min_" + col] = sorted_features[:, 0]
        df["max_" + col] = sorted_features[:, 1]
        return df.drop([col + "1", col + "2"], axis=1)

    def get_neighbors(self, train_df, test_df):
        neighbors = defaultdict(set)
        for df in [train_df, test_df]:
            for q1, q2 in zip(df["qid1"], df["qid2"]):
                neighbors[q1].add(q2)
                neighbors[q2].add(q1)
        return neighbors

    def get_neighbor_features(self, df, neighbors):
        common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
        min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
        df["common_neighbor_ratio"] = common_nc / min_nc
        df["common_neighbor_count"] = common_nc.apply(lambda x: min(x, self.NEIGHBOR_UPPER_BOUND))
        return df

    def get_freq_features(self, df, frequency_map):
        df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], self.FREQ_UPPER_BOUND))
        df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], self.FREQ_UPPER_BOUND))
        return df
    
def make_graph_feature(train_file="../data/train.csv", test_file="../data/test.csv"):
    ge = GraphFeatureExtractor()
    train_df = pd.read_csv(train_file, usecols=['title1_zh', 'title2_zh']).fillna("")
    test_df = pd.read_csv(test_file, usecols=['title1_zh', 'title2_zh']).fillna("")
    print("Hashing the questions...")
    question_dict = ge.create_question_hash(train_df, test_df)
    train_df = ge.get_hash(train_df, question_dict)
    test_df = ge.get_hash(test_df, question_dict)
    print("Number of unique questions:", len(question_dict))
    print("Calculating kcore features...")
    all_df = pd.concat([train_df, test_df])
    kcore_dict = ge.get_kcore_dict(all_df)
    train_df = ge.get_kcore_features(train_df, kcore_dict)
    test_df = ge.get_kcore_features(test_df, kcore_dict)
    train_df = ge.convert_to_minmax(train_df, "kcore")
    test_df = ge.convert_to_minmax(test_df, "kcore")
    print("Calculating common neighbor features...")
    neighbors = ge.get_neighbors(train_df, test_df)
    train_df = ge.get_neighbor_features(train_df, neighbors)
    test_df = ge.get_neighbor_features(test_df, neighbors)
    print("Calculating frequency features...")
    frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])), return_counts=True)))
    train_df = ge.get_freq_features(train_df, frequency_map)
    test_df = ge.get_freq_features(test_df, frequency_map)
    train_df = ge.convert_to_minmax(train_df, "freq")
    test_df = ge.convert_to_minmax(test_df, "freq")
    cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq"]
    return train_df.loc[:, cols].values, test_df.loc[:, cols].values


st = time.time()

print("(*) load data")
train = pd.read_csv("../data/train.csv", usecols=["title1_zh", "title2_zh", "label"]).fillna("")
test = pd.read_csv("../data/test.csv", usecols=["title1_zh", "title2_zh"]).fillna("")
y = train.label.values
print(f"Time {time.time() - st:.02f}s")

extractor = NLPExtractor(stopwords=[])
train_stat_features = extractor.extract_stat_features(train)
test_stat_features = extractor.extract_stat_features(test)
q1s_train, q2s_train, train_ex_features = extractor.extract_extra_features(train)
q1s_test, q2s_test, test_ex_features = extractor.extract_extra_features(test)
print(f"Time {time.time() - st:.02f}s")

ge, ge_test = make_graph_feature(train_file="../data/train.csv", test_file="../data/test.csv")
print(f"Time {time.time() - st:.02f}s")

dd, dd_test = extractor.extract_distances_sparse()
dd, dd_test = np.nan_to_num(dd), np.nan_to_num(dd_test)

X = hstack((csr_matrix(train_stat_features), csr_matrix(train_ex_features), csr_matrix(ge), csr_matrix(dd))).tocsr()
X_test = hstack((csr_matrix(test_stat_features), csr_matrix(test_ex_features), csr_matrix(ge_test), csr_matrix(dd_test))).tocsr()
print(X.shape, X_test.shape)
print(f"Time {time.time() - st:.02f}s")

print("(*) save data")
np.savetxt("../data/X_33.npy", np.nan_to_num(X.toarray()), fmt='%.6e')
np.savetxt("../data/X_test_33.npy", np.nan_to_num(X_test.toarray()), fmt='%.6e')
print(f"Time {time.time() - st:.02f}s")

X = np.nan_to_num(X.toarray())
X_test = np.nan_to_num(X_test.toarray())

print("(*) normalize data")
def normalize_x(x):
    m, s = x.mean(axis=0), x.std(axis=0)
    s = s + 1e-5
    x = (x - m ) / s
    return x, m, s

X, m, s= normalize_x(X)
X_test = (X_test - m) / s

X = np.nan_to_num(X)
X_test = np.nan_to_num(X_test)
np.savetxt("../data/X_33_norm.npy", X, fmt='%.6e')
np.savetxt("../data/X_test_33_norm.npy", X_test, fmt='%.6e')
print(f"Time {time.time() - st:.02f}s")
