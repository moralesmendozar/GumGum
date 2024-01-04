import os
import time, gc
import numpy as np
import xgboost as xgb
from sklearn import metrics
import Sparse_Matrix_IO as smio


def get_data(month, day):
    root = "/mnt/rips2/2016/random_samples"
    addr_in = os.path.join(root,
                           str(month).rjust(2, "0"),
                           str(day).rjust(2, "0"),
                           "day_samp_new_large.npy")
    with open(addr_in, "r") as file_in:
        data = smio.load_sparse_csr(file_in)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def search_cut(prob, y_test):
    score = 0
    recall_best = 0
    filter_rate_best = 0
    net_savings_best = 0
    cut_best = 0
    for cutoff in range(0, 31):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y_pred = prob > cut   # If y values are greater than the cutoff
        recall = metrics.recall_score(y_test, y_pred)
        filter_rate = sum(np.logical_not(y_pred))/float(len(prob))
        if recall*6.7+filter_rate > score:
            score = recall*6.7+filter_rate
            recall_best = recall
            filter_rate_best = filter_rate
            net_savings_best = -5200+127000*filter_rate-850000*(1-recall)
            cut_best = cut
    return score, recall_best, filter_rate_best, cut_best, net_savings_best


def train(param, num_round):
    X_train, y_train = get_data(data[0], data[1])
    data_train = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(param, data_train, num_round)


def test(bst):
    X_test, y_test = get_data(data[0], data[1]+1)
    data_test = xgb.DMatrix(X_test, label=y_test)
    return bst.predict(data_test), y_test


param = {'booster':'gbtree',   # Tree, not linear regression
         'objective':'binary:logistic',   # Output probabilities
         'eval_metric':['auc'],
         'bst:max_depth':5,   # Max depth of tree
         'bst:eta':.1,   # Learning rate (usually 0.01-0.2)
         'bst:gamma':0,   # Larger value --> more conservative
         'bst:min_child_weight':1,
         'scale_pos_weight':30,   # Often num_neg/num_pos
         'subsample':.8,
         'silent':1,   # 0 outputs messages, 1 does not
         'save_period':0,   # Only saves last model
         'nthread':6,   # Number of cores used; otherwise, auto-detect
         'seed':25}
num_round = 100   # Number of rounds of training, increasing this increases the range of output values

data = (6, 19)
print "Training"
bst = train(param, num_round)
print "Done Training"
gc.collect()
print "Testing"
prob, y_test = test(bst)
print "Done Testing"

print search_cut(prob,y_test)
