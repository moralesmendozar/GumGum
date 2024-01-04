import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import csv
from sklearn.cross_validation import KFold #

import time, gc, os
try:
    import cPickle as pickle
except:
    import pickle

# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html
def get_data(month, day):
    root = "/mnt/rips2/2016/random_samples"
    addr_in = os.path.join(root,
                           str(month).rjust(2, "0"),
                           str(day).rjust(2, "0"),
                           "day_samp_large_newer.npy")
    with open(addr_in, "r") as file_in:
        loader = np.load(file_in)
        data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    y = data[:,-1].toarray()
    X = data[:, :-1]
    return X, y



def train(param, num_round, theta):
    X_train, y_train = get_data(data[0], data[1])
    k = int(theta*1000000)
    X_train, y_train = X_train[:k,:], y_train[:k]
    X_train, y_train = X_train[train_index], y_train[train_index]
    data_train = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(param, data_train, num_round)


def test(bst, theta, j):
    X_test, y_test = get_data(data[0], data[1]+j)
    k = int(theta*len(y_test))
    X_test, y_test = X_test[:k], y_test[:k]
    if j == 0:
        X_test, y_test = X_test[test_index], y_test[test_index]
    data_test = xgb.DMatrix(X_test, label=y_test)
    return search_cut(bst.predict(data_test), y_test)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def search_cut(prob, y_test):
    score = 0
    recall_best = 0
    filter_rate_best = 0
    net_savings_best = 0
    cut_best = 0
    for cutoff in range(0, 80):
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
    return [score, recall_best, filter_rate_best, cut_best, net_savings_best]



if __name__ == "__main__":
    # importance = pickle.load(open('/home/kbhalla/Desktop/pickle/features.p','rb'))
    # print importance
    # s = []
    # for key in importance:
    #     if key[1] > 0:
    #         s.append(key[0][1:])
    with open('/home/ubuntu/Krishan/Results/XGB-Learning-Curve-CV-MoreData.csv', 'w') as file:
        # Inputting training and testing set
        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
        data = (6, 19)
        for theta in np.linspace(0.01, 1, 100):
            kf = KFold(int(1000000*theta), n_folds=3, shuffle=True)
            results = [0]*11
            # Setting parameters
            i = 1
            for train_index, test_index in kf:
                eta = 0.05
                param = {'booster':'gbtree',   # Tree, not linear regression
                         'objective':'binary:logistic',   # Output probabilities
                         'eval_metric':['auc'],
                         'bst:max_depth':5,   # Max depth of tree
                         'bst:eta':eta,   # Learning rate (usually 0.01-0.2)
                         'bst:gamma':8.5,   # Larger value --> more conservative
                         'bst:min_child_weight':1,
                         'scale_pos_weight': 20,   # Often num_neg/num_pos
                         'subsample':.8,
                         'silent':1,   # 0 outputs messages, 1 does not
                         'save_period':0,   # Only saves last model
                         #'nthread':1,   # Number of cores used; otherwise, auto-detect
                         'seed':25,
                         'alpha': 0}

                num_round = int(250*(0.2/float(eta)))   # Number of rounds of training, increasing this increases the range of output values

                print "Training"
                bst = train(param, num_round, theta)
                print "Done Training"
                gc.collect()
                print "Testing on Training Set"
                results_train = test(bst, theta, 0)
                print "Done Testing on Training Set"
                gc.collect()
                print "Testing"
                results_test = test(bst, 1, 1)
                print "Done Testing"
                print "------------------"
                print theta
                print "------------------"
                temp = results_train+results_test+[theta]
                print temp
                print "Finished fold %s" % i
                i += 1
                for j in range(len(results)):
                    results[j]+=temp[j]
            results2 = [i/float(3) for i in results]
            print "---------------------------------------"
            print results2
            print "---------------------------------------"
            wr.writerow(results2)