'''
Training and testing the XG Boost model
'''

import xgboost as xgb
import numpy as np
import Sparse_Matrix_IO as smio
from sklearn import metrics
import os
import csv
from sys import maxint


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(month, day, hour=-1, ratio=-1):
    print "Reading Data..."
    root = "/mnt/rips2/2016"
    p0 = str(month).rjust(2,'0')
    p1 = str(day).rjust(2,'0')
    addr_day = os.path.join(root,p0,p1)
    p2 = str(hour).rjust(2,'0')
    if ratio == -1:
        if hour == -1:
            data = os.path.join(addr_day,'day_samp_newer.npy')
        else:
            data = os.path.join(addr_day,p2, 'output_newer.npy')
        d = smio.load_sparse_csr(data)
        return d
    else:
        ratio = ratio/float(100)
        if hour == -1:
            data_pos = os.path.join(addr_day,'PosNeg/day_samp__newer_large_pos.npy')
            data_neg = os.path.join(addr_day,'PosNeg/day_samp_newer_large_neg.npy')
        else:
            data_pos = os.path.join(addr_day,p2, 'output_pos_newer.npy')
            data_neg = os.path.join(addr_day,p2, 'output_neg_newer.npy')
        pos_matrix = smio.load_sparse_csr(data_pos)
        n = np.size(pos_matrix,axis=0)
        neg = int(ratio*n)
        neg_matrix = smio.load_sparse_csr(data_neg)[:neg,:]
        matrix = np.vstack((neg_matrix, pos_matrix))
        np.random.shuffle(matrix)
        return matrix




def format_data(month, day, hour = -1, ratio = -1):
    d = get_data(month, day, hour, ratio)
    X = d[:, :-1]
    y = d[:, -1]
    m = int(np.size(X,1))   # Number of columns
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m)
    return X, y


def net_sav(r,f):
    sav = -5200+123000*f-600000*(1-r)
    return sav


def process_data(month, day, hour = -1, ratio = -1):
    print "Processing %s/%s/%s" % (month, day, hour)
    data, label = format_data(month, day, hour, ratio)
    matrix = xgb.DMatrix(data, label=label)
    return matrix, label


def train_model(month, day):
    dtrain, train_label = process_data(month, day -1)

    p = sum(train_label)   # number of ones
    n = len(train_label) - p   # number of zeros
    # Setting parameters
    param = {'booster': 'gbtree',   # Tree, not linear regression
             'objective': 'binary:logistic',   # Output probabilities
             'eval_metric': ['auc'],
             'bst:max_depth': 5,   # Max depth of tree
             'bst:eta': .05,   # Learning rate (usually 0.01-0.2)
             'bst:gamma': 8.5,   # Larger value --> more conservative
             'bst:min_child_weight': 1,
             'scale_pos_weight': n/float(p),   # Often num_neg/num_pos
             'subsample': .8,
             'silent': 1,   # 0 outputs messages, 1 does not
             'save_period': 0,   # Only saves last model
             'nthread': 16,   # Number of cores used; otherwise, auto-detect
             'seed': 30}
    evallist = [(dtrain,'train')]   # Want to train until eval error stops decreasing
    num_round = 500   # Number of rounds of training
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    verbose_eval=50)
    return bst



cut = 0.1
if __name__ == "__main__":
    for month in range(6, 7):
        for day in range(19, 32):
            try:
                bst = train_model(month, day)
                for hour in range(0, 24):

                    dtest, test_label = process_data(month, day, hour)

                    pred_prop = bst.predict(dtest)
                    pred = pred_prop > cut

                    optimal_results = [-maxint,0]   # o_r[1] is optimal cutoff, +/- 0.01
                    for cutoff in range(0, 41):
                        temp_cut = cutoff/float(100)
                        temp_pred = pred_prop > temp_cut
                        savings = net_sav(metrics.recall_score(test_label, temp_pred),
                                          sum(np.logical_not(temp_pred))/float(len(temp_pred)))
                        if savings > optimal_results[0]:
                            optimal_results[0] = savings
                            optimal_results[1] = temp_cut

                    output_file = "/home/ubuntu/Krishan/Results/OptimalXGB-Bin.csv"
                    if not os.path.isfile(output_file):
                        with open(output_file, "a") as file:
                            wr = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
                            wr.writerow(["Day", "Hour", "Recall", "Filter Rate",
                                         "Savings", "Cutoff",
                                         "Optimal Savings", "Optimal Cutoff"])
                    with open(output_file, "a") as file:
                        results = [0,0,0,0,0,0,0,0]
                        results[0] = day
                        results[1] = hour
                        results[2] = metrics.recall_score(test_label, pred)
                        results[3] = sum(np.logical_not(pred))/float(len(pred))
                        results[4] = net_sav(results[2], results[3])
                        results[5] = cut
                        results[6] = optimal_results[0]
                        results[7] = optimal_results[1]
                        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
                        wr.writerow(results)

                    cut = optimal_results[1]
            except:
                print "%s/%s not in range" %(month, day)
                pass