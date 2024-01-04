import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import os
import csv
from math import sin
import time
from sys import maxint


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(data):
    print "Reading Data..."
    temp = np.load(data)
    d = csr_matrix((temp['data'], temp['indices'], temp['indptr']), shape=temp['shape'], dtype=float).toarray()
    return d


def format_data(data):
    d = get_data(data)
    m = int(np.size(d,1))   # Number of columns
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y


def net_sav(r,f):
    sav = -5200+127000*f-850000*(1-r)
    return sav


def process_data(month, day, hour):
    root = "/mnt/rips2/2016/"
    p1 = str(month).rjust(2, "0")
    p2 = str(day).rjust(2, "0")
    p3 = str(hour).rjust(2, "0")

    # Inputting training and testing set
    name = os.path.join(root, p1, p2, p3, "output_new.npy")
    print "Processing " + name
    data, label = format_data(name)
    matrix = xgb.DMatrix(data, label=label)
    return matrix, label


def train_model(month, day, hour):
    dtrain, train_label = process_data(month, day, hour)

    p = sum(train_label)   # number of ones
    n = len(train_label) - p   # number of zeros
    # Setting parameters
    param = {'booster': 'gbtree',   # Tree, not linear regression
             'objective': 'binary:logistic',   # Output probabilities
             'eval_metric': ['auc'],
             'bst:max_depth': 5,   # Max depth of tree
             'bst:eta': .2,   # Learning rate (usually 0.01-0.2)
             'bst:gamma': 8.5,   # Larger value --> more conservative
             'bst:min_child_weight': 1,
             'scale_pos_weight': n/float(p),   # Often num_neg/num_pos
             'subsample': .8,
             'silent': 1,   # 0 outputs messages, 1 does not
             'save_period': 0,   # Only saves last model
             'nthread': 6,   # Number of cores used; otherwise, auto-detect
             'seed': 30}
    evallist = [(dtrain,'train')]   # Want to train until eval error stops decreasing
    num_round = 250   # Number of rounds of training
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    verbose_eval=50)
    return bst


def dynamic_cutoff(hour):
    return 0.05*sin(hour/4 + 0.5) + 0.07


if __name__ == "__main__":
    for month in range(6, 7):
        for day in range(4, 26):
            for hour in range(12, 24):
                bst = train_model(month, day, hour)   # C_i-1
                if hour == 23:
                    dtest, test_label = process_data(month, day+1, 0)
                else:
                    dtest, test_label = process_data(month, day, hour+1)

                pred_prop = bst.predict(dtest)
                print "Maximum predicted probability is %s" % max(pred_prop)
                print "Minimum predicted probability is %s" % min(pred_prop)
                print "Average predicted probability is %s" % np.mean(pred_prop)
                print "First Quartile is %s" % np.percentile(pred_prop, 25)
                print "Median is %s" % np.percentile(pred_prop, 50)
                print "Third Quartile is %s" % np.percentile(pred_prop, 75)
                # cut = dynamic_cutoff(hour)
                cut = 0.10
                pred = pred_prop > cut

                optimal_results = [-maxint,0]   # o_r[1] is optimal cutoff, +/- 0.01
                for cutoff in range(0, 31):
                    temp_cut = cutoff/float(100)
                    temp_pred = pred_prop > temp_cut
                    savings = net_sav(metrics.recall_score(test_label, temp_pred),
                                      sum(np.logical_not(temp_pred))/float(len(temp_pred)))
                    if savings > optimal_results[0]:
                        optimal_results[0] = savings
                        optimal_results[1] = temp_cut

                output_file = "/home/ubuntu/Jonathan/xgb_numbers_test.csv"
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
                    wr = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
                    wr.writerow(results)
