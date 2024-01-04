import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import os
import csv
import time


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


def netSav(r,f):
    sav = -5200+127000*f-850000*(1-r)
    return sav


if __name__ == "__main__":
    root = "/mnt/rips2/2016/"
    for month in range(6, 7):
        p1 = str(month).rjust(2, "0")
        for day in range(4, 26):
            p2 = str(day).rjust(2, "0")
            for hour in range(0, 24):
                p3 = str(hour).rjust(2, "0")
                p4 = str(hour+1).rjust(2, "0")
                try:
                    # Inputting training and testing set
                    train_data_name = os.path.join(root, p1, p2, p3, "output_new.npy")
                    train_data, train_label = format_data(train_data_name)
                    dtrain = xgb.DMatrix(train_data, label=train_label)

                    test_data_name = os.path.join(root, p1, p2, p4, "output_new.npy")
                    test_data, test_label = format_data(test_data_name)
                    dtest = xgb.DMatrix(test_data, label=test_label)

                    print "Working on " + train_data_name

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
                                    evallist)

                    root2 = "/home/ubuntu/Jonathan"
                    # model_name = os.path.join(root2, "xgb_model" + p1 + p2)
                    # bst.dump_model(model_name)
                    start = time.time()
                    y_pred = bst.predict(dtest)
                    elapsed = time.time() - start
                    with open(os.path.join(root2, "xgb_numbers_hourly.csv"), "a") as file:
                        # J score, AUC score, best recall, best filter rate, best cutoff
                        results = [0, 0, 0, 0, elapsed]
                        for cutoff in range(0, 31):
                            cut = cutoff/float(100)   # Cutoff in decimal form
                            y = y_pred > cut   # If y values are greater than the cutoff
                            recall = metrics.recall_score(test_label, y)
                            filter_rate = sum(np.logical_not(y))/float(len(y_pred))
                            if netSav(recall, filter_rate) > results[0]:
                                results[0] = netSav(recall, filter_rate)
                                results[1] = recall
                                results[2] = filter_rate
                                results[3] = cut
                        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
                        wr.writerow(results)
                except:
                    pass
