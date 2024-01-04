import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, recall_score
import operator
import pandas as pd
from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(data):
    print "Reading Data..."
    temp = np.load(data)
    d = csr_matrix((temp['data'], temp['indices'], temp['indptr']), shape=temp['shape'], dtype=float).toarray()
    return d


def format_data(data):
    d = get_data(data)
    m = int(np.size(d,1))   # Number of columns
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m-1)
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y


def recall(preds, dtrain):
    cutoff = 0.1
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    preds_bin = np.greater(preds, np.zeros(len(labels))+cutoff)
    return "recall", recall_score(labels, preds_bin)


if __name__ == "__main__":
    # Inputting training and testing set
    train_data, train_label = format_data("/home/rmendoza/Documents/Data/DataXGB_jul28/day_samp_bin0604.npy")
    dtrain = xgb.DMatrix(train_data, label=train_label)
    test_data, test_label = format_data("/home/rmendoza/Documents/Data/DataXGB_jul28/day_samp_bin0605.npy")
    dtest = xgb.DMatrix(test_data, label=test_label)

    # Setting parameters
    param = {'booster':'gblinear',#'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'bst:max_depth':8,   # Max depth of tree
             'bst:eta':0.1,   # Learning rate (usually 0.01-0.2)
             'silent':1,   # 0 outputs messages, 1 does not
             'nthread':4}    # Number of cores used; otherwise, auto-detect
    #param['eval_metric'] = 'error'
    evallist = [(dtest,'eval'), (dtrain,'train')]

    num_round = 100   # Number of rounds of training, increasing this increases the range of output values
    #bst = xgb.train(param, dtrain, num_round, evallist, feval=recall, maximize=True)
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    early_stopping_rounds=100)   # If error doesn't decrease in n rounds, stop early
    bst.dump_model('/home/rmendoza/Desktop/dump.raw2.txt')

    y_true = test_label
    y_pred = bst.predict(dtest)
    a = 0.0001
    b = 5
    rangeCutoffs = np.linspace(a,b,100,endpoint = True)
    previous = 1
    #rangeCutoffs = range(1, 10)
    for cutoff in rangeCutoffs:
        cut = cutoff/float(10)   # Cutoff, checking from .1 thru .9
        ypred = np.greater(y_pred, np.zeros(len(y_true))+cut)   # If y values are greater than the cutoff
        recalll = recall_score(y_true, ypred)
        if recalll >=0.95 or previous == 1:
            cf = confusion_matrix(y_true,ypred)
            n = len(y_true)
            #filtered = (cf[0,0]+cf[1,0])/float(n)
            filtered = (cf[0,0])/float(n)
            if filtered >= 0.35:
                print "Cutoff is: %s" % cut
                print "Recall is: %s" % recalll
                print 'Filtering is = ', filtered
                print cf
            #else:
                #print 'Bad recall, not worth reporting'
            if recalll < 0.95:
                previous = 0
            else:
                previous = 1
        else:
            previous = 0

    #xgb.plot_importance(bst, xlabel="test")
    #xgb.plot_tree(bst, num_trees=2)