import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import roc_auc_score
import os
import csv
from sklearn import cross_validation
import operator
import pandas as pd
from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder
import sys # to print in real time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import time
import Get_Data_Rodrigo as gdr  # to get Data from AWS
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV

# Feature Selection source:
## https://www.kaggle.com/sureshshanmugam/santander-customer-satisfaction/xgboost-with-feature-selection/code
# Another source to explore:
# https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code
# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html

def get_data(data):
    printit(  "Reading Data...")
    temp = np.load(data)
    d = csr_matrix((temp['data'], temp['indices'], temp['indptr']), shape=temp['shape'], dtype=float).toarray()
    return d

def format_data(data):
    d = get_data(data)
    m = int(np.size(d,1))   # Number of columns
    n = int(np.size(d,0))   # Number of rows
    printit(  "There are %s data points, each with %s features" % (n, m-1))
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y

def printit(text):
    print text
    sys.stdout.flush()

def recall(preds, dtrain):
    # return a pair metric_name, result
    cutoff = 0.1
    labels = dtrain.get_label()
    # since preds are margin(before logistic transformation, cutoff at 0)
    preds_bin = np.greater(preds, np.zeros(len(labels))+cutoff)
    return "recall", recall_score(labels, preds_bin)

def GetData(month, day, inn): ## Input Weiyi-formatted Data
    """
    Takes the data from a given day/month and outputs a numpy array
    """
    #root = "/mnt/rips2/2016"  #for AWS
    root = "/home/rmendoza/Documents/Data/DataXGB_jul28"  #for local maschine
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    #dataroot = os.path.join(root,p0,p1,"day_samp_bin.npy")  # for AWS
    binName = 'day_samp_new_'+p0+p1+'.npy'  # for local maschine
    dataroot = os.path.join(root,binName)   # for local maschine
    printit(  "Reading Data...")
    train_data, train_label = format_data(dataroot)
    ## FOR AWS
    # addr = os.path.join(root, str(month).rjust(2, "0"), str(day).rjust(2, "0"))  # FOR AWS
    # if inn:  # FOR AWS
    #     train_data, train_label = gdr.get(addr, ratio=7)#), mode="res-25") # AWS
    # else:  # FOR AWS
    #     X_test, y_test = gdr.get(addr)  # FOR AWS
    printit( "Finished reading data file")
    return train_data, train_label

def create_feature_map(features):
    outfile = open('xgb2.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def netSav(r,f):
    netSaving = -5200+127000*f-850000*(1-r)
    return netSaving

def evalModel(y_true, y_pred, start_time, cut):
    res = [0, 0, 0, 0, 0, 0, 0]    #res contains: [ NetSav_scaled, AUC_score, recall, filtering_rate, cut, timer, netSavings]
    y = y_pred > cut   # If y values are greater than the cutoff
    recall = metrics.recall_score(y_true, y)
    print 'got Recall'
    filter_rate = sum(np.logical_not(y))/float(len(y_pred))
    print 'got FitlerRate'
    timer = time.time() - start_time
    res[0] = recall*6.7+filter_rate
    res[1] = metrics.roc_auc_score(test_label, y)
    res[2] = recall
    res[3] = filter_rate
    res[4] = cut
    res[5] = timer
    res[6] = netSav(recall, filter_rate)
    return res


if __name__ == "__main__":
    # with open("/home/ubuntu/Rodrigo/test_XGBDmatrx_AWS_featureImportance.ods", "wr") as output_file:  #AWS
    with open("/home/rmendoza/Desktop/test_XGBDmatrx_AWS_featureImportance.ods", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        l = ['month','TrainDay','testDay','recall','filtered', 'time','NetSavings']
        wr.writerow(l)
        for diff in [1]:  #1,7  # as for now, only [1] means test on next day
            for month in range(6,7): #5,7    # as for now, only range(6,7) means june
                for day in range(4,5): #1,32  # as for now, only range(4,5) means 1st day
                    printit(  '------------------------------------------------')
                    printit(  '------------------------------------------------')
                    printit(  ['month = ', month,' and day = ',  day])
                    # try:
                    start_time = time.time()
                    # Inputting training set
                    X, y = GetData(month, day, 'true')
                    dtrain = xgb.DMatrix(X, label=y)
                    # Inputting testing set
                    test_data, test_label = GetData(month, day+diff,'false')
                    dtest = xgb.DMatrix(test_data, label=test_label)
                    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=1301, stratify=y, test_size=0.3)
                    print 'After XTrain'
                    #Setting parameters
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
                    evallist = [(dtrain,'train'), (dtest,'eval')]
                    num_round = 1000   # Number of rounds of training, increasing this increases the range of output values
                    clf = xgb.XGBClassifier(max_depth = 5,
                                            n_estimators=1525,
                                            learning_rate=0.02,
                                            nthread=4,
                                            subsample=0.95,
                                            colsample_bytree=0.85,
                                            seed=4242)
                    # bst = xgb.train(param,
                    #                 dtrain,
                    #                 num_round,
                    #                 evallist,
                    #                 early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
                    selector = RFECV(clf, step=1, cv=5)
                    selector = selector.fit(X_train, y_train)
                    print 'Selector fit...'
                    # clf.dump_model('/home/rmendoza/Desktop/xgb_june_04_to_05_v2.txt')
                    # bst.save_model('/home/rmendoza/Desktop/xgbtemp.model')
                    y_pred = selector.predict_proba(test_data)
                    cut = 0.1

                    results = [0, 0, 0, 0, 0, 0, 0]
                    for cutoff in range(10, 15):
                        cut = cutoff/float(100)   # Cutoff in decimal form
                        y = y_pred > cut   # If y values are greater than the cutoff
                        recall = metrics.recall_score(test_label, y)
                        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
                        filter_rate = sum(np.logical_not(y))/float(len(y_pred))
                        if recall*6.7+filter_rate > results[0]:
                            timer = time.time() - start_time
                            results = evalModel(test_label, y_pred, start_time, cut)
                    print results


                    # except:
                    #     pass
                    #     print 'Failure x-P, something crashed.'
                    # print '_____________________________________________________________________'
                    # print '_____________________________________________________________________'