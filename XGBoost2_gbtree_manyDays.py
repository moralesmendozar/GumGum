import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, recall_score
import os
import csv
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

def GetData(month, day): ## Input Weiyi-formatted Data
    """
    Takes the data from a given day/month and outputs a numpy array
    :param month:
    :param day:
    :return:
    """
    #root = "/mnt/rips2/2016"  #for AWS
    root = "/home/rmendoza/Documents/Data/DataXGB_jul28"  #for local maschine
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    #dataroot = os.path.join(root,p0,p1,"day_samp_bin.npy")  # for AWS
    #binName = 'day_samp_bin'+p0+p1+'.npy'  #for local maschine #old data
    binName = 'day_samp_new_'+p0+p1+'.npy'## New data
    dataroot = os.path.join(root,binName)   #for local maschine
    print "Reading Data..."
    train_data, train_label = format_data(dataroot)

    #temp = np.load(dataroot)  #old code
    #Data = csr_matrix((  temp['data'], temp['indices'], temp['indptr']),shape = temp['shape'], dtype=float).toarray()

    print "Finished reading data file"
    return train_data, train_label

def netSav(r,f):
    netSaving = -5200+127000*f-850000*(1-r)
    return netSaving

if __name__ == "__main__":
    with open("/home/rmendoza/Desktop/resultsXGB_1.ods", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        l = ['month','day','cutoff','recalll','filtered']
        wr.writerow(l)
        for diff in [1]:  #1,7  # as for now, only [1] means test on next day
            for month in range(6,7): #5,7    # as for now, only range(6,7) means june
                for day in range(4,26): #1,32  # as for now, only range(4,5) means 1st day
                    print '------------------------------------------------'
                    print '------------------------------------------------'
                    print 'month = ', month,' and day = ',  day
                    try:
                        # Inputting training and testing set
                        train_data, train_label = GetData(month, day)
                        dtrain = xgb.DMatrix(train_data, label=train_label)
                        test_data, test_label = GetData(month, day+diff)
                        dtest = xgb.DMatrix(test_data, label=test_label)

                        # Setting parameters
                        param = {'booster':'gbtree',#'gblinear',   # Tree, not linear regression
                                 'objective':'binary:logistic',   # Output probabilities
                                 'bst:max_depth': 4,   # Max depth of tree
                                 'bst:eta':0.5,   # Learning rate (usually 0.01-0.2)
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
                                        early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
                        bst.dump_model('/home/rmendoza/Desktop/dump.raw2.txt')

                        y_true = test_label
                        y_pred = bst.predict(dtest)
                        a = 0.0001
                        b = 0.5
                        rangeCutoffs = np.linspace(a,b,100,endpoint = True)
                        previous = 1
                        #rangeCutoffs = range(1, 10)
                        recCutoff = 0.94
                        for cutoff in rangeCutoffs:
                            cut = cutoff/float(10)   # Cutoff, checking from .1 thru .9
                            ypred = np.greater(y_pred, np.zeros(len(y_true))+cut)   # If y values are greater than the cutoff
                            recalll = recall_score(y_true, ypred)
                            if recalll >=recCutoff or previous == 1:
                                cf = confusion_matrix(y_true,ypred)
                                n = len(y_true)
                                #filtered = (cf[0,0]+cf[1,0])/float(n)
                                filtered = (cf[0,0])/float(n)
                                if filtered >= 0.35:
                                    print "Cutoff is: %s" % cut
                                    print "Recall is: %s" % recalll
                                    print 'Filtering is = ', filtered
                                    print cf
                                    l = [month,day,cutoff,recalll,filtered]
                                    wr.writerow(l)
                                #else:
                                    #print 'Bad recall, not worth reporting'
                                if recalll < recCutoff:
                                    previous = 0
                                else:
                                    previous = 1
                            else:
                                previous = 0
                        #xgb.plot_importance(bst, xlabel="test")
                        #xgb.plot_tree(bst, num_trees=2)
                        #wr = csv.writer(f,delimiter="\n")

                    except:
                        pass
                        print 'failure, no such day'
                    print '_____________________________________________________________________'
                    print '_____________________________________________________________________'