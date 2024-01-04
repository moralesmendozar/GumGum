import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
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
    cutoff = 0.1
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    preds_bin = np.greater(preds, np.zeros(len(labels))+cutoff)
    return "recall", recall_score(labels, preds_bin)

#def GetData(month, day, inn): ## Input Formatted Data  #AWS
def GetData(month, day): ##    #LOCAL
    """
    Takes the data from a given day/month and outputs a numpy array
    :param month:
    :param day:
    :return:
    """
    #root = "/mnt/rips2/2016"  #for AWS
    #root = "/mnt/rips2/2016"  #for AWS
    root = "/home/rmendoza/Documents/Data/DataXGB_jul28"  #for local maschine
    #root = "/home/rmendoza/Documents/Data/DataReservoir_aug1"
    #root = "/mnt/rips2/2016"   #For AWS
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    #dataroot = os.path.join(root,p0,p1,"day_samp_bin.npy")  # for AWS
    #binName = 'day_samp_bin'+p0+p1+'.npy'  #for local maschine #old data
    binName = 'day_samp_new_'+p0+p1+'.npy'## New data
    dataroot = os.path.join(root,binName)   #for local maschine
    printit(  "Reading Data...")
    train_data, train_label = format_data(dataroot)
    #temp = np.load(dataroot)  #old code
    #Data = csr_matrix((  temp['data'], temp['indices'], temp['indptr']),shape = temp['shape'], dtype=float).toarray()
    ## FOR AWS
    #addr = os.path.join(root, str(month).rjust(2, "0"), str(day).rjust(2, "0")) #AWS
    #print "Reading Data..."
    #train_data, train_label = format_data(dataroot)  #local Maschine
    #if inn:
    #    train_data, train_label = gdr.get(addr, ratio=7)#), mode="res-25") # AWS
    #else:
    #    X_test, y_test = gdr.get(addr)
    printit( "Finished reading data file")
    return train_data, train_label

def netSav(r,f):
    netSaving = -5200+127000*f-850000*(1-r)
    return netSaving

if __name__ == "__main__":
    #with open("/home/ubuntu/Rodrigo/test_XGBDmatrx_AWS.ods", "wr") as output_file:  #AWS
    with open("/home/rmendoza/Desktop/resultsXGB_1.ods", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        l = ['month','TrainDay','testDay','recall','filtered', 'time','NetSavings']
        wr.writerow(l)
        for diff in [1]:  #1,7  # as for now, only [1] means test on next day
            for month in range(6,7): #5,7    # as for now, only range(6,7) means june
                for day in range(4,5): #1,32  # as for now, only range(4,5) means 1st day
                    printit(  '------------------------------------------------')
                    printit(  '------------------------------------------------')
                    printit(  ['month = ', month,' and day = ',  day])
                    try:
                        start_time = time.time()
                        # Inputting training and testing set
                        #train_data, train_label = GetData(month, day)
                        X, y = GetData(month, day)  #local
                        printit('got X, y')
                        #X, y = GetData(month, day, 'true')  #aws
                        #dtrain = xgb.DMatrix(train_data, label=train_label)
                        dtrain = xgb.DMatrix(X, label=y)
                        test_data, test_label = GetData(month, day+diff) #local
                        #test_data, test_label = GetData(month, day+diff,'false')  #aws
                        dtest = xgb.DMatrix(test_data, label=test_label)

                        printit(  'selectKbest(f_classif, k =360) .....')
                        #selectK = SelectKBest(f_classif, k=360)#
                        selectK = SelectKBest(f_classif, k =1000)
                        # printit(  'selectK.fit(X, y) ...')
                        selectK.fit(X, y)
                        # printit(  'selectK.transform(X) ...')
                        X_sel = selectK.transform(X)
                        printit('selectK.get_support(): ')
                        printit(selectK.get_support())
                        # printit(  'X.columns[selectK.get_support()] ...')
                        # printit(  'X[np.ix_(range(len(X[:,0])), np.array(selectK.get_support()))] ...')
                        #features = X.columns[selectK.get_support()]
                        #features = X[np.ix_(range(len(X[:,0])), np.array([True, True, False, True]))]
                        features = X[np.ix_(range(len(X[:,0])), np.array(selectK.get_support()))]
                        #printit(  ['features', features] )
                        # printit(  'brief pause')
                        #time.sleep(20)  #sleep

                        ### Clean these chunk of code to be able to train/test

                        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)
                        clf = xgb.XGBClassifier(max_depth = 5,
                                                n_estimators=1525,
                                                learning_rate=0.02,
                                                nthread=4,
                                                subsample=0.95,
                                                colsample_bytree=0.85,
                                                seed=4242)
                        clf.fit(X_train, y_train, early_stopping_rounds=3, eval_metric="error", eval_set=[(X_test, y_test)])
                        ###  Clean also these
                        printit( ['Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]) ] )

                        sel_test = selectK.transform(test_data)
                        y_pred = clf.predict_proba(sel_test)
                        printit(['y_pred = ', y_pred])




                        # submission = pd.DataFrame({"ID":test_data.index, "TARGET":y_pred[:,1]})
                        # #printit('submission successful')
                        # submission.to_csv("submission.csv", index=False)
                        # #printit('submission successful')
                        mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
                        ts = pd.Series(clf.booster().get_fscore())
                        ts.index = ts.reset_index()['index'].map(mapFeat)
                        ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))
                        printit('ts.sort_values()... successful')

                        featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
                        plt.title('XGBoost Feature Importance')
                        fig_featp = featp.get_figure()
                        fig_featp.savefig('feature_importance_xgb22.png', bbox_inches='tight', pad_inches=1)


                        #### OLD CODE: ...

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
                        bst.dump_model('/home/rmendoza/Desktop/dump_raw2.txt')

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
                                    #get time:
                                    timer = time.time() - start_time
                                    print 'Time = ', timer
                                    l = [month,day,cutoff,recalll,filtered,timer]
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
                        print 'Failure x-P, something crashed.'
                    print '_____________________________________________________________________'
                    print '_____________________________________________________________________'