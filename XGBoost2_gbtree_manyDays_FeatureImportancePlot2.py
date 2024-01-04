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
                    try:
                        start_time = time.time()
                        # Inputting training set
                        X, y = GetData(month, day, 'true')
                        dtrain = xgb.DMatrix(X, label=y)
                        # Inputting testing set
                        test_data, test_label = GetData(month, day+diff,'false')
                        dtest = xgb.DMatrix(test_data, label=test_label)
                        # Selecting Kbest:
                        selectK = SelectKBest(f_classif, k =300)
                        selectK.fit(X, y)
                        # Transform X into X_sel with the selectK
                        X_sel = selectK.transform(X)
                        # Note :-> selectK.get_support() #contains the false, neg of each feature.
                        # Get the matrix of only the "significant" features
                        features = X[np.ix_(range(len(X[:,0])), np.array(selectK.get_support()))]
                        # Now train the model using cross validation
                        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)


                        # # Setting parameters
                        # param = {'booster':'gbtree',#'gblinear',   # Tree, not linear regression
                        #          'objective':'binary:logistic',   # Output probabilities
                        #          'bst:max_depth': 4,   # Max depth of tree
                        #          'bst:eta':0.5,   # Learning rate (usually 0.01-0.2)
                        #          'silent':1,   # 0 outputs messages, 1 does not
                        #          'nthread':4}    # Number of cores used; otherwise, auto-detect
                        # evallist = [(dtest,'eval'), (dtrain,'train')]
                        # num_round = 100   # Number of rounds of training, increasing this increases the range of output values
                        # bst = xgb.train(param,
                        #                 dtrain,
                        #                 num_round,
                        #                 evallist,
                        #                 early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
                        # bst.dump_model('/home/rmendoza/Desktop/dump_raw2.txt')
                        # clf = bst
                        clf = xgb.XGBClassifier(max_depth = 5,   # Maximum tree depth for base learners.
                                                n_estimators=1525,   # Number of boosted trees to fit
                                                learning_rate=0.2,  #eta, (usually 0.01-0.2)
                                                nthread=-1, # Number of parallel threads used to run xgboost
                                                subsample=0.95,   # Subsample ratio of the training instance
                                                colsample_bytree=0.85, # Subsample ratio of columns when constructing each tree.
                                                seed=4242)
                        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(X_test, y_test)])
                        # # Test Data :->
                        sel_test = selectK.transform(test_data)
                        y_pred = clf.predict_proba(sel_test)
                        y_pred2 = clf.predict(sel_test)
                        printit('y_pred successful... now getting evaluation of results: (...) ') #just to know what we are up to...
                        results = evalModel(test_label, y_pred2, start_time, cut=0.01)
                        printit(['results :-> [ NetSav_scaled, AUC_score, recall, filtering_rate, cut, timer, netSavings  ] = '])
                        printit(results)
                        ####FEATURES NAMES and Plotting:
                        features = ["f" + str(i) for i in range(0,2531)]   # Feature names are f0..f2530
                        create_feature_map(features)

                        mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
                        ts = pd.Series(clf.booster().get_fscore())
                        ts.index = ts.reset_index()['index'].map(mapFeat)
                        ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

                        featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
                        plt.title('XGBoost Feature Importance')
                        fig_featp = featp.get_figure()
                        fig_featp.savefig('feature_importance_xgb212.png', bbox_inches='tight', pad_inches=1)

                        #### OLD CODE: ...


                        #xgb.plot_importance(bst, xlabel="test")
                        #xgb.plot_tree(bst, num_trees=2)
                        #wr = csv.writer(f,delimiter="\n")

                    except:
                        pass
                        print 'Failure x-P, something crashed.'
                    print '_____________________________________________________________________'
                    print '_____________________________________________________________________'