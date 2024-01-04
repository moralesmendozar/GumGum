import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import RFE
import time
import os
import csv
import operator
import pandas as pd
from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

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
    #x = d[:, :m-1]
    #y = d[:, m-1]
    nn = int(round(n*0.2))
    x = d[:nn, :m-1]
    y = d[:nn, m-1]
    print 'm = ', m
        #Get rid of column 2457 of d that is not bernoulli
    x = np.delete(x,np.s_[2457],1)
    # Just checking that indeed all non-Bernoulli variables are deleted
    for counter in range(m-2):
        if x[1,counter] != 0:
            if x[1,counter]!=1:
                'Oh no! Still a nonbernoulli variable is found! ... '
                print 'counter = ', counter
                print 'x[1, counter] = ', x[1, counter]

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

#clf = BernoulliNB(class_prior=[0.001, 0.999])
forest = ExtraTreesClassifier(n_estimators=50, random_state=1514)
if __name__ == "__main__":
    with open("/home/rmendoza/Desktop/temporaryOds.ods", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        l = ['month','TrainDay','testDay','recall','filtered', 'time','NetSavings']
        #wr.writerow(l)
        for diff in [1]:  #1,7  # as for now, only [1] means test on next day
            for month in range(6,7): #5,7    # as for now, only range(6,7) means june
                for day in range(4,5): #1,32  # as for now, only range(4,5) means 1st day
                    print '------------------------------------------------'
                    print '------------------------------------------------'
                    print 'month = ', month,' and day = ',  day
                    try:
                        start_time = time.time()
                        # Inputting training and testing set
                        train_data, train_label = GetData(month, day)
                        test_data, test_label = GetData(month, day+diff)
                        print 'Data Read'
                        #time.sleep(20)  #sleep
                        print 'Training Data...'
                        forest.fit(train_data, train_label)
                        print 'importances...'
                        importances = forest.feature_importances_
                        print 'std...'
                        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
                        print 'indices'
                        indices = np.argsort(importances)[::-1]
                        print("Feature ranking:")
                        for f in range(train_data.shape[1]):
                            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                        # Plot the feature importances of the forest

                        X = train_data
                        plt.figure()
                        plt.title("Feature importances")
                        plt.bar(range(X.shape[1]), importances[indices],
                               color="r", yerr=std[indices], align="center")
                        plt.xticks(range(X.shape[1]), indices)
                        plt.xlim([-1, X.shape[1]])

                        #print 'waiting (100 seconds...'
                        # time.sleep(50)
                        # selector = RFE(clf)#selector = RFE(clf, 5, step=1)
                        # selector = selector.fit(train_data, train_label) #selector = selector.fit(train_data, train_label, classes=[0, 1])
                        # print 'The selected features are...'
                        # print selector.support_
                        # print 'and the ranking... '
                        # print selector.ranking_
                        # clf.partial_fit(train_data, train_label, classes=[0, 1])
                        print 'Data Trained...'
                        y_true = test_label
                        n = len(y_true)
                        ### Here's a problem...
                        #print 'predictin jo ...'
                        clf = forest
                        y_pred = clf.predict(test_data)
                        #print 'getting conf matrix...'
                        cf = confusion_matrix(y_true,y_pred)
                        #print 'calculating recall...'
                        recalll = recall_score(y_true, y_pred)
                        #print 'calculating filtering'
                        filtered = (cf[0,0])/float(n)
                        print "Recall is: %s" % recalll
                        print 'Filtering is = ', filtered
                        print cf
                        #get time:
                        timer = time.time() - start_time
                        print 'Time = ', timer
                        NetSavings = netSav(recalll,filtered)
                        #print("--- %s seconds ---" % (time.time() - start_time))
                        # testday = day + diff
                        # l = [month,day,testday,recalll,filtered,timer,NetSavings]
                        # wr.writerow(l)
                        # ###get sum of columns
                        # vectorSums = np.sum(test_data, axis=0) #gets the count of the columns of each feature
                        # v1, v2, v3, v4, v5 =np.array_split(vectorSums,5) #split vector into 5 just to write it
                        # ##toPrint.append(v1)
                        # ##toPrint.append(v2)
                        # wr.writerow(v1)
                        # wr.writerow(v2)
                        # wr.writerow(v3)
                        # wr.writerow(v4)
                        # wr.writerow(v5)
                        # ###get Probas to read
                        # featureLogProb_ = clf.feature_log_prob_
                        # toPrint = []
                        # featProb = np.exp(featureLogProb_)  #contains the proba(xi | y), in the form size = 2xk,   y = 0, y = 1 and xi = 1
                        # toRead = np.divide(featProb[1,:],featProb[0,:])  #this is the proportion p(xi=1 | y =1) / p(xi = 1 | y = 0)
                        # v1, v2, v3, v4, v5 =np.array_split(toRead,5) #split vector into 5 just to write it
                        # # wr.writerow(v1)
                        # # wr.writerow(v2)
                        # # wr.writerow(v3)
                        # # wr.writerow(v4)
                        # # wr.writerow(v5)
                        # #toPrint.append(featProb)
                        # #toPrint.append(toRead)
                        # #print 'len(toPrint)', len(toPrint)
                        # #for jk in range(len(toPrint)):
                        # #    wr.writerow(toPrint[jk])
                        print 'plotting...'
                        plt.show()
                    except:
                        pass
                        print 'failure, no such day (or some other potential error)'
                        #time.sleep(20)  #sleep
                    print '_____________________________________________________________________'
                    print '_____________________________________________________________________'