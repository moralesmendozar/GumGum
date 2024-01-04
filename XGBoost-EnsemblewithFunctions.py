import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import csv
from SendEmail import sendEmail
from sys import maxint
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
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m-1)
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y

def getBst(dtrain, evallist, train_label, modelName,num_round, eta,dumpname):
    p = np.count_nonzero(train_label)
    if p == 0:
        p = 1
        print 'there was a p = 0'
    n = len(train_label) - p
    if n == 0:
        n = 1
        print 'there was a n = 0'
    # Setting parameters
    # Train Model 1...
    try:
        bst = xgb.Booster() #init model
        print 'Loading the model... '
        #print '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2.model'
        bst.load_model(modelName) # load model
        return bst
    except Exception as e:
        print e
        #time.sleep(30)
        print "Some mistake in loading the model, so we should better train..." #pass  #to skip
        param = {'booster':'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'eval_metric':['auc'],
             'bst:max_depth':5,   # Max depth of tree
             'bst:eta':eta,   # Learning rate (usually 0.01-0.2)
             'bst:gamma':8.5,   # Larger value --> more conservative
             'bst:min_child_weight':1,
             'scale_pos_weight':n/float(p),   # Often num_neg/num_pos
             'subsample':.8,
             'silent':1,   # 0 outputs messages, 1 does not
             'save_period':0,   # Only saves last model
             'nthread':6,   # Number of cores used; otherwise, auto-detect
             'seed':25}
        #num_round = int(100*0.2/float(eta))   # Number of rounds of training, increasing this increases the range of output values
        bst = xgb.train(param,
                    dtrain,
                    num_round)#,
                    #evallist)   # If error doesn't decrease in n rounds, stop early#
        bst.save_model(modelName)
        print 'model saved'
        bst.dump_model(dumpname)
        print 'model dumped'
        return bst


def getAyAByB(y_hatTrain,train_data,train_label):
    ## Function recieves the y predicted for the train matrix and gives the x and y for the 0s and 1s
    ####y_hatTrain = bst.predict(dtrain)  #contains the 0s and 1s predicted by Model 1
    y_predTrain = []
    savings = [-maxint]
    # dacut = 0.09
    # y = y_hatTrain > dacut
    # recall = metrics.recall_score(train_label, y)
    # filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
    # if 127000*filter_rate -5200 -850000*(1-recall) > 0:
    #     savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
    #print 'beginn cutoffs'
    dacut = 0
    for cutoff in range(0, 20):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y = y_hatTrain > cut   # If y values are greater than the cutoff
        # print y
        recall = metrics.recall_score(train_label, y)
        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
        filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
        if 127000*filter_rate -5200 -850000*(1-recall) > savings[0]:
            savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
            dacut = cut
    #print 'serious binarize'
    # for kk in range(len(y_hatTrain)):
    #     if y_hatTrain[kk] > dacut:
    #         y_predTrain.append(1)
    #     else:
    #         y_predTrain.append(0)
    y_predTrain = y_hatTrain > dacut
    pred_pos = []
    pred_neg = []
    #print 'done binirize'
    for i in range(len(y_hatTrain)):
        entry = train_data[i].tolist()
        entry.append(train_label[i])
        if y_predTrain[i] == 1:
            # entry.append(1)
            pred_pos.append(entry)
        else:
            # entry.append(0)
            pred_neg.append(entry)
    pred_pos = np.array(pred_pos)
    pred_neg = np.array(pred_neg)
    #print pred_pos
    #print 'pred_pos[0:20,-10:] = '
    #print pred_pos[0:20,-10:]
    print 'Height of pred_neg is %s' % np.size(pred_neg, 0)
    #print 'Width of pred_neg is %s' % np.size(pred_neg, 1)
    print 'Height of pred_pos is %s' % np.size(pred_pos, 0)
    #print 'Width of pred_pos is %s' % np.size(pred_pos, 1)
    A = pred_neg[:,:-1]   #new A to train on
    yA = pred_neg[:,-1]   # new testA
    B = pred_pos[:,:-1]   # new B to train on
    yB = pred_pos[:,-1]   # new testB to train on ...
    return A, yA, B, yB

def getAyAByBtest(y_hatTrain,train_data,train_label):
    ## Function recieves the y predicted for the test matrix and gives the x and y for the 0s and 1s
    ####y_hatTrain = bst.predict(dtrain)  #contains the 0s and 1s predicted by Model 1
    posempty = 0
    negempty = 0
    y_predTrain = []
    savings = [-maxint]
    dacut = 0
    # dacut = 0.09
    # y = y_hatTrain > dacut
    # recall = metrics.recall_score(train_label, y)
    # filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
    # if 127000*filter_rate -5200 -850000*(1-recall) > 0:
    #     savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
    #print 'beginn cutoffs'
    for cutoff in range(0, 20):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y = y_hatTrain > cut   # If y values are greater than the cutoff
        # print y
        recall = metrics.recall_score(train_label, y)
        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
        filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
        if 127000*filter_rate -5200 -850000*(1-recall) > savings[0]:
            savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
            dacut = cut
    #print 'serious binarize'
    # for kk in range(len(y_hatTrain)):
    #     if y_hatTrain[kk] > dacut:
    #         y_predTrain.append(1)
    #     else:
    #         y_predTrain.append(0)
    y_predTrain = y_hatTrain > dacut
    pred_pos = []
    pred_neg = []
    #print 'done binirize'
    for i in range(len(y_hatTrain)):
        entry = train_data[i].tolist()
        entry.append(train_label[i])
        if y_predTrain[i] == 1:
            entry.append(1)
            pred_pos.append(entry)
        else:
            entry.append(0)
            pred_neg.append(entry)
    pred_pos = np.array(pred_pos)
    pred_neg = np.array(pred_neg)
    #print pred_pos
    #print 'pred_pos[0:20,-10:] = '
    #print pred_pos[0:20,-10:]
    print 'TESTHeight of pred_neg is %s' % np.size(pred_neg, 0)
    #print 'Width of pred_neg is %s' % np.size(pred_neg, 1)
    print 'TESTHeight of pred_pos is %s' % np.size(pred_pos, 0)
    #print 'Width of pred_pos is %s' % np.size(pred_pos, 1)
    if np.size(pred_pos,0)== 0:
        A = pred_neg[:,:-2]   #new A to train on
        yA = pred_neg[:,-2]   # new testA
        B = []   # new B to train on
        yB = []   # new testB to train on ...
        posempty = 1
    elif np.size(pred_neg,0)== 0:
        A = []   #new A to train on
        yA = []   # new testA
        B = pred_pos[:,:-2]   # new B to train on
        yB = pred_pos[:,-2]   # new testB to train on ...
        negempty = 1
    else:
        A = pred_neg[:,:-2]   #new A to train on
        yA = pred_neg[:,-2]   # new testA
        B = pred_pos[:,:-2]   # new B to train on
        yB = pred_pos[:,-2]   # new testB to train on ...
    return A, yA, B, yB,negempty, posempty

errorCounter = 0
if __name__ == "__main__":
    with open('/home/rmendoza/Desktop/XGBoost/XGB-Ensemble_10TIMING.csv', 'w') as file:
        #try:
        # Inputting training and testing set
        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
        wr.writerow(['eta1','eta2','eta3','Net_Savings', 'num_round', 'day_trained', 'day_predicted','hour_trainedAndTested', 'Time SpentOn Round'])
        # for eta1 in [.01, .05, .1, .15, .2]:
        #     for eta2 in [.01, .05, .1, .15, .2]:
        #         for eta3 in [.01, .05, .1, .15, .2]:
        eta = 0.15
        eta1 = eta
        eta =0.015
        eta2 = 0.15
        eta3 = 0.015
        for i in range(4,8):  #i is the day, goes to 24 to test on 25 and end. :P
            for j in range(17,24): # j is the hour
                print 'Beginning   eta1 =  ',eta1,', eta2 = ',eta2,'   eta3 = ', eta3,'   day = ', i, '  hour =  ', j
                start = time.time()
                print 'Loading Data'
                num_round = 500
                #eta = 0.1
                ph0 = str(j).rjust(2,'0')  #the hour on which to train and test
                p0 = str(i).rjust(2,'0')  #the day to train
                p1 = str(i+1).rjust(2,'0')  #the day to test
                #train_data, train_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p0+".npy")
                train_data, train_label = format_data('/media/54D42AE2D42AC658/DataHourly/output_new_06'+p0+ph0+'.npy')
                dtrain = xgb.DMatrix(train_data, label=train_label)
                #test_data, test_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p1+".npy")
                test_data, test_label = format_data('/media/54D42AE2D42AC658/DataHourly/output_new_06'+p1+ph0+'.npy')
                dtest = xgb.DMatrix(test_data, label=test_label)
                evallist = [(dtrain,'train'), (dtest,'eval')]
                print 'It took', time.time()-start, 's TIME to download the data.'
                timeThen = time.time()
                print 'Training bst (1)'
                modName = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta1'+str(eta1)+ '.model'
                dumpname = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta1'+str(eta1)+ '_v2.txt'
                bst = getBst(dtrain, evallist, train_label, modName,num_round, eta1,dumpname)
                print 'It took', time.time()-timeThen, 's TIME to train/load the bst (1).'
                timeThen = time.time()
                # train model 1A and 1B
                # first, divide train in trainA (yhat = 0) and trainB  (yhat = 1) and
                print 'predicting on dtrain...'
                y_hatTrain = bst.predict(dtrain)
                A, yA, B, yB = getAyAByB(y_hatTrain,train_data,train_label)
                print 'It took', time.time()-timeThen, 's TIME to getAyAByB(y_hatTrain,train_data,train_label).'
                timeThen = time.time()
                print 'Training BstA and B'
                dtrainA = xgb.DMatrix(A, label=yA)
                dtrainB = xgb.DMatrix(B, label=yB)
                evallistA = [(dtrainA,'train'), (dtest,'eval')]
                evallistB = [(dtrainB,'train'), (dtest,'eval')]
                #### Create new models 1A and 1B
                modNameA = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta2'+str(eta2)+ '_A.model'
                dumpnameA = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta2'+str(eta2)+ '_A.txt'
                modNameB = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta3'+str(eta3)+ '_B.model'
                dumpnameB = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 +'eta3'+str(eta3)+ '_B.txt'
                bstA = getBst(dtrainA, evallistA, yA, modNameA,num_round, eta2,dumpnameA)
                print 'It took', time.time()-timeThen, 's TIME to load-train the Bst (A).'
                timeThen = time.time()
                bstB = getBst(dtrainB, evallistB, yB, modNameB,num_round, eta3,dumpnameB)
                print 'It took', time.time()-timeThen, 's TIME to load-train the Bst (A).'
                timeThen = time.time()
                #print 'B[0:20,-10:] = ',B[0:20,-10:]
                # print 'yB[1:20] = '
                # print yB[0:20]
                # print 'y = y_pred > cut' =
                print 'Predict/test the model on next day'
                ######### Predict/test the model on next day
                y_true = test_label
                y_pred = bst.predict(dtest)
                ### Get the Xa and Xb for the predicted ones and zeroes
                testA, testyA, testB, testyB, abc,dfg = getAyAByBtest(y_pred,test_data,test_label)
                print 'It took', time.time()-timeThen, 's TIME to getAyAByBtest(y_pred,test_data,test_label).'
                timeThen = time.time()
                print 'Testing on Bst A and B'
                #### Now simply predict on testA and testB with models 1A and 1B and get the numbers...
                dtestA = xgb.DMatrix(testA, label=testyA)
                y_predA = bstA.predict(dtestA)
                #####Loop here
                dtestB = xgb.DMatrix(testB, label=testyB)
                #####Loop here
                y_predB = bstB.predict(dtestB)
                #print 'y_predB', y_predB
                ### Getting the splits of negatives and positives for each one
                testAneg, testyAneg, testApos, testyApos,negemptyA, posemptyA = getAyAByBtest(y_predA,testA,testyA)
                print 'It took', time.time()-timeThen, 's TIME to getAyAByBtest(y_predA,testA,testyA).'
                timeThen = time.time()
                print 'Training Bst B'
                testBneg, testyBneg, testBpos, testyBpos,negemptyB, posemptyB = getAyAByBtest(y_predB,testB,testyB)
                print 'It took', time.time()-timeThen, 's TIME to getAyAByBtest(y_predB,testB,testyB).'
                timeThen = time.time()
                print 'getting metrics ... '
                ### Get the recall and Net Savings
                labels = []
                preds = []
                if negemptyA == 0:
                    labels.extend(testyAneg.tolist())
                    preds.extend([0]*len(testyAneg))
                if posemptyA == 0:
                    labels.extend(testyApos.tolist())
                    preds.extend([1]*len(testyApos))
                if negemptyB == 0:
                    labels.extend(testyBneg.tolist())
                    preds.extend([0]*len(testyBneg))
                if posemptyB == 0:
                    labels.extend(testyBpos.tolist())
                    preds.extend([1]*len(testyBpos))
                labels = np.array(labels)
                preds = np.array(preds)
                recall = metrics.recall_score(labels, preds)
                filter_rate = sum(np.logical_not(preds))/float(len(preds))
                savings = 127000*filter_rate -5200 -850000*(1-recall)
                print '[day, hour, recall, filtering rate, savings ] '
                #['Net_Savings', 'num_round', 'day_trained', 'day_predicted','hour_trainedAndTested']
                timeSpentOnRound = time.time()-start
                print '[i, j, recall, filter_rate, savings, timeSpentOnRound ]'
                print [i, j, recall, filter_rate, savings, timeSpentOnRound ]
                results = [eta1, eta2, eta3,savings, num_round,p0, p1,ph0, timeSpentOnRound]
                print 'It took', time.time()-timeThen, 's TIME to get the metrics: recall filtering and so on.'
                timeThen = time.time()
                wr.writerow(results)
                print 'done for the hour', j
                print '--------------------------'
            print 'done for the DAY', i
            print '-------------------------------------'
            print '-------------------------------------'
        print '_______________________________________________________________________'
        print '_______________________________________________________________________'
        # except Exception as e:
        #     print e
        #     print 'ooops'
        #     #pass
        #     errorCounter += 1
        #     print 'There was an error, count ', errorCounter
        #     subjeto = 'Error on code... countOfError' + str(errorCounter)
        #     #sendEmail('moralesmendozar@gmail.com',subjeto,"XGBoost-trainHoursDaily.py encountered an error. :P")
            #time.sleep(20)  #sleep


sendEmail('moralesmendozar@gmail.com','Code Done2',"XGBoost-EnsemblewithFunctions.py ended running in the local RIPS computer. :P")