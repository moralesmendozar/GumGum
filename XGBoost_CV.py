import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import time
try:
    import cPickle as pickle
except:
    import pickle


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html
# Code from http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


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
    x = d[:200000, :m-1]   # To reduce load on computer
    y = d[:200000, m-1]
    return x, y


def modelfit(alg, train_data, train_label, cv_folds=5, early_stopping_rounds=1):

    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(train_data, label=train_label)
    cvresult = xgb.cv(xgb_param,
                      xgtrain,
                      num_boost_round=alg.get_params()['n_estimators'],
                      nfold=cv_folds,
                      metrics=['auc'],
                      early_stopping_rounds=early_stopping_rounds,
                      show_progress=True)
    alg.set_params(n_estimators=cvresult.shape[0])   # Goal of CV is to tune the number of rounds, which is set here

    # Note: can change to a different day to see what happens
    start = time.time()
    alg.fit(train_data,
            train_label,
            eval_metric='auc')
    print "Time to fit: %s" % (time.time()-start)

    pickle.dump(alg, open("/home/jche/Desktop/xgboost.p", "w+"))   # Save model

    start = time.time()
    dtrain_predprob = alg.predict_proba(train_data)[:,1]
    print "Time to predict: %s" % (time.time() - start)

    for cutoff in range(0, 41):
        cut = cutoff/float(100)   # Cutoff in decimal form
        dtrain_predictions = dtrain_predprob > cut   # If y values are greater than the cutoff
        # Print model report:
        print "\nModel Report for cutoff %s" % cut
        print "Accuracy : %.4g" % metrics.accuracy_score(train_label, dtrain_predictions)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(train_label, dtrain_predprob)
        print "Recall is: %s" % metrics.recall_score(train_label, dtrain_predictions)
        print metrics.confusion_matrix(train_label, dtrain_predictions)


if __name__ == "__main__":
    train_data, train_label = format_data("/home/jche/Data/alldata5_new.npy")
    xgb1 = XGBClassifier(learning_rate =0.1,
                         n_estimators=1000,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         nthread=6,   # used to be 4
                         scale_pos_weight=5,   # used to be 1
                         seed=20)
    modelfit(xgb1, train_data, train_label)