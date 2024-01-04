"""
Training an SVM model
"""
import numpy as np
import os
from Get_Data import get
# get is some method which returns samples and their labels
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import linear_model, preprocessing
try:
    import cPickle as pickle
except:
    import pickle

def data_format(train_day):
    """
    Changes the labels from {0,1} to {-1,1}, performs feature selection, and scales the testing samples
    :param train_day: The address of the day we wish to train on
    :return: The train set and its labels, plus the number of samples, and the number of positive samples
    """

    X, y = get(addr_day=train_day)
    n = np.size(y, 0)
    k = np.count_nonzero(y)
    y_test = 2*y[:,-1]-np.ones(n)
    # We change any zeros in the label to negative ones
    X_test = BestK.transform(X)
    # We select the K Best features only
    scaler.transform(X_test)
    # We scale each feature in a standardized way
    return X_test, y_test, n, k

def training(train_day):
    """
    Training the model on data
    :param train_day: The day on which we train
    :return: A trained model for the filter to use
    """
    X_train, y_train, n, k = data_format(train_day)
    clf = linear_model.SGDClassifier(learning_rate='optimal',# SGD Classifier is much faster in practice
                             class_weight={-1:1, 1:5*n/k}, # This controls the tradeoff between filter rate and recall
                                                             # The constant should be 5 for ~350 features, 2.5 for ~750
                                                             # and roughly 1.5 for ~2000 features.
                             n_jobs=-1,                      # Use all cores
                             alpha=0.000001,
                             warm_start=True,                # Online Learning
                             n_iter = 10,                    # Number of iterations
                             penalty='l2',
                             average=True,
                             eta0=0.0625)                    # Parameter affecting the learning rate

    clf.fit(X_train,y_train)
    # Fit the model
    pickle.dump(clf,open( os.path.join(train_day,"Models/SGD.p"), "wb"))
    # Save the model



for month in range(5,7):
    root = '/mnt/rips2/2016'
    p0 = "0" + str(month)
    Data , label = get('chosen day')
    # Here we standardise feature selection and scaling
    BestK = SelectKBest(chi2, k = 350)
    BestK.fit(Data, label)
    # Feature Selection - Select K Best
    Data = BestK.transform(Data)
    scaler = preprocessing.StandardScaler().fit(Data)
    # The method via which we scale the data
    for day in range(1,32):
        try:
            p1 = str(day).rjust(2,'0')
            p2 = str(day-1).rjust(2,'0')
            train_day = os.path.join(root,p0,p1)
            training(train_day=train_day)
            # Train the model
        except:
            pass
            # Ensures we train only on valid days

