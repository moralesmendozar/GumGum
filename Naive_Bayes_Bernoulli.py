import os
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pickle
from scipy.sparse import csr_matrix
#test changes

def save_sparse_csr(filename, array):
    array_sparse = csr_matrix(array)
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']).toarray()


def get_io_addr():
    root = "/mnt/rips/2016"
    filename_in = "output_bin.npy"
    list_day = [i for i in range(1, 2)]
    list_hour = [i for i in range(10)]
    list_month = [5]

    list_io_addr = []
    for month in list_month:
        for day in list_day:
            if month == 6:
                day += 18
            for hour in list_hour:
                io_addr = os.path.join(root,
                                       str(month).rjust(2, "0"),
                                       str(day).rjust(2, "0"),
                                       str(hour).rjust(2, "0"))
                addr_in = os.path.join(io_addr, filename_in)
                list_io_addr.append(addr_in)
    return list_io_addr
alfa = 0.01
clf = BernoulliNB(class_prior=[alfa, 1-alfa])
list_io_addr = get_io_addr()

for i in range(len(list_io_addr)):
    path_in = list_io_addr[i]

    print "Processing {}".format(path_in)

    with open(path_in, "r") as file_in:
        X = load_sparse_csr(file_in)

    m = len(X[0])
    n = len(X)

    X_train = X[:, 0:m-1]
    y_train = X[:, m-1]

    print "Done"
    print

    sm = SMOTE(ratio=0.9)
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

    print
    print "Fitting Model"
    clf.partial_fit(X_train_sm, y_train_sm, classes=[0, 1])
    print "Done"
    print

with open("/home/ubuntu/Weiyi/model_05_01", "w") as file_out:
    pickle.dump(clf, file_out)

# with open("/home/wlu/Desktop/model_bernoulli", "r") as file_in:
#     clf = pickle.load(file_in)
#
# print "Generating the training set"
# with open("/home/wlu/Desktop/test_sparse.npy", "r") as file_in:
#     X = load_sparse_csr(file_in).toarray()
#
# m = len(X[0])
# n = len(X)
#
# X = np.array(X)
#
# X_test = X[:, 0:m-1]
# y_test = X[:, m-1]
#
# print "Done"
# print
#
# probas = clf.predict_proba(X_test)
#
# for alpha in [0.99+0.001*i for i in range(10)]:
#     print "________________________________________________________"
#     print "alpha = ", alpha
#     print
#     prediction = []
#     for k in range(len(y_test)):
#         if probas[k,0] > alpha:
#             prediction.append(0)
#         else:
#             prediction.append(1)
#
#     total = len(prediction)
#     tn = 0
#     tp = 0
#     fp = 0
#     fn = 0
#     for i in range(total):
#         if prediction[i]-y_test[i] == 0:
#             if  prediction[i] == 1:
#                 tp+=1
#             else:
#                 tn +=1
#         else:
#             if prediction[i] == 1:
#                 fp +=1
#             else:
#                 fn +=1
#     fitering = (tn + fn) / float(total)
#     print 'filtering = ', fitering
#     print 'recall = ', tp/ float(tp + fn)
#     print metrics.precision_recall_fscore_support(y_test, prediction, average = 'binary', beta = 12 )
#     print metrics.confusion_matrix(y_test, prediction)
#     print "________________________________________________________"
