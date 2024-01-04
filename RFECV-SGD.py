print(__doc__)
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn import metrics
try:
    import cPickle as pickle
except:
    import pickle


def M_score(y_true,y_pred):
    r = metrics.recall_score(y_true,y_pred)
    cf = metrics.confusion_matrix(y_true,y_pred)
    filtered = (cf[0,0]+cf[1,0])/float(cf[0,1] + cf[1,1])
    return 127000*filtered - 5200 - 850000*(1-r)


def GetData(data_list): ## Input Weiyi-formatted Data
    print "Reading Data..."
    count, temp = 1, np.load(data_list[0])
    Data = csr_matrix((  temp['data'], temp['indices'], temp['indptr']),
                         shape = temp['shape'], dtype=float).toarray()
    print "Finished reading data file %s of %s..." %(count, len(data_list))
    for i in range(1,len(data_list)):
        count +=1
        temp = np.load(data_list[i])
        temp2 = csr_matrix((temp['data'], temp['indices'], temp['indptr']),
                         shape = temp['shape'], dtype=float)
        temp3 = temp2.toarray()
        Data = np.append(Data,temp3,0)
        print "Finished reading data file %s of %s..." %(count, len(data_list))
    print "All data files read"
    return Data


Data = GetData(["/home/kbhalla/Desktop/Data/day_samp_new.npy"])

X, y = Data[:200000,:-1], Data[:200000,-1]
n = len(y)
K = np.count_nonzero(y)
y = 2*y - np.ones(n)
print "X, y created"
# Create the RFE object and compute a cross-validated score.
clf = SGDClassifier(learning_rate='optimal',
                             class_weight={-1:1, 1:5*n/K}, n_jobs=-1,alpha=0.00001, warm_start=True,
                             n_iter = 10, penalty='l2', average=True, eta0=0.0625)
# The "accuracy" scoring is proportional to the number of correct
# classifications
print "RFE Cross Val Time"
Mscore = make_scorer(M_score, greater_is_better=True)
rfecv = RFECV(estimator=clf, step=25, cv=StratifiedKFold(y, 2),
              scoring=Mscore)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of subset of features (25*index = number of features?)")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_)+ 1), rfecv.grid_scores_)
plt.show()
feature_vec = rfecv.get_support()
pickle.dump(feature_vec, open("/home/kbhalla/Desktop/feature_vec.p", "w"))