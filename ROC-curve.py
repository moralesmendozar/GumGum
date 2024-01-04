import numpy as np
from imblearn.over_sampling import SMOTE
from scipy import interp
import matplotlib.pyplot as plt
import Sparse_Matrix_IO as smio
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn import metrics

###############################################################################
# Data IO and generation
# import some data to play with
#file = "/home/kbhalla/Desktop/Data/day_samp_new.npy"
file = "/home/rmendoza/Documents/Data/day_samp_new_0604.npy"
with open(file, "r") as file_in:
        matrix = smio.load_sparse_csr(file_in)

X = matrix[:,:-1]
FA = FeatureAgglomeration(n_clusters=250)
print np.shape(X)
y = matrix[:,-1]
X = FA.fit_transform(X,y)
n_samples, n_features = X.shape
k = int(0.8*n_samples)
#random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 2*n_features)]
X_test, y_test = X[k:,:], y[k:]
X, y = X[:k, :], y[:k]
sm = SMOTE(ratio=0.95)
X,y = sm.fit_sample(X, y)
print np.shape(X)
start = time.time()




###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
classifier = BernoulliNB(class_prior=[0.01,0.99])
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

probas_ = classifier.fit(X, y).predict_proba(X_test)
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
for i in range(len(fpr)):
    if tpr[i] >= 0.97 and fpr[i] <= 0.3:
        print "The scores for threshold %s are recall %s, filtered %s" % (thresholds[i], tpr[i], 1-fpr[i])
        print metrics.confusion_matrix(y_test, probas_[:,1] > thresholds[i])
        break
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')


mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()