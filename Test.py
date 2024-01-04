import os
import sys
import time
from sklearn.ensemble import RandomForestClassifier
import Get_Data_Rodrigo as gdr
from sklearn.metrics import confusion_matrix, recall_score
import csv  #to print to csvs

def netSav(r,f):
    netSaving = -5200+127000*f-850000*(1-r)
    return netSaving

def printit(text):
    print text
    sys.stdout.flush()


data = (6, 4)
month = data[0]
day = data[1]
root = "/mnt/rips2/2016"
addr_train = os.path.join(root, str(month).rjust(2, "0"), str(day).rjust(2, "0"))
addr_test = os.path.join(root, str(month).rjust(2, "0"), str(day+1).rjust(2, "0"))

X_train, y_train = gdr.get(addr_train, ratio=2.6)#), mode="res-25")
X_test, y_test = gdr.get(addr_test)

clf = RandomForestClassifier(n_estimators=40,
                             max_features="sqrt",
                             n_jobs=-1,
                             random_state=1514)

start = time.time()
printit( ["Start Training"])
sys.stdout.flush()
clf.fit(X_train, y_train)
printit( ["Training Completed in {} seconds".format(round(time.time()-start, 2))])
sys.stdout.flush()

prediction = clf.predict(X_test)
cf = confusion_matrix(y_test, prediction)
recalll = recall_score(y_test, prediction)
filtered = (cf[0, 0]) / float(len(prediction))
printit( ["Recall is = ", recalll])
printit( [ "Filtering is = ", filtered])
printit( ["Net Savings are = ", netSav(recalll, filtered)])
sys.stdout.flush()

# with open("/home/rmendoza/Desktop/testNB_sampPosNeg.ods", "wr") as output_file:
with open("/home/ubuntu/Rodrigo/test_EraseNoProb2.ods", "wr") as output_file:
    wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
    l = ['month = ', month,' and day = ',  day]
    wr.writerow(l)
    printit( l )