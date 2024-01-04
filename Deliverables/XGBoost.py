"""
Training and testing the XG Boost model
"""


import xgboost as xgb
import numpy as np
import Sparse_Matrix_IO as smio
#  A method to load in a sparse csr matrix
from sklearn import metrics
import os
import csv
from sys import maxint
# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(month, day, hour, ratio):
    """
    Takes in processed data as a sparse_csr matrix. We keep this format for memory efficiency
    :param month:
    :param day:
    :param hour:
    :param ratio: The ratio of negative to positive samples in our set (x100)
    :return: A matrix of features and labels
    """
    root = "/mnt/rips2/2016"
    p0 = str(month).rjust(2,'0')
    p1 = str(day).rjust(2,'0')
    addr_day = os.path.join(root,p0,p1)
    p2 = str(hour).rjust(2,'0')
    if ratio == -1:
        if hour == -1:
            data = os.path.join(addr_day,'day_samp_newer.npy')                 # Data for a day
        else:
            data = os.path.join(addr_day,p2, 'output_newer.npy')               # Data for an hour
        d = smio.load_sparse_csr(data)
        return d
    else:
        ratio = ratio/float(100)
        if hour == -1:
            data_pos = os.path.join(addr_day,'PosNeg/day_samp__newer_pos.npy') # Matrix of successful requests
            data_neg = os.path.join(addr_day,'PosNeg/day_samp_newer_neg.npy')  # Matrix of unsuccessful requests
        else:
            data_pos = os.path.join(addr_day,p2, 'output_pos_newer.npy')
            data_neg = os.path.join(addr_day,p2, 'output_neg_newer.npy')
        pos_matrix = smio.load_sparse_csr(data_pos)
        n = np.size(pos_matrix,axis=0)
        neg = int(ratio*n)
        neg_matrix = smio.load_sparse_csr(data_neg)[:neg,:]
        matrix = np.vstack((neg_matrix, pos_matrix))
        np.random.shuffle(matrix)
        return matrix


def format_data(month, day, hour=-1, ratio=-1):
    """

    :param month:
    :param day:
    :param hour:
    :param ratio:
    :return: Data set and labels as numpy arrays
    """
    d = get_data(month, day, hour, ratio)
    X = d[:, :-1]
    y = d[:, -1]
    return X, y

def netsav(recall, filtered):
    """
    Outputs the monetary values for the model
    :param recall: Recall
    :param filtered: Filter Rate
    :return: Savings from infrastructure reduction, Net RTB Income Lost, and Net Savings
    """
    InfrastruceReduction = 123000*filtered
    IncomeLost = 600000*(1-recall)
    NetSavings = InfrastruceReduction - IncomeLost - 5200
    return InfrastruceReduction, IncomeLost, NetSavings


def process_data(month, day, hour = -1, ratio = -1):
    """
    Put the data into DMatrix format
    :param month:
    :param day:
    :param hour:
    :param ratio:
    :return: Data set and labels in DMatrix format
    """
    data, label = format_data(month, day, hour, ratio)
    matrix = xgb.DMatrix(data, label=label)
    return matrix, label


def train_model(month, day):
    """
    Train the model
    :param month:
    :param day:
    :return: A trained XG-Boost Model
    """
    dtrain, train_label = process_data(month=month, day=day-1, ratio=100)
    # We found balancing the data to be best (ratio = 100). Below are our optimal hyperparameters
    p = sum(train_label)                        # number of ones
    n = len(train_label) - p                    # number of zeros
    # Setting parameters
    param = {'booster': 'gbtree',               # Tree, not linear regression
             'objective': 'binary:logistic',    # Output probabilities
             'eval_metric': ['auc'],
             'bst:max_depth': 5,                # Max depth of tree
             'bst:eta': .05,                    # Learning rate (usually 0.01-0.2)
             'bst:gamma': 8.5,                  # Larger value --> more conservative
             'bst:min_child_weight': 1,
             'scale_pos_weight': n/float(p),    # Often num_neg/num_pos
             'subsample': .8,
             'silent': 1,                       # 0 outputs messages, 1 does not
             'save_period': 0,                  # Only saves last model
             'nthread': 16,                     # Number of cores used; otherwise, auto-detect
             'seed': 25}
    evallist = [(dtrain,'train')]
    num_round = 500                             # Number of rounds of training
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    verbose_eval=50)
    return bst




if __name__ == "__main__":
    cut = 0.1                                   # Initialize the cutoff
    for month in range(5, 7):
        for day in range(1, 32):                # Loop through appropriate days
            try:
                bst = train_model(month, day)   # The trained model
                for hour in range(0, 24):
                    dtest, test_label = process_data(month, day, hour)
                                                # We test on each hour
                    pred_prob = bst.predict(dtest)
                                                # The predicted probabilities
                    pred = pred_prob > cut      # The binary predictions
                    optimal_results = [-maxint,0,0,0]   # o_r[3] is optimal cutoff, +/- 0.01
                    for cutoff in range(0, 41):
                        temp_cut = cutoff/float(100)
                        temp_pred = pred_prob > temp_cut
                        filter_rate = sum(np.logical_not(temp_pred))/float(len(temp_pred))
                        # We make use of the fact that True == 1 and False == 0 in python
                        recall = metrics.recall_score(test_label, temp_pred)
                        InfrastuctureReduction, IncomeLost, NetSavings =\
                            netsav(recall=recall, filtered=filter_rate)
                        if NetSavings > optimal_results[0]:
                            optimal_results[0] = NetSavings
                            optimal_results[1] = InfrastuctureReduction
                            optimal_results[2] = IncomeLost
                            optimal_results[1] = temp_cut
                                                        # This is the best that the model can do
                    output_file = "output_address/XGB.csv"
                    # Write the results to a csv
                    if not os.path.isfile(output_file):
                        with open(output_file, "a") as file:
                            wr = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
                            wr.writerow(["Day", "Hour", "Recall", "Filter Rate",
                                         "Savings", "Cutoff",
                                         "Optimal Savings","Savings from infrastructure reduction",
                                         "Net RTB Income Lost","Optimal Cutoff"])   # Labelling the results
                    with open(output_file, "a") as file:                            # Writing the results
                        results = [0,0,0,0,0,0,0,0]
                        results[0] = day
                        results[1] = hour
                        results[2] = metrics.recall_score(test_label, pred)
                        results[3] = sum(np.logical_not(pred))/float(len(pred))
                        # We make use of the fact that True == 1 and False == 0 in python
                        results[4] = netsav(results[2], results[3])
                        results[5] = cut
                        results[6] = optimal_results[0]
                        results[7] = optimal_results[1]
                        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
                        wr.writerow(results)

                    cut = optimal_results[1]       # Update the next hours cutoff
            except:
                pass
                # Train/Test only when valid