import os
import numpy as np
from scipy import vstack
import Sparse_Matrix_IO as smio


__FEATURES = ["hour", "day", "country", "margin", "tmax", "bkc", "site_typeid", "site_cat", "browser_type",
             "bidder_id", "vertical_id", "bid_floor", "format", "product", "banner", "response"]
# __FEATURES_TO_GET = []


def get_feature_indices():
    get_feature_length = {
        "hour": 24,
        "day": 7,
        "country": 193,
        "margin": 5,
        "tmax": 4,
        "bkc": 1,
        "site_typeid": 3,
        "site_cat": 26,
        "browser_type": 9,
        "bidder_id": 35,
        "vertical_id": 16,
        "bid_floor": 6,
        "format": 20,
        "product": 6,
        "banner": 5,
        "response": 1
    }
    feature_indices = {}
    begin = 0
    end = 0
    for item in __FEATURES:
        length = get_feature_length[item]
        end += length
        feature_indices.update({item:(begin, end)})
        begin += length
    return feature_indices


def get_cutoffs(features_to_get):
    feature_indices = get_feature_indices()
    cutoffs = []
    for item in features_to_get:
        indices = feature_indices[item]
        cutoffs.append(indices[0])
        cutoffs.append(indices[1])
    return sorted(cutoffs)


def select_features(matrix, features_to_get):
    cutoffs = get_cutoffs(features_to_get)
    matrix_new = []
    for line in matrix:
        new_line = []
        for i in range(0, len(cutoffs), 2):
            new_line.extend(line[cutoffs[i]:cutoffs[i+1]])
        new_line.append(line[len(line)-1])
        matrix_new.append(new_line)
    return np.array(matrix_new)


# With a given address of the day to take random sample from,
# obtain a numpy matrix that contain 100,000 impressions,
# with the specified ratio of positive responses and negative responses,
# and return the feature matrix X and the corresponding response vector y
# The ratio is given by pos/neg

def get(addr_day, hour = -1, cluster_number = -1, ratio=-1, features_to_get=None):
    '''
     With a given address of the day to take random sample from,
     obtain a numpy matrix that contain 100,000 impressions,
     with the specified ratio of positive responses and negative responses,
     and return the feature matrix X and the corresponding response vector y
     The ratio is given by pos/neg
    :param addr_day: e.g. "/mnt/rips2/2016/05/01"
    :param cluster_number: -1 by default. Can take values in range(4)
    :param ratio:  -1 by default. Ratio of pos to neg examples
    :param features_to_get:
    :return: X, y. Outputted features and labels
    '''
    if hour !=-1:
        p2 = str(hour).rjust(2,'0')
        with open(os.path.join(addr_day,p2, "output_new.npy"), "r") as file_in:
            matrix = smio.load_sparse_csr(file_in)
    else:
        if ratio != -1:
            n = 100000
            neg = int(n / (1+ratio))
            pos = n - neg
            if cluster_number != -1:
                with open(os.path.join(addr_day,"Clusters", "day_samp_new_neg_" + str(cluster_number) + ".npz"), "r") as file_neg:
                    matrix_neg = smio.load_sparse_csr(file_neg)
                    print "Negative Cluster %s Loaded" % cluster_number
                matrix_neg = matrix_neg[:, :]
                with open(os.path.join(addr_day,"Clusters", "day_samp_new_pos_" + str(cluster_number) + ".npz"), "r") as file_pos:
                    matrix_pos = smio.load_sparse_csr(file_pos)
                    print "Positive Cluster %s Loaded" % cluster_number
                matrix_pos = matrix_pos[:, :]
                # I cannot guarantee what size the clustered data samples might be, so have avoided the ratio part.
                matrix = vstack((matrix_neg, matrix_pos))
                np.random.shuffle(matrix)
            else:
                with open(os.path.join(addr_day,"PosNeg", "day_samp_new_neg.npy"), "r") as file_neg:
                    matrix_neg = smio.load_sparse_csr(file_neg)
                matrix_neg = matrix_neg[:neg, :]
                with open(os.path.join(addr_day,"PosNeg", "day_samp_new_pos.npy"), "r") as file_pos:
                    matrix_pos = smio.load_sparse_csr(file_pos)
                matrix_pos = matrix_pos[:pos, :]

                matrix = vstack((matrix_neg, matrix_pos))
                np.random.shuffle(matrix)
        else:
            if cluster_number != -1:
                with open(os.path.join(addr_day,"Clusters", "day_samp_new_cluster_" + str(cluster_number) + ".npz"), "r") as file:
                    matrix = smio.load_sparse_csr(file)
                    print "Cluster %s Loaded" % cluster_number
            else:
                with open(os.path.join(addr_day, "day_samp_new.npy"), "r") as file_in:
                    matrix = smio.load_sparse_csr(file_in)

    if (not features_to_get == None) and (len(features_to_get) > 0):
        matrix = select_features(matrix, features_to_get)

    width = np.size(matrix, 1)
    X = matrix[:, :width-1]
    y = matrix[:, width-1]

    return X, y
