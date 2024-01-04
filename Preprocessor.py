import os
import sys
import json
import time
import numpy as np
import multiprocessing
from scipy.sparse import csr_matrix, vstack
from Preprocessing import Driver


start = time.time()


def get_io_addr_hour():
    # may = [(5, i, j) for i in range(1, 8) for j in range(17, 18)]
    may = [(5, i, 17) for i in range(1, 8)]
    # june = [(6, i, j) for i in range(20, 21) for j in range(1)]
    june = []
    root = "/mnt/rips2/2016"

    list_io_addr = []
    for date in may+june:
        month = date[0]
        day = date[1]
        hour = date[2]
        addr_io = os.path.join(root,
                               str(month).rjust(2, "0"),
                               str(day).rjust(2, "0"),
                               str(hour).rjust(2, "0"))
        addr_in = os.path.join(addr_io, "output_neg_raw")
        addr_out = os.path.join(addr_io, "output_neg_newer.npy")
        list_io_addr.append((addr_in, addr_out))

    return list_io_addr


def get_io_addr_day_samp():
    may = [(5, i) for i in range(1, 32)]
    # may = []
    june = [(6, i) for i in range(1, 4)]
    # june = []
    mode_in = "neg"

    if mode_in == "normal":
        filename_in = "day_samp_raw"
        filename_out = "day_samp_newer.npy"
    else:
        filename_in = "PosNeg/day_samp_raw_{}".format(mode_in)
        filename_out = "PosNeg/day_samp_newer_{}.npy".format(mode_in)

    root = "/mnt/rips2/2016"
    list_io_addr = []
    for item in may+june:
        month = item[0]
        day = item[1]
        io_addr = os.path.join(root,
                               str(month).rjust(2, "0"),
                               str(day).rjust(2, "0"))
        addr_in = os.path.join(io_addr, filename_in)
        addr_out = os.path.join(io_addr, filename_out)
        list_io_addr.append((addr_in, addr_out))

    return list_io_addr


def get_io_addr_random_sample():
    list_io_addr = []
    root = "/home/ubuntu/random_samples"
    prefix = ["new", "all", ""]
    suffix = [i for i in range(6)]
    for i in prefix:
        for j in suffix:
            file_name = i+"data"+str(j)
            addr_in = os.path.join(root, file_name+"_raw")
            addr_out = os.path.join(root, file_name+"_new.npy")
            list_io_addr.append((addr_in, addr_out))
    return list_io_addr


def crawl(io_addr):
    dumped = 0
    data_sparse_list = []
    addr_in = io_addr[0]
    addr_out = io_addr[1]
    if os.path.isfile(addr_in):
        with open(addr_in, "r") as file_in:
            print "Processing {}".format(addr_in)
            sys.stdout.flush()
            for line in file_in:
                try:
                    entry = json.loads(line)
                    result = []
                    Driver.process(entry, result)
                    data_sparse_list.append(csr_matrix(result))

                except:
                    dumped += 1

        data_matrix = vstack(data_sparse_list)
        with open(addr_out, 'w') as file_out:
            np.savez(file_out,
                     data=data_matrix.data,
                     indices=data_matrix.indices,
                     indptr=data_matrix.indptr,
                     shape=data_matrix.shape)

    else:
        print "\nFile Missing: {}\n".format(addr_in)
        sys.stdout.flush()

    return dumped


if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    list_io_addr = get_io_addr_day_samp()

    dumped = 0
    for result in p.imap(crawl, list_io_addr):
        dumped += result

    print "{} lines dumped".format(dumped)
    sys.stdout.flush()

print "Completed in {} seconds\n".format(round(time.time()-start, 2))
sys.stdout.flush()
