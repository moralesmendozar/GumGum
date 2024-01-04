import multiprocessing
import time
import json
import os


formats_ = [16, 31, 9, 12, 14, 3, 2, 7, 5, 21, 8, 20, 15, 6, 22, 27, 25, 26, 30, 13, 23]


def get_io_addr():
    may = [(5, i, j) for i in range(1, 8) for j in range(24)]
    june = [(6, i, j) for i in range(4, 26) for j in range(24)]
    # june = []

    filename_in = "part-00000"
    root_in = "/mnt/rips/2016"
    root_out = "/mnt/rips2/2016"

    list_dates = may + june
    list_io_addr = []
    for date in list_dates:
        month = date[0]
        day = date[1]
        hour = date[2]
        io_addr = os.path.join(str(month).rjust(2, "0"),
                               str(day).rjust(2, "0"),
                               str(hour).rjust(2, "0"))
        addr_in = os.path.join(root_in, io_addr, filename_in)
        path_out = os.path.join(root_out, io_addr)
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        addr_out = os.path.join(path_out)
        list_io_addr.append((addr_in, addr_out))
    return list_io_addr


def crawl(io_addr):
    addr_in = io_addr[0]
    addr_out = io_addr[1]

    filtered = 0
    dumped = 0

    pos = []
    neg = []
    if os.path.isfile(addr_in):
        with open(addr_in, "r") as file_in:
            print addr_in
            for line in file_in:
                try:
                    entry = json.loads(line)
                    result = {}
                    result_pos = []
                    result_neg = []

                    auction = entry["auction"]
                    if_continue = filter(auction)   # Filter out auctions that do not contain any bid requests
                    if if_continue == 1:
                        filtered += 1
                        continue

                    event_process(entry, result)
                    auction_process(auction, result)
                    auction_site_process(auction, result)
                    auction_dev_process(auction, result)
                    auction_bidrequests_process(auction, result, result_pos, result_neg)

                    for item in result_pos:
                        pos.append(item)
                    for item in result_neg:
                        neg.append(item)

                except:
                    dumped += 1

        with open(os.path.join(addr_out, "output_pos"), 'w') as file_out:
            for line in pos:
                entry = json.dumps(line)
                file_out.write(entry)
                file_out.write("\n")

        with open(os.path.join(addr_out, "output_neg"), 'w') as file_out:
            for line in neg:
                entry = json.dumps(line)
                file_out.write(entry)
                file_out.write("\n")

    else:
        print "\nFile Missing: {}\n".format(addr_in)

    return [dumped, filtered]


def filter(auction):
    if not auction.has_key("bidrequests"):
        return 1
    else:
        bidreq_list = auction["bidrequests"]
        bidreq_list_copy =bidreq_list[:]
        index = 0
        for bidreq in bidreq_list_copy:
            if (bidreq["bidderid"] == 35 or bidreq["bidderid"] == 37) or (not bidreq.has_key("impressions")):
                bidreq_list.remove(bidreq)
                continue
            else:
                imp_list = bidreq["impressions"][:]
                for imp in imp_list:
                    # Filter out ad formats that should be ignored
                    if (not imp["format"] in formats_) or (imp["bidfloor"] < 0):
                        bidreq_list[index]["impressions"].remove(imp)
            if len(bidreq_list[index]["impressions"]) == 0:
                bidreq_list.remove(bidreq)
                continue
            index += 1
        if len(auction["bidrequests"]) == 0:
            return 1


def update_result(result, key, val):
    result.update({key: val})


def event_process(entry, result):
    # Event - t
    event = entry["em"]
    t = event["t"]
    update_result(result, "t", t)

    # Event - country
    try:
        country = event["cc"]
    except:
        country = "None"
    update_result(result, "cc", country)

    # Event - region
    try:
        region = event["rg"]
    except:
        region = "None"
    update_result(result, "rg", region)


def auction_process(auction, result):
    # Auction - margin
    margin = auction["margin"]
    update_result(result, "margin", margin)

    # Auction - tmax
    try:
        tmax = auction["tmax"]
    except:
        tmax = "None"
    update_result(result, "tmax", tmax)

    # Auction - bkc
    try:
        bkc = auction["user"]["bkc"]
    except:
        bkc = ""
    update_result(result, "bkc", bkc)


def auction_site_process(auction, result):
    try:
        site = auction["site"]
        # Auction - Site - type id
        typeid = site["typeid"]

        # Auction - Site - site category
        try:
            cat = site["cat"]
        except:
            cat = []

        # Auction - Site - page category
        try:
            pcat = site["pcat"]
        except:
            pcat = []

        # Auction - Site - domain
        try:
            domain = site["domain"]
        except:
            domain = "None"
        if len(domain) == 0:
            domain = "None"

    except:
        typeid = -1
        cat = []
        pcat = []
        domain = "None"

    update_result(result, "typeid", typeid)
    update_result(result, "cat", cat)
    update_result(result, "pcat", pcat)
    update_result(result, "domain", domain)


def auction_dev_process(auction, result):
    # Auction - Dev - browser type
    try:
        bti = auction["dev"]["bti"]
    except:
        bti = -1
    update_result(result, "bti", bti)


def auction_bidrequests_process(auction, result, result_pos, result_neg):
    # Record the bid ids and corresonding impression ids that are responded if any
    bid_responded = {}
    if auction.has_key("bids"):
        for bid in auction["bids"]:
            bid_responded.update({bid["requestid"]:bid["impid"]})

    for bidreq in auction["bidrequests"]:
        result_bid = result.copy()

        # Auction - Bidrequests - bidder id
        bidderid = bidreq["bidderid"]
        update_result(result_bid, "bidderid", bidderid)

        # Auction - Bidrequests - vertical id
        verticalid = bidreq["verticalid"]
        update_result(result_bid, "verticalid", verticalid)

        auction_bidrequest_impressions_process(bidreq, bid_responded, result_bid, result_pos, result_neg)


def auction_bidrequest_impressions_process(bidreq, bid_responded, result_bid, result_pos, result_neg):
    bidreq_id = bidreq["id"]
    # Determine if this impression is responded by any DSP
    impid_responded = -1
    if bid_responded.has_key(bidreq_id):
        impid_responded = bid_responded[bidreq_id]

    for imp in bidreq["impressions"]:
        # Auction - Bidrequests - Impressions - bid Floor
        result_imp = result_bid.copy()
        bidfloor = imp["bidfloor"]
        update_result(result_imp, "bidfloor", bidfloor)

        # Auction - Bidrequests - Impressions - format
        format = imp["format"]
        update_result(result_imp, "format", format)

        # Auction - Bidrequests - Impressions - product
        product = imp["product"]
        update_result(result_imp, "product", product)

        # Auction - Bidrequests - Impressions - banner
        try:
            w = imp["banner"]["w"]
            h = imp["banner"]["h"]
        except:
            w = -1
            h = -1
        update_result(result_imp, "w", w)
        update_result(result_imp, "h", h)

        # Response
        if imp["id"] == impid_responded:
            update_result(result_imp, "response", 1)
            result_pos.append(result_imp)
        else:
            update_result(result_imp, "response", 0)
            result_neg.append(result_imp)


if __name__ == '__main__':
    start = time.time()

    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    list_io_addr = get_io_addr()

    dumped = 0
    filtered = 0

    for result in p.imap(crawl, list_io_addr):
        dumped += result[0]
        filtered += result[1]

    print "{} lines filtered".format(filtered)
    print "{} lines dumped".format(dumped)

    print "Completed in {} seconds\n".format(round(time.time()-start, 2))