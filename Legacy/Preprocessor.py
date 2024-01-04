import json
import csv
from datetime import datetime
import os
import multiprocessing
import time
import Fields_and_Methods

start = time.time()


def get_io_addr_random_sample():
    list_io_addr = []
    root = "/home/ubuntu/random_samples"
    prefix = ["all", "", "new"]
    suffix = [i for i in range(6)]
    for i in prefix:
        for j in suffix:
            file_name = i+"data"+str(j)
            addr_in = os.path.join(root, file_name+".txt")
            addr_out = os.path.join(root, file_name+"_num.ods")
            list_io_addr.append((addr_in, addr_out))
    return list_io_addr


def get_io_addr():
    list_day = [i for i in range(2,3)]
    list_hour = [i for i in range(1)]
    list_month = [5]

    filename_in = "part-00000"
    filename_out = "output.ods"

    return Fields_and_Methods.make_io_addr(list_month,
                                           list_day,
                                           list_hour,
                                           filename_in,
                                           filename_out)


def crawl(io_addr):
    addr_in = io_addr[0]
    addr_out = io_addr[1]

    filtered = 0
    dumped = 0

    with open(addr_out, 'w') as file_out:
        wr = csv.writer(file_out, quoting = csv.QUOTE_MINIMAL)
        wr.writerow(Fields_and_Methods.__HEADER)
        with open(addr_in, "r") as file_in:
            print addr_in
            for line in file_in:
                try:
                    entry = json.loads(line)
                    result = []
                    result_list = []

                    auction = entry["auction"]
                    if_continue = filter(auction)   # Filter out auctions that do not contain any bid requests
                    if if_continue == 1:
                        filtered += 1
                        continue

                    event_process(entry, result)
                    auction_process(auction, result)
                    auction_site_process(auction, result)
                    auction_dev_process(auction, result)
                    auction_bidrequests_process(auction, result, result_list)

                    for item in result_list:
                        wr.writerow(item)
                except:
                    dumped += 1

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
                    if Fields_and_Methods.__FORMAT_MASK[imp["format"]] == 1:
                        bidreq_list[index]["impressions"].remove(imp)
            if len(bidreq_list[index]["impressions"]) == 0:
                bidreq_list.remove(bidreq)
                continue
            index += 1
        if len(auction["bidrequests"]) == 0:
            return 1


def event_process(entry, result):
    event = entry["em"]
    t = event["t"] / 1000
    result.append(datetime.fromtimestamp(t).hour)
    result.append(datetime.fromtimestamp(t).weekday())
    try:
        result.append(Fields_and_Methods.__DICT_COUNTRY[event["cc"]])
    except:
        result.append(Fields_and_Methods.__DICT_COUNTRY["EMPTY"])


def auction_process(auction, result):
    # Auction - Margin
    margin = int(auction["margin"])+1 # Take the floor
    if margin > 5:
        margin = 5
    result.append(margin)

    # Auction - Tmax
    if auction.has_key("tmax"):
        tmax = auction["tmax"]
        if tmax < 85:
            tmax = 1
        elif tmax == 85:
            tmax = 2
        else:
            tmax = 3
    else:
        tmax = 4
    result.append(tmax)

    # Auction - BKC
    if auction["user"].has_key("bkc"):
        result.append(1)
    else:
        result.append(0)


def auction_site_process(auction, result):
    # Auction - Site - Typeid
    site = auction["site"]
    result.append(site["typeid"])

    # Auction - Site - Cat
    site_cats = [0]*26  # Parse 26 different types of IAB categories
    if site.has_key("cat"):
        for cat in site["cat"]:
            cat_int = IAB_parser(cat)
            if site_cats[cat_int-1] == 0:
                site_cats[cat_int-1] = 1
    result.extend(site_cats)


def IAB_parser(str):
    s = str.split("IAB")
    str = s[1]
    if not str.isdigit():
        s = str.split("-")  # Ignore sub-category like IAB1-3
        str = s[0]
    return int(str)


def auction_dev_process(auction, result):
    try:
        type_index = Fields_and_Methods.__BROWSER_TYPE.index(auction["dev"]["bti"]) + 1
    except:
        type_index = 0
    result.append(type_index)


def auction_bidrequests_process(auction, result, result_list):
    # Record the bid ids and corresonding impression ids that are responded if any
    bid_responded = {}
    if auction.has_key("bids"):
        for bid in auction["bids"]:
            bid_responded.update({bid["requestid"]:bid["impid"]})

    for bidreq in auction["bidrequests"]:
        result_bid = result[:]
        bidder_id = bidreq["bidderid"]
        # Adjusting the index for DSP 36 since we ignore DSP 35 and 37
        if bidder_id == 36:
            bidder_id = 35
        result_bid.append(bidder_id)
        result_bid.append(bidreq["verticalid"])
        auction_bidrequest_impressions_process(bidreq, bid_responded, result_bid, result_list)


def auction_bidrequest_impressions_process(bidreq, bid_responded, result_bid, result_list):
    bidreq_id = bidreq["id"]

    # Determine if this impression is responded by any DSP
    impid_responded = -1
    if bid_responded.has_key(bidreq_id):
        impid_responded = bid_responded[bidreq_id]

    for imp in bidreq["impressions"]:
        # Auction - Bidrequests - Impressions - Bid Floor
        result_imp = result_bid[:]
        bid_floor = int(imp["bidfloor"])+1    # Take the floor
        if bid_floor > 5:
            if bid_floor < 11:
                bid_floor = 5
            else:
                bid_floor = 6
        result_imp.append(bid_floor)

        # Auction - Bidrequests - Impressions - Format
        result_imp.append(Fields_and_Methods.__FORMAT_INDEX[imp["format"]])

        # Auction - Bidrequests - Impressions - Product
        result_imp.append(imp["product"])

        # Auction - Bidrequests - Impressions - Banner
        if imp.has_key("banner"):
            width = imp["banner"]["w"]
            height = imp["banner"]["h"]
            if height <= 0 or width <= 0:
                banner = 1
            elif height < 400:
                if width < 500:
                    banner = 2
                else:
                    banner = 3
            else:
                if width < 500:
                    banner = 4
                else:
                    banner = 5
        else:
            banner = 0
        result_imp.append(banner)

        # Response
        if imp["id"] == impid_responded:
            result_imp.append(1)
        else:
            result_imp.append(0)
        result_list.append(result_imp)


if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    list_io_addr = get_io_addr_random_sample()

    dumped = 0
    filtered = 0

    for result in p.imap(crawl, list_io_addr):
        dumped += result[0]
        filtered += result[1]

    print "{} lines filtered".format(filtered)
    print "{} lines dumped".format(dumped)

print "Completed in {} seconds".format(round(time.time()-start, 2))
print
