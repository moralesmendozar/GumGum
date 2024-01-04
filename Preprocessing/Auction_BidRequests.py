import Shared as sd


formats_ = [16, 31, 9, 12, 14, 3, 2, 7, 5, 21, 8, 20, 15, 6, 22, 27, 25, 26, 30, 13, 23]
banners_ = [(300, 250), (728, 90), (160, 600), (320, 50), (300, 600), (970, 90), (468, 60), (234, 60),
            (13, 13), (12, 12), (17, 17), (18, 18), (10, 10), (300, 120), (16, 16), (250, 100), (19, 19), (320, 480),
            (250, 70), (0, 0), (450, 100), (21, 21), (20, 20), (400, 400), (300, 100), (-1, -1)]    # (-1, -1) indicates missing banner


def if_multiple_bid_floor(result_imp, bid_floor, n):
    bid_floor_tmp = n*bid_floor
    if bid_floor_tmp == int(bid_floor_tmp):
        result_imp.append(1)
    else:
        result_imp.append(0)


def process(margin, entry, result, mode):
    # Auction - Bidrequests - bidder id
    bidder_id = entry["bidderid"]
    if bidder_id == 36: # Adjusting the index for DSP 36 since we ignore DSP 35 and 37
        bidder_id = 35
    sd.binarize(result, bidder_id-1, 35)

    # Auction - Bidrequests - vertical id
    sd.binarize(result, entry["verticalid"]-1, 16)

    # Auction - Bidrequests - Impressions - bid Floor
    bid_floor = round(float(entry["bidfloor"]), 2)

    if bid_floor-margin == 0:
        result.append(0)
    else:
        result.append(1)
    if mode == "bin":
        index = 0
        if bid_floor < 28:
            index = int(bid_floor*20)
        bid_floor_list = [0]*560
        bid_floor_list[index] = 1
        result.extend(bid_floor_list)
    else:
        result.append(bid_floor)

    # Determine if bid floor is a multiple of 0.05 or of 0.1
    if_multiple_bid_floor(result, bid_floor, 20)
    if_multiple_bid_floor(result, bid_floor, 10)

    index = 0
    thres_list = [1.5, 2, 2.5, 3, 28]
    for thres in thres_list:
        if bid_floor > thres:
            result.append(1)
            index += 1
        else:
            n = len(thres_list) - index
            result.extend([0]*n)
            break

    # Auction - Bidrequests - Impressions - format
    sd.binarize(result, formats_.index(entry["format"]), len(formats_))

    # Auction - Bidrequests - Impressions - product
    sd.binarize(result, entry["product"]-1, 6)

    # Auction - Bidrequests - Impressions - banner
    width = entry["w"]
    height = entry["h"]
    banner_cat = [0, 0, 0]
    if 0 < height <= 200:
        if 0 < width <= 500:
            banner_cat[0] = 1
        elif width > 500:
            banner_cat[1] = 1
    elif (height > 200) and (width <= 500):
        banner_cat[2] = 1
    sd.add_to_result(result, (width, height), banners_)
    result.extend(banner_cat)


def get_hearder():
    bidder_id = ("bidder_id", 35)
    vertical_id = ("vertical_id", 16)
    bid_floor = ("bid_floor", 9)
    format = ("format", len(formats_))
    product = ("product", 6)
    banner = ("banner", 3+len(banners_)+1)

    return [bidder_id, vertical_id, bid_floor, format, product, banner]