import Shared as sd


margins_ = [3.5, 2.45, 3.0, 2.0, 1.65, 0.85, 1.25, 0.45, 0.25, 0.15, 0.1, 4.5, 0.0, 4.0]
bkcids_ = sd.get_dict("bkc")


def if_multiple_tmax(result, tmax, n):
    tmax_tmp = tmax / float(n)
    if tmax_tmp == int(tmax_tmp):
        result.append(1)
    else:
        result.append(0)


def process(entry, result):
    # Auction - margin
    margin = round(float(entry["margin"]), 2)
    sd.add_to_result(result, margin, margins_)

    # Auction - tmax
    tmax = entry["tmax"]
    if not tmax == "None":
        # Determine if tmax is multiple of 5 or 10
        if_multiple_tmax(result, tmax, 5)
        if_multiple_tmax(result, tmax, 10)

        for thres in [500, 700]:
            if tmax <= thres:
                result.append(1)
            else:
                result.append(0)

        if tmax <= 20:
            result.append(1)
            result.extend([0]*80)
        elif tmax <= 85:
            result.append(0)
            result_tmp = [0]*65
            result_tmp[tmax-21] = 1
            result.extend(result_tmp)
            result.extend([0]*15)
        elif tmax <= 135:
            result.extend([0]*66)
            result_tmp = [0]*10
            result_tmp[(tmax-86) / 5] = 1
            result.extend(result_tmp)
            result.extend([0]*5)
        else:
            result.extend([0]*76)
            result_tmp = [0]*5
            if tmax <= 200:
                result_tmp[0] = 1
            elif tmax <= 500:
                result_tmp[1] = 1
            elif tmax <= 999:
                result_tmp[2] = 1
            elif tmax == 1000:
                result_tmp[3] = 1
            else:
                result_tmp[4] = 1
            result.extend(result_tmp)
        result.append(0)
    else:
        result.extend([0]*85)
        result.append(1)    # Use the last column to indicate missing tmax

    # Auction - bkc
    bkc_result = [0]*(len(bkcids_)+2)
    bkc_str = entry["bkc"]
    if len(bkc_str) == 0:
        bkc_result[len(bkc_result)-1] = 1
    else:
        bkc_list = bkc_str.split(",")
        for item in bkc_list:
            try:
                index = bkcids_.index(item)
            except:
                index = len(bkc_result)-2
            bkc_result[index] = 1
    result.extend(bkc_result)

    return margin


def get_header():
    margin = ("margin", len(margins_)+1)
    tmax = ("tmax", 86)
    bkc = ("bkc", len(bkcids_)+2)

    return [margin, tmax, bkc]
