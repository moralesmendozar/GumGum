import Shared as sd


margins_ = [3.5, 2.45, 3.0, 2.0, 1.65, 0.85, 1.25, 0.45, 0.25, 0.15, 0.1, 4.5, 0.0, 4.0]
bkcids_ = sd.get_dict("bkc")


def process(entry, result):
    # Auction - margin
    margin = round(float(entry["margin"]), 2)
    sd.add_to_result(result, margin, margins_)

    # Auction - tmax
    tmax = entry["tmax"]
    if not tmax == "None":
        # Determine if tmax is multiple of 5 or 10
        if tmax % 5 == 0:
            result.append(1)
        else:
            result.append(0)

        if tmax % 10 == 0:
            result.append(1)
        else:
            result.append(0)

        for thres in [500, 700]:
            if tmax <= thres:
                result.append(1)
            else:
                result.append(0)

        index = 0
        tmax_list = [30, 45, 50, 70, 85, 1000]
        for item in tmax_list:
            if tmax == item:
                result.append(1)
                result.extend([0]*(len(tmax_list)-index-1))
                break
            result.append(0)
            index += 1
        if index >= len(tmax_list):
            result.append(1)
        else:
            result.append(0)

        result.append(0)
    else:
        result.extend([0]*11)
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
    tmax = ("tmax", 12)
    bkc = ("bkc", len(bkcids_)+2)

    return [margin, tmax, bkc]
