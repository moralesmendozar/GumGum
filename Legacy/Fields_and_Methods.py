import json
import os
import errno

with open("io_addr_root.txt", "r") as file_addr_root:
    __ADDR_ROOT = file_addr_root.readline().replace("\"", "").rstrip("\n")

__HEADER = ["hour", "day", "country", "margin", "tmax", "bkc", "site_typeid", "browser_type",
             "bidder_id", "vertical_id", "bid_floor", "format", "product", "banner", "response"]
__INDEX_SITE_TYPE = __HEADER.index("site_typeid")
__HEADER[__INDEX_SITE_TYPE+1:1] = ["site_cat {}".format(i+1) for i in range(26)]

__FORMAT_COUNT = 31
__FORMAT_TO_IGNORE = [1,2,3,4,6,7,20,21,22,25,26]
__FORMAT_MASK = [0]*(__FORMAT_COUNT+1)
for i in __FORMAT_TO_IGNORE:
    __FORMAT_MASK[i] = 1
__FORMAT_INDEX = {}
count = 1
for i in range(1,__FORMAT_COUNT+1):
    if __FORMAT_MASK[i] == 0:
        __FORMAT_INDEX.update({i:count})
        count += 1

__BROWSER_TYPE = [1,2,5,7,10,11,12,13]

ADDR_COUNTRY_DICT = "dict_country.json"
with open(ADDR_COUNTRY_DICT, "r") as file_in:
    __DICT_COUNTRY = json.load(file_in)

# For Converter:

__FEATURES = ["hour", "day", "country", "margin", "tmax", "site_typeid", "browser_type",
             "bidder_id", "vertical_id", "bid_floor", "format", "product", "banner"]
__FEATURES_LEN = len(__FEATURES)
__INDICES = [__HEADER.index(__FEATURES[i]) for i in range(__FEATURES_LEN)]

__INDEX_BKC = __HEADER.index("bkc")
__INDEX_RESPONSE = __HEADER.index("response")

__INDEX_SITE_CAT = __INDEX_SITE_TYPE+1


def get_classes():
    get_classes_options = {
        "hour": create_class(0, 24),
        "day": create_class(0, 7),
        "country": create_class(0, 193),
        "margin": create_class(1, 5),
        "tmax":create_class(1, 4),
        "site_typeid": create_class(1, 3),
        "browser_type": create_class(1, 8),
        "bidder_id": create_class(1, 35),
        "vertical_id": create_class(1, 16),
        "bid_floor": create_class(1, 6),
        "format": create_class(1, 20),
        "product": create_class(1, 6),
        "banner": create_class(1, 5)
    }
    classes = []
    for item in __FEATURES:
        classes.append(get_classes_options[item])
    return classes


def create_class(indexing, length):
    return [i+indexing for i in range(length)]


__CLASSES = get_classes()


# Some methods for both Preprocessor and Converter

def make_output_addr(list_file_dir):
    out_path = os.path.join(__ADDR_ROOT, list_file_dir)
    try:
        os.makedirs(out_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return out_path


def make_io_addr(list_month, list_day, list_hour, filename_in, filename_out):
    list_io_addr = []
    for month in list_month:
        for day in list_day:
            if month == 6:
                day += 18
            for hour in list_hour:
                io_addr = os.path.join(__ADDR_ROOT,
                                       str(month).rjust(2, "0"),
                                       str(day).rjust(2, "0"),
                                       str(hour).rjust(2, "0"))
                addr_in = os.path.join(io_addr, filename_in)
                addr_out = os.path.join(io_addr, filename_out)
                list_io_addr.append((addr_in, addr_out))
    return list_io_addr
