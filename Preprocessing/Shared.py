import os, json


dicts_root_ = "dicts"


def get_dict(var):
    with open(os.path.join(dicts_root_, "dict_"+var+".txt"), "r") as file_in:
        dict_var = [line.rstrip("\r\n") for line in file_in]
    return dict_var


def get_dict_json(var):
    with open(os.path.join(dicts_root_, var), "r") as file_in:
        return json.load(file_in)


def binarize(result, item, length):
    if item < 0:
        raise IndexError
    my_list = [0]*length
    my_list[item] = 1
    result.extend(my_list)


def add_to_result(result, var, dict_var):
    try:
        index = dict_var.index(var)
    except:
        index = len(dict_var)
    binarize(result, index, len(dict_var)+1)
