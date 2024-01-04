import Shared as sd


domains_ = sd.get_dict("domain")
browsers_ = [1, 2, 10, 13, 5, 11, 12, 7]


def process(entry, result):
    # Auction - Site - typeid
    if entry["typeid"] == -1:
        result.extend([0]*3)
        result.append(1)    # Use the last column to indicate missing site object
    else:
        sd.binarize(result, entry["typeid"]-1, 4)

    # Auction - Site - cat
    # Auction - Site - pcat
    for var in ["cat", "pcat"]:
        cat_process(result, entry, var)

    # Auction - Site - domain
    index = 0
    domain = entry["domain"]
    for item in domains_:
        if item in domain:
            break
        index += 1
    sd.binarize(result, index, len(domains_)+1)

    # Auction - Dev - browser type
    sd.add_to_result(result, entry["bti"], browsers_)


def cat_process(result, site, var):
    cats = [0]*27  # Parse 26 different types of IAB categories
    if len(site[var]) == 0:
        cats[len(cats)-1] = 1
    else:
        for cat in site[var]:
            if "IAB" in cat:
                cat_int = IAB_parser(cat)
                cats[cat_int-1] = 1
    result.extend(cats)


def IAB_parser(str):
    s = str.split("IAB")
    str = s[1]
    if not str.isdigit():
        s = str.split("-")  # Ignore sub-category like IAB1-3
        str = s[0]
    return int(str)


def get_hearder():
    site_typeid = ("site_typeid", 4)
    site_cat = ("site_cat", 27)
    site_pcat = ("site_pcat", 27)
    domain = ("domain", len(domains_)+1)
    browser = ("browser", len(browsers_)+1)

    return [site_typeid, site_cat, site_pcat, domain, browser]
