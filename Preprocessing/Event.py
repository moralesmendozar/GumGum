import Shared as sd
import pytz
from datetime import datetime


countries_ = ["None", "US", "GB", "CA", "DE", "FR", "NL", "IT"]
regions_ = sd.get_dict("region")
region_timezone_ = sd.get_dict_json("region_timezone.json")


def process(entry, result):
    # Event - time
    t = entry["t"] / 1000

    pst_t = datetime.fromtimestamp(t)
    min = pst_t.minute
    sd.binarize(result, min, 60)

    hour = pst_t.hour
    sd.binarize(result, hour, 24)

    day = pst_t.weekday()
    sd.binarize(result, day, 7)

    if day == 5 or day == 6:
        result.append(1)
    else:
        result.append(0)

    if day == 4 or day == 5:
        result.append(1)
    else:
        result.append(0)

    try:
        utc_t = pytz.utc.localize(datetime.utcfromtimestamp(t))
        country = entry["cc"]
        if country in ["US", "CA", "AU"]:
            tz = pytz.timezone(region_timezone_[entry["rg"]])
        else:
            tz = pytz.timezone(pytz.country_timezones(country)[0])

        local_t = tz.normalize(utc_t.astimezone(tz))
        local_hour = local_t.hour
        sd.binarize(result, local_hour, 24)

        local_day = local_t.weekday()
        sd.binarize(result, local_day, 7)
    except:
        result.extend([0]*31)

    # Event - country
    sd.add_to_result(result, entry["cc"], countries_)

    # Event - region
    sd.add_to_result(result, entry["rg"], regions_)


def get_header():
    minute = ("minute", 60)
    hour = ("hour", 24)
    day = ("day", 7)
    weekend = ("weekend", 1)
    fri_or_sat = ("fri_or_sat", 1)
    local_hour = ("local_hour", 24)
    local_day = ("local_day", 7)
    country = ("country", len(countries_)+1)
    region = ("region", len(regions_)+1)

    return [minute, hour, day, weekend, fri_or_sat, local_hour, local_day, country, region]
