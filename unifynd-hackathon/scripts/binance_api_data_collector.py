from __future__ import print_function

import argparse
import datetime
import multiprocessing
import os
import sys
import textwrap
import time
from datetime import timedelta
from functools import partial
from itertools import product
from multiprocessing.dummy import Pool

import numpy as np
import pandas as pd
import requests
from datadiff import diff

url = "https://api.binance.com"


def get_daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


def klines(symbol, interval="15m", starttime=None, endtime=None, limit=500):
    api_url = url + "/api/v1/klines"
    querystring = {"symbol": symbol, "interval": interval, "limit": limit}
    if endtime:
        endtime = int(datetime.datetime.fromtimestamp(starttime / 1000.0).replace(hour=23, minute=45, second=0).timestamp()) * 1000
    # print(symbol, "---", interval, "---", starttime, "---", endtime, "---", limit)
    if starttime:
        querystring.update(startTime=starttime, endTime=endtime)
    # print(querystring)
    headers = {
        'Cache-Control': "no-cache",
        'Postman-Token': "f02e3909-623d-8fbe-9748-0cf315d75d10"
        }

    try:
        response = requests.request("GET", api_url, headers=headers, params=querystring)

        # print(response.status_code)
        # print(datetime.datetime.fromtimestamp(int(starttime)), response.status_code)
        data = response.json()

        if response.status_code != 200:
            # print("Retrying")
            time.sleep(2)
            return klines(symbol, interval, starttime, endtime, limit)

    except Exception as e:
        print("In error")
        print(e)
        data = []
        iszip = False
    if starttime:
        day = datetime.datetime.fromtimestamp(starttime / 1000.0).day
        month = datetime.datetime.fromtimestamp(starttime / 1000.0).month
        year = datetime.datetime.fromtimestamp(starttime / 1000.0).year
        print(len(data))
        return (day, month, year, data)
    else:
        return (0, 0, 0, data)


def multi_run_wrapper(args):
   return klines(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""\
                                                                   Fetch Binance Klines Values.
                                                                   -----------------------------------------
                                                                   Example:
                                                                   $ python binance_api_data_collector.py --symbol BTCUSDT --interval 15m --frm 1/1/2016 --to 6/6/2018

                                                                   $ intervals --> m -> minutes; h -> hours; d -> days; w -> weeks; M -> months

                                                                   $ 1m   ,3m   ,5m   ,15m   ,30m   ,1h   ,2h   ,4h   ,6h   ,8h   ,12h   ,1d   ,3d   ,1w   ,1M

                                                                   """
                                                                )
                                    )

    parser.add_argument("--symbol", dest="symbol", type=str, help="Enter valid symbol")
    parser.add_argument("--interval", dest="interval", type=str, help="Enter valid interval")
    parser.add_argument("--frm", dest="frm", type=str, help="Pass Date from")
    parser.add_argument("--to", dest="to", type=str, help="Pass Date to")
    parser.add_argument("--limit", dest="limit", type=str, help="Max No. of records")

    args = parser.parse_args()
    print(args)
    symbol = args.symbol
    interval = args.interval
    duration_frm = args.frm
    duration_to = args.to
    limit = args.limit

    if len(sys.argv) > 1:
        start = time.time()

        arg_data = []
        if duration_frm:
            duration_frm = datetime.datetime.strptime(duration_frm, "%d/%m/%Y")
            duration_to = datetime.datetime.strptime(duration_to, "%d/%m/%Y")

            pool_size = (duration_to - duration_frm).days

            startdate = [int(dt.timestamp()) * 1000 for dt in get_daterange(duration_frm, duration_to)]
            enddate = [int((dt + timedelta(days=1)).timestamp()) * 1000 for dt in get_daterange(duration_frm, duration_to)]

            # print(startdate)
            # print(enddate)
            pool = Pool(pool_size)
            for n in range(len(startdate) - 1):
                arg_data.append([symbol, interval, startdate[n], enddate[n]])
            # print(arg_data)
            response_data = pool.map(multi_run_wrapper, arg_data)
            pool.close()
            pool.join()
        else:
            response_data = klines(symbol, interval)
        # klines_partial = partial(symbol, interval, klines, klines, 500)

        response_data = sorted(response_data)

        final_data = []
        for pool_data in response_data:
            for data in pool_data[3]:
                final_data.append(data)

        columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
            "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
        df = pd.DataFrame(final_data, columns=columns).sort_values(by="Open time", ascending=True)
        # df.to_csv("../data/%s-%s-%s.csv"%(symbol, str(duration_frm).replace('00:00:00',''),str(duration_to).replace('00:00:00','')), index=False)
        df.to_csv("../data/%s-raw.csv" % symbol, index = False)
        print("Total time taken --> " + str(time.time() - start))
    else:
        parser.print_help()
