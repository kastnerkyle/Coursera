#!/usr/bin/env python

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

def find_events(ls_symbols, d_data):
    df_close = d_data['actual_close']
    ts_market = df_close['SPY']

    df_events = copy.deepcopy(df_close)
    df_events *= np.NAN

    ldt_timestamps = df_close.index

    threshold = 9
    def drop_below(x):
        return (x[0] >= threshold) & (x[1] < threshold)
    #Look at the absolute changes in the market
    stock_change = pd.rolling_apply(df_close, 2, drop_below)
    for k in stock_change.columns:
        df_events[k].ix[stock_change[k] ==True] = 1

    """
    Original tutorial code
    #Look at the percent changes in the market, and set values if both criteria match
    #Market above 2%
    #Stock below 3%
    stock_change = df_close.pct_change(periods=1)
    market_change = ts_market.pct_change(periods=1)
    for k in stock_change.columns:
        df_events[k].ix[(market_change >= 0.02) & (stock_change[k] <= -0.03)] = 1
    """

    return df_events

if __name__ == "__main__":
    dt_start = dt.datetime(2008,1,1)
    dt_end = dt.datetime(2009,12,31)
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Open the dataset and read in the closing price
    dataobj = da.DataAccess('Yahoo')
    #ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    ls_symbols = dataobj.get_symbols_from_list('sp5002008')
    ls_symbols.append('SPY')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    df_events = find_events(ls_symbols, d_data)
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                     s_filename='Homework2Events.pdf', b_market_neutral=True,
                     b_errorbars=True)
