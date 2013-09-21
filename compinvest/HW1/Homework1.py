#!/usr/bin/env python

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

NUM_TRADING_DAYS = 252

def simulate(dt_start, dt_end, ls_symbols, ls_allocation):
    # Formatting the date timestamps
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Open the dataset and read in the closing price
    all_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_keys = ['close']
    c_dataobj = da.DataAccess('Yahoo')
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Calculate the portfolio value
    d_normal = d_data['close'].values.copy() / d_data['close'].values[0,:]
    alloc = np.array(ls_allocation).reshape(4,1)
    portVal = np.dot(d_normal, alloc)

    # Calculate the daily returns
    dailyVal = portVal.copy()
    tsu.returnize0(dailyVal)

    # Calculate statistics
    daily_ret = np.mean(dailyVal)
    vol = np.std(dailyVal)
    sharpe = np.sqrt(NUM_TRADING_DAYS) * daily_ret / vol
    cum_ret = portVal[portVal.shape[0] -1][0]

    return [vol, daily_ret, sharpe, cum_ret]


def print_simulate(dt_start, dt_end, ls_symbols, ls_allocation):
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, ls_allocation)
    print "Start Date: ", dt_start
    print "End Date: ", dt_end
    print "Symbols: ", ls_symbols
    print "Optimal Allocations:\n", ls_allocation
    print "Sharpe Ratio: ", sharpe
    print "Volatility (stdev): ", vol
    print "Average Daily Return: ", daily_ret
    print "Cumulative Return: ", cum_ret


#TODO: gradient ascent
def optimal_allocation_4(dt_start, dt_end, ls_symbols):
    #Generate all valid allocations from 0 to 100% by 10%
    valid_alloc = pd.DataFrame(filter(lambda x: sum(x) == 1,
                               product(np.linspace(0,1,11), repeat=4)),
                               columns=map(str,range(1,5)))

    #Simulate every valid allocation and store as a DataFrame
    def run_sim(alloc): return simulate(dt_start, dt_end, ls_symbols, alloc)
    sim_alloc = valid_alloc.apply(run_sim, axis=1)
    sim_alloc.columns = ["vol", "daily_ret", "sharpe", "cum_ret"]

    #Get the allocation with the highest sharpe ratio
    max_alloc = valid_alloc.ix[sim_alloc["sharpe"].idxmax()]
    return max_alloc


if __name__ == "__main__":
    dt_start = dt.datetime(2010,1,1)
    dt_end = dt.datetime(2010,12,31)
    ls_symbols = ['BRCM', 'TXN', 'AMD', 'ADI']
    max_alloc = optimal_allocation_4(dt_start, dt_end, ls_symbols)
    print_simulate(dt_start, dt_end, ls_symbols, max_alloc)
