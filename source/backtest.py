import numpy as np
from tqdm import tqdm
from utils import month_delta, enumerate_months
from models import MeanVariance


def backtest(model,
             invest_start,
             invest_period,
             lookback_period,
             rebalance_period,
             historical_data):
    # get invest months
    invest_months = enumerate_months(invest_start, invest_period)
    # optim_value: if model == MeanVariance, then weights
    #              if model == Momentum or ResidualMomentum, then assets
    optim_values = []
    returns = []
    # iterate through invest months
    for i, invest_month in tqdm(enumerate(invest_months)):
        # if rebalance period, optimize model
        if i % rebalance_period == 0:
            # calculate start & end of the lookback period
            start = month_delta(invest_month, -lookback_period)
            end = month_delta(invest_month, -1)
            # calculate the month right before the next rebalance month
            # start ~ end2: lookback_period + holding period
            end2 = month_delta(end, rebalance_period)
            # drop all columns with null values
            lookback_data = historical_data[start:end2].replace(0, np.nan).dropna(axis=1)
            # select the data within the lookback period
            lookback_data = lookback_data[start:end]
            optim_value = model.optimize(lookback_data)
        optim_values.append(optim_value)
        if isinstance(model, MeanVariance):
            asset_ret = historical_data[invest_month][lookback_data.columns]
            returns.append(np.dot(asset_ret, optim_value)[0])
        else:
            asset_ret = historical_data[invest_month][optim_value]
            returns.append((asset_ret.sum(axis=1) / len(optim_value))[0])
    return np.array(optim_values), np.array(returns)