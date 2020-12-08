import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import month_delta, get_sp500_index
from models import MeanVariance, Momentum, ResidualMomentum
from backtest import backtest
from visualize import plot_annualized_returns, plot_cumulative_returns
from visualize import plot_drawdowns, plot_return_distributions
from evaluation import summary


if __name__ == '__main__':
    print('**************************** Portfolio optimization ****************************')
    print('* 1. Strategies ---------> momentum, residual momentum, gmv                    *')
    print('* 2. Invest start -------> YYYY-MM, ex) 2003-02                                *')
    print('* 3. Invest period ------> M, ex) 203                                          *')
    print('* 4. Rebalance period ---> M, ex) 3                                            *')
    print('********************************************************************************')
    strategies = input('1.Strategies: ')
    invest_start = input('2.Invest start: ')
    invest_period = int(input('3.Invest period: '))
    rebalance_period = int(input('4.Rebalance period: '))
    print('****************************** Optimization begin ******************************')
    # exception
    if (len(invest_start.split('-')) != 2):
        print('Invalid invest_start format, must be "YYYY-MM".')
        exit(1)
    year, month = map(int, invest_start.split('-'))
    if ((year == 2003 and month <= 1) or (year < 2003 and month <= 12)):
        print('Inavalid invset_start, must be after 2003-01')
        exit(1)
    year, month = map(int, month_delta(invest_start, invest_period).split('-'))
    if ((year == 2020 and month >= 11) or (year > 2020 and month >= 1)):
        print('Invalid invest_period, the last month of the investment period must be before 2020-11')
        exit(1)
    # load S&P500 from 2000-02 ~ 2020-10
    sp500 = pd.read_excel('../data/sp500.xlsx', index_col=0, skip_blank_lines=False)
    sp500 = sp500.drop(['ABK', 'CHK', 'DXC', 'GL', 'J', 'LHX', 'RTX', 'SBL', 'TT', 'WELL'], axis=1)
    sp500 = sp500.pct_change().iloc[1:]
    sp500.index = sp500.index.to_period('M')
    # load fama-french three factor data
    ff3 = pd.read_excel('../data/ff3.xlsx', index_col=0)
    ff3.index = ff3.index.to_period('M')
    # define models
    models = {}
    strategies = [x.strip() for x in strategies.split(',')]
    for key in strategies:
        key = key.lower()
        if key == 'momentum':
            models['Momentum'] = Momentum()
        elif key == 'residual momentum':
            
            models['ResidualMomentum'] = ResidualMomentum(ff3)
        elif key == 'gmv':
            models['GMV'] = MeanVariance()
        else:
            print('Invalid strategies.')
            exit(1)
    # backtest results
    assets = {}
    returns = {}
    for key, model in models.items():
        print(f'Optimizing {key}...')
        data = sp500
        if key == 'Momentum':
            lookback_period = 12
        elif key == 'ResidualMomentum':
            lookback_period = 36
        else:
            lookback_period = 36
        start = time.time()
        a, r = backtest(
            model=model,
            invest_start=invest_start,
            invest_period=invest_period,
            lookback_period=lookback_period,
            rebalance_period=rebalance_period,
            historical_data=data
        )
        end = time.time()
        print(f'Optimization took: {end - start:8.4f}sec')
        assets[key] = a
        returns[key] = r
    pfo_rets = pd.DataFrame(returns,
                       index=pd.period_range(invest_start, freq='M', periods=invest_period))
    # get S&P500 index
    pfo_rets['S&P500'] = get_sp500_index(invest_start, invest_period)
    print('***************************** Optimization summary *****************************')
    invest_end = month_delta(invest_start, invest_period - 1)
    print(summary(pfo_rets, ff3[invest_start:invest_end]['RF']))
    # visualize result
    print('**************************** Optimization finished *****************************')
    plot_annualized_returns(pfo_rets, 1)
    plot_cumulative_returns(pfo_rets, 2)
    plot_drawdowns(pfo_rets, 3)
    plot_return_distributions(pfo_rets, 4)
    plt.show()