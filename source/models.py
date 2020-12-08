import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
from scipy.optimize import minimize
from tqdm import tqdm


class MeanVariance:
    def optimize(self, data):
        # Warning!: data must be monthly returns
        # set initial weight
        n_asset = len(data.columns)
        initial_w = np.ones(n_asset) / n_asset
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = ((0, 1),) * n_asset
        # calculate sigma of returns and annualize
        sigma = data.cov() * 12
        # optimize weights s.t. constraints & bounds
        optim = minimize(
            lambda w : np.dot(w, np.dot(sigma, w)),
            initial_w,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        return optim.x

# class MeanVariance:
#     def optimize(self, data):
#         # Warning!: data must be monthly returns
#         # get initial weight
#         n_asset = len(data.columns)
#         w = cp.Variable(n_asset)
#         constraints = [cp.sum(w) == 1, w >= 0]
#         # calculate sigma of returns and annualize
#         sigma = data.cov() * 12
#         print(w.shape)
#         print(sigma.shape)
#         # optimize weights s.t. constriants
#         problem = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints)
#         problem.solve(solver=cp.ECOS)
#         return w.value

class Momentum:
    def optimize(self, data):
        # Warning!: data must be monthly returns
        data = data.iloc[:-1]
        total_rets = (1 + data).product(axis=0) - 1
        total_rets = total_rets.sort_values(ascending=False)
        # take top 10% of the assets
        n_asset = len(total_rets.index) // 10
        return list(total_rets.index[:n_asset])


class ResidualMomentum:
    def __init__(self, factor_data):
        self.factor_data = factor_data
        
    def optimize(self, data):
        # Warning!: data must be monthly returns
        # get start & end of the lookback period
        # drop the most recent one month data
        start = data.index[0]
        end = data.index[-2]
        # select the data within the period
        data = data[start:end]
        factor_data = self.factor_data[start:end]
        # prepare data for linear regression
        y = data - factor_data.iloc[:, -1].values.reshape(-1, 1)
        x = sm.add_constant(factor_data.iloc[:, :-1])
        # calculate residual returns
        resid_rets = {}
        for asset in data.columns:
            model = sm.OLS(y[asset], x)
            result = model.fit()
            resid_rets[asset] = result.resid.values[-12:]
            resid_rets[asset] += result.params[0]
        resid_rets = pd.DataFrame(resid_rets)
        resid_rets = resid_rets.sum() / resid_rets.std()
        resid_rets = resid_rets.sort_values(ascending=False)
        # take top 10% of the assets
        n_asset = len(resid_rets.index) // 10
        return list(resid_rets.index[:n_asset])