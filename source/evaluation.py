import pandas as pd
import statsmodels.api as sm


def expected_return(returns, interval):
    if interval.upper() == 'Y':
        returns = (1 + returns).resample('Y').agg('prod') - 1
    return returns.mean()

def expected_volatility(returns, interval):
    if interval.upper() == 'Y':
        returns = (1 + returns).resample('Y').agg('prod') - 1
    return returns.std()


def sharpe_ratio(returns, risk_free, interval):
    if interval.upper() == 'Y':
        returns = (1 + returns).resample('Y').agg('prod') - 1
        risk_free = (1 + risk_free).resample('Y').agg('prod') - 1
    return (returns - risk_free.values.reshape(-1, 1)).mean() / returns.std()


def alpha_beta(returns, risk_free):
    # the last column of returns must be market index
    y = returns.iloc[:, :-1] - risk_free.values.reshape(-1, 1)
    x = sm.add_constant(returns.iloc[:, -1] - risk_free)
    alphas = {}
    betas = {}
    for col in y.columns:
        model = sm.OLS(y[col], x)
        result = model.fit()
        alphas[col] = result.params.const
        betas[col] = result.params[0]
    return pd.Series(alphas), pd.Series(betas)


def drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    return drawdowns


def conditional_value_at_risk(returns):
    var = returns.quantile(0.05)
    cvar = {}
    for col in returns.columns:
        cvar[col] = returns[col].loc[returns[col] <= var[col]].mean()
    return pd.Series(cvar)


def summary(returns, risk_free):
    mean = expected_return(returns, 'Y') * 100
    std = expected_volatility(returns, 'M') * 100
    alpha, beta = alpha_beta(returns, risk_free)
    alpha *= 100
    sharpe = sharpe_ratio(returns, risk_free, 'M')
    mdd = -drawdown(returns).min() * 100
    var = -returns.quantile(0.05) * 100
    cvar = -conditional_value_at_risk(returns) * 100
    summary = pd.DataFrame({
        'Annual return': mean.map('{:.2f}%'.format),
        'Alpha': alpha.map('{:.2f}%'.format),
        'Beta': beta.map('{:.3f}'.format),
        'Volatility': std.map('{:.2f}%'.format),
        'Sharpe ratio': sharpe.map('{:.3f}'.format),
        'Max draw-down': mdd.map('{:.2f}%'.format),
        'VaR at 95%': var.map('{:.2f}%'.format),
        'CVaR at 95%': cvar.map('{:.2f}%'.format)
    }).T
    # rearrange the order of columns
    return summary[returns.columns]