import pandas_datareader.data as web


def month_delta(start, months):
    # start: 'YYYY-MM'
    # months: integer value
    splits = map(int, start.split('-'))
    
    y, m = splits
    chunk = m + months
    
    quotient = chunk // 12
    remainder = chunk % 12
    
    if remainder == 0:
        quotient -= 1
        remainder = 12
    
    new_y = str(y + quotient)
    new_m = '0' + str(remainder) if remainder < 10 else str(remainder)
    
    return new_y + '-' + new_m


def enumerate_months(start, months):
    periods = [start]
    for i in range(1, months):
        periods.append(month_delta(start, i))
    return periods


def get_sp500_index(invest_start, invest_period):
    start = month_delta(invest_start, -1)
    end = month_delta(invest_start, invest_period)
    sp500 = web.get_data_yahoo('^GSPC', start=start, end=end, interval='m')['Adj Close']
    sp500 = sp500.pct_change().dropna()
    sp500.index = sp500.index.to_period('M')
    return sp500