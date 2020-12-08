import matplotlib.pyplot as plt
from evaluation import drawdown


def move_figure(position):
    # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib  
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    py = mgr.canvas.height()
    px = mgr.canvas.width()

    d = 10 # width of the window border in pixels
    if position == "top-left":
        # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
        mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "top-right":
        mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "bottom-left":
        mgr.window.setGeometry(d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)
    elif position == "bottom-right":
        mgr.window.setGeometry(px/2 + d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)


def plot_annualized_returns(returns, fig_num):
    # Warning!: returns must be monthly data
    annual_returns = (1 + returns).resample('Y').agg('prod') - 1
    fig = plt.figure(num=fig_num, figsize=(8, 4))
    #move_figure('top-left')
    ax = fig.add_subplot(111)
    ax.set_title('Annualized portfolio returns', fontsize=12, pad=10)
    ax.set_xlabel('Year', fontsize=10, labelpad=5)
    ax.set_ylabel('Returns', fontsize=10, labelpad=5)
    annual_returns.plot(kind='bar', ax=ax)


def plot_cumulative_returns(returns, fig_num):
    # Warning!: returns must be monthly data
    cumulative_returns = (1 + returns).cumprod(axis=0) - 1
    fig = plt.figure(num=fig_num, figsize=(8, 4))
    #move_figure('top-right')
    ax = fig.add_subplot(111)
    ax.set_title('Cumulative monthly portfolio returns', fontsize=12, pad=10)
    ax.set_xlabel('Month', fontsize=10, labelpad=5)
    ax.set_ylabel('Returns', fontsize=10, labelpad=5)
    cumulative_returns.plot(kind='line', ax=ax)


def plot_drawdowns(returns, fig_num):
    # Warning!: returns must be monthly data
    drawdowns = drawdown(returns)
    fig = plt.figure(num=fig_num, figsize=(8, 4))
    #move_figure('bottom-left')
    ax = fig.add_subplot(111)
    ax.set_title('Drawdowns', fontsize=12, pad=10)
    ax.set_xlabel('Month', fontsize=10, labelpad=5)
    ax.set_ylabel('Returns', fontsize=10, labelpad=5)
    drawdowns.plot(kind='line', ax=ax)
    for col in drawdowns.columns:
        ax.fill_between(drawdowns.index, drawdowns[col], alpha=0.5)


def plot_return_distributions(returns, fig_num):
    # Warning!: returns must be monthly data
    fig = plt.figure(num=fig_num, figsize=(8, 4))
    #move_figure('bottom-left')
    ax = fig.add_subplot(111)
    ax.set_title('Monthly portfolio return distribution', fontsize=12, pad=10)
    ax.set_xlabel('Strategies', fontsize=10, labelpad=5)
    ax.set_ylabel('Returns', fontsize=10, labelpad=5)
    returns.plot(kind='box', ax=ax)