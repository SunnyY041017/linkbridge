"""
风险收益指标模块

夏普比率 (Sharpe Ratio)、索提诺比率 (Sortino Ratio)、
贝塔系数 (Beta)、Alpha、最大回撤 (Max Drawdown)、VaR / CVaR
"""

import numpy as np
import pandas as pd
from scipy import stats


def daily_returns(prices: pd.Series) -> pd.Series:
    """从价格序列计算日收益率"""
    return prices.pct_change().dropna()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """计算累计收益率"""
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    年化收益率。

    Parameters:
        returns:           日收益率序列
        periods_per_year:  年化交易日数（A股约 244，美股 252）
    """
    if returns.empty:
        return np.nan
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率"""
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """
    夏普比率 — 单位风险的超额收益。

    Sharpe = (R_p - R_f) / σ_p
    """
    if returns.empty:
        return np.nan
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.025,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    索提诺比率 — 仅处罚下行波动的风险调整收益。

    Sortino = (R_p - R_f) / σ_downside
    """
    if returns.empty:
        return np.nan
    excess = returns - target_return / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    最大回撤 — 从历史峰值到谷底的最大跌幅。

    Returns:
        负数，如 -0.25 表示最大亏损 25%
    """
    if returns.empty:
        return np.nan
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """最大回撤持续天数"""
    if returns.empty:
        return 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    is_dd = cumulative < running_max

    max_dur = 0
    cur_dur = 0
    for v in is_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar 比率 — 年化收益 / 最大回撤"""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(ann_ret / mdd)


def beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta 系数 — 资产相对于基准的系统性风险。

    β = Cov(R_i, R_m) / Var(R_m)
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return 0.0
    return float(cov / var)


def alpha(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """
    Jensen's Alpha — 超额收益中不能被 Beta 解释的部分。

    α = R_p - [R_f + β(R_m - R_f)]
    """
    b = beta(stock_returns, benchmark_returns)
    ann_stock = annualized_return(stock_returns, periods_per_year)
    ann_bench = annualized_return(benchmark_returns, periods_per_year)
    return float(ann_stock - (risk_free + b * (ann_bench - risk_free)))


def information_ratio(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    信息比率 — 跟踪误差每单位的超额收益。

    IR = mean(R_p - R_m) / std(R_p - R_m) × √252
    """
    excess = stock_returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def tracking_error(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """跟踪误差 — 超额收益的标准差"""
    excess = stock_returns - benchmark_returns
    return float(excess.std() * np.sqrt(periods_per_year))


def value_at_risk_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    历史模拟法 VaR。

    Returns:
        正数表示损失，如 0.03 表示在给定置信度下最大日亏损 3%
    """
    if returns.empty:
        return np.nan
    return float(-np.percentile(returns, (1 - confidence) * 100))


def value_at_risk_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    参数法 VaR（假设正态分布）。

    VaR_α = -(μ - z_α × σ)
    """
    if returns.empty:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu - z * sigma))


def value_at_risk_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sim: int = 10000,
    horizon: int = 1,
    seed: int = 42,
) -> float:
    """
    蒙特卡洛模拟 VaR。

    Parameters:
        horizon: 预测天数（1=日 VaR, 10=十日 VaR）
    """
    if returns.empty:
        return np.nan
    np.random.seed(seed)
    mu = returns.mean()
    sigma = returns.std()
    sim_returns = np.random.normal(mu, sigma, n_sim) * np.sqrt(horizon)
    return float(-np.percentile(sim_returns, (1 - confidence) * 100))


def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    CVaR (Expected Shortfall) — 超过 VaR 的平均损失。

    CVaR_α = -E[R | R < -VaR_α]
    """
    if returns.empty:
        return np.nan
    var_val = value_at_risk_historical(returns, confidence)
    tail = returns[returns < -var_val]
    if tail.empty:
        return var_val
    return float(-tail.mean())


def downside_risk(returns: pd.Series, target: float = 0.0) -> float:
    """下行风险 — 收益率低于目标值时的标准差"""
    downside = returns[returns < target]
    if downside.empty:
        return 0.0
    return float(downside.std())


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index — 衡量回撤深度和持续时间"""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_pct = (cumulative - running_max) / running_max
    return float(np.sqrt(np.mean(drawdown_pct ** 2)))
