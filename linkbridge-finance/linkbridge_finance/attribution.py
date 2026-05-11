"""
绩效归因与因子分析模块

Brinson 归因（配置效应 + 选择效应 + 交互效应）、
Fama-French 因子暴露估计
"""

import numpy as np
import pandas as pd


def brinson_attribution(
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    portfolio_sector_returns: np.ndarray,
    benchmark_sector_returns: np.ndarray,
    benchmark_total_return: float,
) -> dict:
    """
    Brinson 归因模型 — 将超额收益分解为配置效应、选择效应和交互效应。

    配置效应 (Allocation): (w_p - w_b) × (R_b_sector - R_b_total)
    选择效应 (Selection):  w_b × (R_p_sector - R_b_sector)
    交互效应 (Interaction):  (w_p - w_b) × (R_p_sector - R_b_sector)

    Parameters:
        portfolio_weights:          组合中各类资产权重
        benchmark_weights:          基准中各类资产权重
        portfolio_sector_returns:   组合中各类资产收益率
        benchmark_sector_returns:   基准中各类资产收益率
        benchmark_total_return:     基准总收益率

    Returns:
        {
            "allocation_effect": {sector: value},
            "selection_effect": {sector: value},
            "interaction_effect": {sector: value},
            "total_excess": float,
        }
    """
    allocation = portfolio_weights - benchmark_weights
    sector_excess_return = benchmark_sector_returns - benchmark_total_return
    allocation_effect = allocation * sector_excess_return

    selection_effect = benchmark_weights * (portfolio_sector_returns - benchmark_sector_returns)
    interaction_effect = allocation * (portfolio_sector_returns - benchmark_sector_returns)

    total_excess = np.sum(allocation_effect + selection_effect + interaction_effect)

    return {
        "allocation_effect": dict(zip(
            [f"sector_{i}" for i in range(len(allocation_effect))],
            np.round(allocation_effect * 100, 4).tolist(),
        )),
        "selection_effect": dict(zip(
            [f"sector_{i}" for i in range(len(selection_effect))],
            np.round(selection_effect * 100, 4).tolist(),
        )),
        "interaction_effect": dict(zip(
            [f"sector_{i}" for i in range(len(interaction_effect))],
            np.round(interaction_effect * 100, 4).tolist(),
        )),
        "total_excess_pct": round(total_excess * 100, 4),
    }


def factor_exposure(
    stock_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """
    因子暴露估计 — 通过多元线性回归估计股票对各因子的暴露（Beta）。

    R_i = α + β₁F₁ + β₂F₂ + ... + ε

    Parameters:
        stock_returns:  股票日收益率序列
        factor_returns: 因子日收益率 DataFrame（每列一个因子）

    Returns:
        {"alpha": float, "betas": dict, "r_squared": float, "residual_vol": float}
    """
    import statsmodels.api as sm

    aligned = pd.concat([stock_returns, factor_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 20:
        return {"error": "数据不足，至少需要 20 个观测值"}

    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return {
        "alpha": round(float(model.params["const"]) * 252 * 100, 4),  # 年化 % alpha
        "betas": {col: round(float(model.params[col]), 4) for col in factor_returns.columns},
        "r_squared": round(float(model.rsquared), 4),
        "residual_volatility": round(float(model.resid.std() * np.sqrt(252)) * 100, 2),
    }


def style_analysis(
    stock_returns: pd.Series,
    style_returns: pd.DataFrame,
) -> dict:
    """
    Style Analysis (Sharpe 1992) — 带约束的二次规划，权重 [0, 1] 且和为 1。

    用于推断基金/组合的风格暴露（如大盘/小盘/价值/成长等）。

    Parameters:
        stock_returns:  标的收益率序列
        style_returns:  风格指数收益率 DataFrame

    Returns:
        {"weights": dict, "r_squared": float, "tracking_error": float}
    """
    if stock_returns.empty or style_returns.empty:
        return {"error": "无有效数据"}
    aligned = pd.concat([stock_returns, style_returns], axis=1).dropna()
    if aligned.empty:
        return {"error": "无有效数据"}

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    n_styles = X.shape[1]

    def objective(w):
        residual = y - np.dot(X, w)
        return float(np.sum(residual ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_styles

    from scipy.optimize import minimize

    result = minimize(objective, np.ones(n_styles) / n_styles, bounds=bounds, constraints=constraints)
    w = result.x

    fitted = np.dot(X, w)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "weights": {col: round(float(w[i]), 4) for i, col in enumerate(style_returns.columns)},
        "r_squared": round(r2, 4),
        "tracking_error": round(float(np.std(y - fitted) * np.sqrt(252)) * 100, 2),
    }


def information_coefficient(signals: np.ndarray, forward_returns: np.ndarray) -> dict:
    """
    信息系数 (IC) 分析 — 评估因子信号的预测能力。

    Parameters:
        signals:          因子信号值（如估值分位、动量等）
        forward_returns:  对应的未来实际收益率

    Returns:
        {
            "ic": float,              # Pearson 相关系数
            "rank_ic": float,         # Spearman 秩相关系数
            "ic_mean": float,         # IC 均值
            "ic_ir": float,           # IC 信息比率 = mean(IC) / std(IC)
        }
    """
    if len(signals) < 10 or len(forward_returns) < 10:
        return {"error": "数据不足"}

    valid = ~(np.isnan(signals) | np.isnan(forward_returns))
    s = signals[valid]
    f = forward_returns[valid]

    if len(s) < 10:
        return {"error": "有效数据不足"}

    from scipy import stats

    ic = float(np.corrcoef(s, f)[0, 1])
    rank_ic, _ = stats.spearmanr(s, f)

    return {
        "ic": round(ic, 4),
        "rank_ic": round(float(rank_ic), 4),
    }
