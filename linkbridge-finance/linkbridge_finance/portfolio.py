"""
投资组合分析模块

相关性矩阵、协方差矩阵、有效前沿 (Efficient Frontier)、
最小方差组合、最大夏普比率组合（切线组合）、风险平价
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """从价格 DataFrame（每列为一只资产）计算日收益率矩阵"""
    return prices.pct_change().dropna()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """计算资产收益率的相关性矩阵"""
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """计算资产收益率的协方差矩阵（年化）"""
    return returns.cov() * 252


def portfolio_return(weights: np.ndarray, annual_returns: np.ndarray) -> float:
    """组合年化预期收益率"""
    return float(np.dot(weights, annual_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """组合年化波动率"""
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def portfolio_sharpe(
    weights: np.ndarray,
    annual_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free: float = 0.025,
) -> float:
    """组合夏普比率"""
    ret = portfolio_return(weights, annual_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return float((ret - risk_free) / vol)


def portfolio_variance_decomposition(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """组合风险贡献分解 — 每项资产对组合总方差的边际贡献"""
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return np.zeros(len(weights))
    marginal = np.dot(cov_matrix, weights)  # 边际风险贡献
    component = weights * marginal / port_vol  # 风险贡献（归一化为波动率单位）
    return component


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """等权重组合"""
    return np.ones(n_assets) / n_assets


def minimum_variance_portfolio(
    cov_matrix: np.ndarray,
    bounds: tuple = (0.0, 1.0),
) -> dict:
    """
    最小方差组合 — 在给定协方差矩阵下寻找最小波动率的权重。

    Returns:
        {"weights": np.ndarray, "volatility": float, "return": float}
    """
    n = len(cov_matrix)
    init_guess = equal_weight_portfolio(n)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bnds = [bounds] * n

    def objective(w):
        return portfolio_volatility(w, cov_matrix)

    result = minimize(objective, init_guess, method="SLSQP", bounds=bnds, constraints=constraints)
    w = result.x
    return {
        "weights": np.round(w, 4),
        "volatility": float(portfolio_volatility(w, cov_matrix)),
        "return": float(portfolio_return(w, np.zeros(n))),
    }


def max_sharpe_portfolio(
    annual_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free: float = 0.025,
    bounds: tuple = (0.0, 1.0),
) -> dict:
    """
    最大夏普比率组合（切线组合）。

    Returns:
        {"weights": np.ndarray, "return": float, "volatility": float, "sharpe": float}
    """
    n = len(cov_matrix)
    init_guess = equal_weight_portfolio(n)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bnds = [bounds] * n

    def objective(w):
        return -portfolio_sharpe(w, annual_returns, cov_matrix, risk_free)

    result = minimize(objective, init_guess, method="SLSQP", bounds=bnds, constraints=constraints)
    w = result.x
    return {
        "weights": np.round(w, 4),
        "return": float(portfolio_return(w, annual_returns)),
        "volatility": float(portfolio_volatility(w, cov_matrix)),
        "sharpe": float(portfolio_sharpe(w, annual_returns, cov_matrix, risk_free)),
    }


def efficient_frontier(
    annual_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_points: int = 50,
    bounds: tuple = (0.0, 1.0),
) -> list[dict]:
    """
    计算有效前沿 — 不同目标收益下的最小波动率组合。

    Returns:
        [{"return": float, "volatility": float, "weights": list}, ...]
    """
    n = len(cov_matrix)
    # 确定收益范围
    min_ret = float(np.min(annual_returns))
    max_ret = float(np.max(annual_returns))
    target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier = []
    bnds = [bounds] * n
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, annual_returns) - t},
        ]
        init_guess = equal_weight_portfolio(n)

        result = minimize(
            lambda w: portfolio_volatility(w, cov_matrix),
            init_guess,
            method="SLSQP",
            bounds=bnds,
            constraints=constraints,
        )
        if result.success:
            frontier.append({
                "return": round(target * 100, 2),
                "volatility": round(float(portfolio_volatility(result.x, cov_matrix)) * 100, 2),
                "weights": np.round(result.x, 4).tolist(),
            })

    return frontier


def risk_parity_portfolio(
    cov_matrix: np.ndarray,
    bounds: tuple = (0.0, 1.0),
) -> dict:
    """
    风险平价组合 — 每项资产对组合总风险的贡献相等。

    目标：最小化各资产风险贡献的方差。
    """
    n = len(cov_matrix)
    init_guess = equal_weight_portfolio(n)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bnds = [bounds] * n

    def objective(w):
        port_vol = portfolio_volatility(w, cov_matrix)
        if port_vol == 0:
            return 1e10
        marginal = np.dot(cov_matrix, w)
        risk_contrib = w * marginal / port_vol
        target_contrib = port_vol / n
        return float(np.sum((risk_contrib - target_contrib) ** 2))

    result = minimize(objective, init_guess, method="SLSQP", bounds=bnds, constraints=constraints)
    w = result.x
    return {
        "weights": np.round(w, 4),
        "volatility": float(portfolio_volatility(w, cov_matrix)),
        "risk_contribution": np.round(portfolio_variance_decomposition(w, cov_matrix), 4).tolist(),
    }


def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    分散化比率 — 加权平均波动率 / 组合波动率。
    值 > 1 表示存在分散化收益。
    """
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return 1.0
    indiv_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, indiv_vols)
    if port_vol == 0:
        return 0.0
    return float(weighted_avg_vol / port_vol)


def portfolio_beta(weights: np.ndarray, asset_betas: np.ndarray) -> float:
    """组合加权 Beta"""
    return float(np.dot(weights, asset_betas))
