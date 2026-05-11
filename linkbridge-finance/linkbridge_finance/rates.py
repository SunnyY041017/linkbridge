"""
利率风险指标模块

久期 (Duration)、修正久期 (Modified Duration)、凸性 (Convexity)
用于衡量固定收益类资产对利率变动的敏感性。
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def macaulay_duration(
    cashflows: np.ndarray,
    times: np.ndarray,
    ytm: float,
) -> float:
    """
    麦考利久期 — 现金流的加权平均回收时间。

    Parameters:
        cashflows: 各期现金流
        times:     各期距离现在的时间（年）
        ytm:       到期收益率（年化）

    Returns:
        麦考利久期（年）
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)
    pv = cashflows / (1 + ytm) ** times
    total_pv = pv.sum()
    if total_pv == 0:
        return 0.0
    return float(np.sum(times * pv) / total_pv)


def modified_duration(
    cashflows: np.ndarray,
    times: np.ndarray,
    ytm: float,
) -> float:
    """
    修正久期 — 收益率变动 1% 时，债券价格的百分比变动。

    D_mod = D_mac / (1 + ytm / n)
    """
    d_mac = macaulay_duration(cashflows, times, ytm)
    return d_mac / (1 + ytm)


def effective_duration(
    price_func,
    ytm: float,
    delta_y: float = 0.0001,
) -> float:
    """
    有效久期 — 适用于含权债券，通过数值求导计算。

    D_eff = (P_down - P_up) / (2 * P_0 * Δy)

    Parameters:
        price_func: 给定 YTM 返回债券价格的函数
        ytm:        当前到期收益率
        delta_y:    收益率扰动幅度

    Returns:
        有效久期
    """
    p0 = price_func(ytm)
    p_up = price_func(ytm + delta_y)
    p_down = price_func(ytm - delta_y)
    if p0 == 0:
        return 0.0
    return float((p_down - p_up) / (2 * p0 * delta_y))


def convexity(
    cashflows: np.ndarray,
    times: np.ndarray,
    ytm: float,
) -> float:
    """
    凸性 — 久期之外的第二阶利率敏感性。

    C = Σ [CF_t * t * (t+1) / (1+ytm)^(t+2)] / P

    Returns:
        凸性值
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)
    pv = cashflows / (1 + ytm) ** times
    total_pv = pv.sum()
    if total_pv == 0:
        return 0.0
    convex = np.sum(cashflows * times * (times + 1) / (1 + ytm) ** (times + 2))
    return float(convex / total_pv)


def effective_convexity(
    price_func,
    ytm: float,
    delta_y: float = 0.0001,
) -> float:
    """
    有效凸性 — 数值求导法，适用于含权债券。

    C_eff = (P_down + P_up - 2*P_0) / (P_0 * (Δy)²)
    """
    p0 = price_func(ytm)
    p_up = price_func(ytm + delta_y)
    p_down = price_func(ytm - delta_y)
    if p0 == 0:
        return 0.0
    return float((p_down + p_up - 2 * p0) / (p0 * delta_y ** 2))


def price_impact_duration(
    modified_dur: float,
    convex: float,
    delta_yield: float,
) -> float:
    """
    利率变动对债券价格的近似影响（含凸性修正）。

    ΔP/P ≈ -D_mod × Δy + 0.5 × C × (Δy)²

    Returns:
        价格变动百分比（如 -0.02 表示下跌 2%）
    """
    return float(-modified_dur * delta_yield + 0.5 * convex * delta_yield ** 2)


def bond_price(
    face_value: float,
    coupon_rate: float,
    periods: int,
    ytm: float,
    freq: int = 1,
) -> float:
    """
    标准付息债券定价。

    Parameters:
        face_value:  面值
        coupon_rate: 票面利率（年化）
        periods:     剩余付息期数
        ytm:         到期收益率（年化）
        freq:        年付息次数（1=年付, 2=半年付）
    """
    coupon = face_value * coupon_rate / freq
    t = np.arange(1, periods + 1) / freq
    pv_coupons = np.sum(coupon / (1 + ytm / freq) ** (t * freq))
    pv_face = face_value / (1 + ytm / freq) ** (periods)
    return float(pv_coupons + pv_face)


def bond_portfolio_metrics(
    bonds: list[dict],
    yields: list[float],
) -> dict:
    """
    债券组合久期与凸性。

    bonds: 每个元素含 {'market_value', 'cashflows', 'times'}
    yields: 各债券对应 YTM

    Returns:
        {'portfolio_duration', 'portfolio_convexity', 'weighted_ytm'}
    """
    total_mv = sum(b["market_value"] for b in bonds)
    dur = 0.0
    conv = 0.0
    w_ytm = 0.0

    for b, y in zip(bonds, yields):
        w = b["market_value"] / total_mv
        dur += w * macaulay_duration(np.array(b["cashflows"]), np.array(b["times"]), y)
        conv += w * convexity(np.array(b["cashflows"]), np.array(b["times"]), y)
        w_ytm += w * y

    return {
        "portfolio_duration": round(dur, 4),
        "portfolio_convexity": round(conv, 4),
        "weighted_ytm": round(w_ytm, 6),
    }
